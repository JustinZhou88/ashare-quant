"""每周扫描：因子筛选 → 硬规则过滤 → LLM 分析 → 推荐 + 记账。

    python scan_weekly.py                    # 用最新数据扫描（无 key 则 dry-run）
    python scan_weekly.py --date 2026-06-26  # 回溯某一天（只用该日及以前数据）
    python scan_weekly.py --journal          # 只看累计战绩，不扫描

流程（两级漏斗，LLM 放在最后一步）：
  全市场 → 硬规则过滤 → 已验证的反转因子排序 → 前 N 只 → LLM → 推荐 3 只

为什么 LLM 放最后：因子层已经被统计检验过（DSR 0.963 通过），
它筛出来的池子有正期望。LLM 是在一个已经不错的池子里做选择，
而不是从几千只里瞎挑。同时成本降到 1/250。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aq import journal as J
from aq.agents import analysts as AN
from aq.agents.context import build_facts, facts_to_text, market_context
from aq.agents.forecast import AnalogForecaster
from aq.agents.forecast import to_text as forecast_text
from aq.agents.llm import LLM
from aq.agents.rules import Rules, apply_hard_filters, soft_prefs_text
from aq.config import BENCHMARK
from aq.data.altdata import (build_margin_panel, load_margin_many,
                            load_reports, reports_to_factors)
from aq.data.global_ctx import build_global_panel
from aq.data.global_ctx import to_text as global_text
from aq.data.loader import fetch_all_symbols, load_index, load_many
from aq.data.sentiment import collect as collect_sentiment
from aq.data.sentiment import geopolitical_gate
from aq.data.universe import from_file
from aq.engine.panel import build_panel
from aq.strategies.factors import cs_rank

OUT = Path("journal")


# 中周期截面反转的「有效高原」。2804 只全市场、6 折滚动前推实测的样本外夏普：
#   20日 0.34 | 30日 0.92 | 40日 0.60 | 60日 0.62 | 90日 0.87
#   120日 0.52 | 180日 0.33 | 250日 0.54
# 30~120 日整片有效（高原度 0.75），只有两端塌陷 —— 是真信号不是孤峰。
REVERSAL_WINDOWS = (30, 60, 90)


def factor_score(panel, D: dict, n: int = 20) -> pd.DataFrame:
    """截面中周期反转，高原内等权。

    为什么只用这一个因子族（2804 只全市场、6 折前推的实测结论）：

    **加进来的其他因子增量 alpha 全都不显著**，等权只会稀释有效信号：
        两融反转 t=1.60 | 股东户数 t=0.62 | 波动倾斜 t=0.24
    有效因子 IR=0.6 与无效因子等权，组合 IR 约降到 0.6/√2 —— 加一个砍一刀。
    而且它们彼此相关 0.6~0.83，连分散化的好处都拿不到。

    样本外表现对比（年化 / 夏普 / 回撤）：
        本函数 30/60/90 等权   +23.21% / 0.88 / -31.08%   ← 采用
        单挑最优的 30 日       +25.43% / 0.92 / -36.36%   夏普只高 0.04，
                                                          回撤多 5pct，且有选择偏差
        含弱参数 20/60/180     +12.54% / 0.56 / -42.55%
        旧版(两融+价格混合)     +7.17% / 0.39 / -40.29%
        沪深300                -2.27% / -0.04 / -45.60%

    不用两融还有个运维好处：全市场两融覆盖率只有 58.6%，
    用它会让四成股票因为缺数据而系统性地拿不到公平打分。

    真找到第二个**独立且显著**的因子会立刻加回来 —— 问题是目前一个都没找到。
    """
    parts = [cs_rank(-panel.close.pct_change(nn)) for nn in REVERSAL_WINDOWS]
    return sum(parts) / len(parts)


def open_positions(panel, rules_path: str) -> set[str]:
    """模拟盘里还没平仓的股票（含已推荐、待下个交易日执行的）。

    没平仓就不重复推荐同一只票：模拟盘那边本来就会以「已持有，跳过重复建仓」
    拒绝，重复推荐只会白白消耗一份 LLM 分析，并让同一笔头寸在战绩统计里被
    计入多次（虚增或虚减胜率）。平仓后它会自动重新回到候选池。

    读不到持仓就返回空集合 —— 去重是优化不是刚需，不该让扫描因此中断。
    """
    try:
        import yaml

        from aq.paper import simulate
        risk = (yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
                or {}).get("risk") or {}
        res = simulate(panel, risk)
        op = res.open_trades
        op = op() if callable(op) else op
        held = {str(t.symbol).zfill(6) for t in op}
        held |= {str(x.get("symbol")).zfill(6) for x in (res.pending or [])
                 if x.get("symbol")}
        return held
    except Exception as e:                                     # noqa: BLE001
        print(f"  ! 读不到模拟盘持仓，本轮不去重: {type(e).__name__}")
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="扫描基准日，默认最新交易日")
    ap.add_argument("--rules", default="rules.yaml")
    ap.add_argument("--universe", default="universe.txt")
    ap.add_argument("--dry-run", action="store_true", help="强制不调用 API")
    ap.add_argument("--journal", action="store_true", help="只看累计战绩")
    ap.add_argument("--no-llm", action="store_true", help="只出因子候选，不调 LLM")
    ap.add_argument("--provider", default=None,
                    help="gemini(免费) / deepseek / zhipu / qwen，默认按环境变量自动选")
    ap.add_argument("--roles", default="all",
                    help="启用的分析师角色，逗号分隔或 all。"
                         "可选 trend,capital,global_macro,value_trap,risk_screen")
    ap.add_argument("--sentiment", default="polymarket,guba,x_influencer,mediacrawler",
                    help="启用的情绪源，逗号分隔：polymarket,mediacrawler,x_influencer")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    rules = Rules.load(args.rules)

    syms = from_file(args.universe)
    data = load_many(syms, verbose=False)
    panel = build_panel(data)
    idx = load_index(BENCHMARK)

    if args.journal:
        j = J.update_performance(panel, idx)
        print(J.report(j))
        return

    as_of = pd.Timestamp(args.date) if args.date else panel.dates[-1]
    as_of = panel.dates[panel.dates <= as_of][-1]
    print(f"扫描基准日: {as_of.date()}（只使用该日及以前的数据）\n")

    # cache_only：扫描绝不触发下载。补两融数据请单独跑 download_margin.py
    margin = load_margin_many(panel.symbols, start="2016-01-01", cache_only=True)
    mp = build_margin_panel(margin, panel.dates, panel.symbols, lag=1)
    D = {**mp, "close": panel.close, "amount": panel.amount}

    # 分析师研报 -> 评级修正/覆盖度因子（价值陷阱排查用）。缺数据时静默跳过。
    try:
        rep = load_reports(start="2017-01-01")
        if not rep.empty:
            D.update(reports_to_factors(rep, panel.dates, panel.symbols, window=60))
            print(f"  研报数据: {len(rep)} 篇，覆盖 {rep['symbol'].nunique()} 只")
    except Exception as e:                                     # noqa: BLE001
        print(f"  ! 研报数据不可用（价值陷阱分析将缺少依据）: {type(e).__name__}")

    try:
        meta = fetch_all_symbols()
    except Exception:                                          # noqa: BLE001
        meta = None

    # ---------------------------------------------------- 1. 硬规则过滤
    print("【1/4】硬规则过滤")
    passed, reasons = apply_hard_filters(panel, rules, as_of, meta=meta, verbose=True)
    if not passed:
        print("没有股票通过硬规则过滤 —— 检查 rules.yaml 是不是设得太严。")
        return

    # ---------------------------------------------------- 2. 因子排序
    n_cand = int(rules.get("schedule", "n_candidates_to_llm", 20))
    sc = factor_score(panel, D)
    row = sc.loc[as_of, passed].dropna().sort_values(ascending=False)
    held = open_positions(panel, args.rules)
    dup = [x for x in row.index if x in held]
    if dup:
        row = row.drop(index=dup)
        print("  已持仓/待执行，本轮跳过："
              + "、".join(f"{x} {panel.names.get(x, '')}" for x in dup))

    cands = row.head(n_cand).index.tolist()
    print(f"\n【2/4】因子排序：{len(passed)} 只 -> 取前 {len(cands)} 只候选")
    for i, s in enumerate(cands, 1):
        print(f"    {i:2d}. {s} {panel.names.get(s,''):8s} 因子分 {row[s]:.3f}")

    facts = {}
    for i, s in enumerate(cands, 1):
        f = build_facts(panel, D, s, as_of, factor_rank=i, meta=meta)
        if f:
            facts[s] = f
    fact_texts = [facts_to_text(facts[s]) for s in cands if s in facts]

    # 国际环境：隔夜美股/恒生/汇率 + 地缘概率。全部是 A股开盘前已知的信息。
    enabled = tuple(x.strip() for x in args.sentiment.split(",") if x.strip())
    readings = collect_sentiment(str(as_of.date()), enable=enabled, symbols=cands)
    try:
        gp = build_global_panel(panel.dates)
    except Exception as e:                                     # noqa: BLE001
        print(f"  ! 国际环境拉取失败: {type(e).__name__}")
        gp = pd.DataFrame()
    mkt = market_context(idx, as_of)
    gtxt = global_text(gp, as_of, readings) if not gp.empty else ""
    if gtxt:
        mkt = mkt + "\n\n" + gtxt
        print("\n" + gtxt)

    # 地缘熔断：尾部概率抬升时压缩本周推荐数量
    base_n = int(rules.get("schedule", "n_recommendations", 3))
    gated_n, gate_note = geopolitical_gate(readings, base_n)
    if gate_note:
        print(f"\n  ⚠️ {gate_note}")

    (OUT / f"candidates_{as_of.date()}.json").write_text(
        json.dumps({"as_of": str(as_of.date()), "facts": facts},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.no_llm:
        print(f"\n候选信息包已写入 journal/candidates_{as_of.date()}.json（未调 LLM）")
        return

    # ---------------------------------------------------- 3. LLM 分析
    llm = LLM(provider=args.provider, dry_run=args.dry_run)
    tag = "DRY-RUN 占位模式" if llm.dry_run else f"{llm.provider}/{llm.model}"
    print(f"\n【3/4】LLM 分析（{tag}）")
    if llm.dry_run and not args.dry_run:
        print("    ⚠️ 输出为占位内容，不是真实分析。")
        print("    " + llm.missing_key_hint().replace("\n", "\n    "))
    roles = None if args.roles == "all" else tuple(
        x.strip() for x in args.roles.split(",") if x.strip())
    res = AN.analyze(llm, mkt, fact_texts, soft_prefs_text(rules),
                     str(as_of.date()), gated_n, rules.risk, roles=roles)

    # ---------------------------------------------------- 4. 输出 + 记账
    print(f"\n【4/4】本周推荐（基准日 {as_of.date()}）")
    print("=" * 64)
    picks = res.get("picks") or []
    if not picks:
        print("本周不推荐任何标的。")
        print(res.get("note", ""))
    # 历史类比预测：给每只推荐一个有数据支撑的价格区间与持有周期。
    # 刻意不让 LLM 给目标价 —— 它没有模型，只会给虚假精确。
    fc_engine = AnalogForecaster(panel, horizon=15) if picks else None
    for p in picks:
        sym = str(p.get("symbol", "")).zfill(6)
        print(f"\n  {sym} {p.get('name') or panel.names.get(sym,'')}"
              f"   信心 {p.get('score','-')}/10")
        print(f"  理由: {p.get('reason','')}")
        if p.get("risks"):
            print(f"  风险: {'; '.join(p['risks'])}")
        if p.get("entry_note"):
            print(f"  进场: {p['entry_note']}")
        if fc_engine is not None:
            fc = fc_engine.forecast(sym, as_of)
            p["forecast"] = fc
            print(forecast_text(fc))
    print("\n" + "=" * 64)
    if res.get("market_view"):
        print(f"大盘观点: {res['market_view']}")
    print(llm.usage.report())

    (OUT / f"scan_{as_of.date()}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    J.record(str(as_of.date()), picks, facts,
             {"dry_run": llm.dry_run, "model": llm.model})
    j = J.update_performance(panel, idx)
    print("\n--- 累计战绩 ---")
    print(J.report(j))
    print(f"\n完整过程（含分析师意见与辩论）已存 journal/scan_{as_of.date()}.json")


if __name__ == "__main__":
    main()
