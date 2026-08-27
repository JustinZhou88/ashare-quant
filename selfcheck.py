"""发布前自检 —— 一条命令跑完所有模块的健康检查。

    python selfcheck.py           # 全部检查
    python selfcheck.py --quick   # 跳过需要加载全量行情的项（快 10 倍）

设计原则：**每一项都要能明确判定通过或失败**，不输出"看起来还行"这种话。
这个项目踩过四次「静默失效」（缺 import、缓存 off-by-one、JS 字面量 \\n、
JSON 里的 NaN），共同点都是**看起来正常但实际没工作**。所以这里的检查
一律做实际断言，不看日志、不看返回码。
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    """装饰器：把函数变成一项检查。返回 (bool, 说明) 或抛异常。"""
    def deco(fn):
        try:
            ok, msg = fn()
        except Exception as e:                                 # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {str(e)[:150]}"
            if os.environ.get("SELFCHECK_TRACE"):
                traceback.print_exc()
        _results.append((name, ok, msg))
        print(f"  {'✓' if ok else '✗'} {name:32s} {msg}")
        return fn
    return deco


def run_all(quick: bool = False) -> int:
    print("=" * 74)
    print("A股量化系统 · 发布前自检")
    print("=" * 74)

    # ---------------------------------------------------------- 1 依赖与导入
    print("\n【1】模块导入")

    @check("全部模块可导入")
    def _():
        mods = ["aq.config", "aq.envfile", "aq.journal", "aq.paper", "aq.analysis",
                "aq.data.loader", "aq.data.universe", "aq.data.altdata",
                "aq.data.sentiment", "aq.data.global_ctx", "aq.data.guba",
                "aq.data.grok_x", "aq.engine.panel", "aq.engine.metrics",
                "aq.strategies.factors", "aq.strategies.library",
                "aq.strategies.indicators", "aq.strategies.grid",
                "aq.validation.deflated", "aq.validation.wfa",
                "aq.validation.stats", "aq.validation.random_bench",
                "aq.agents.rules", "aq.agents.context", "aq.agents.llm",
                "aq.agents.analysts", "aq.agents.forecast"]
        bad = []
        for m in mods:
            try:
                importlib.import_module(m)
            except Exception as e:                             # noqa: BLE001
                bad.append(f"{m}({type(e).__name__})")
        return not bad, f"{len(mods) - len(bad)}/{len(mods)} 正常" + (
            f"，失败: {bad[:3]}" if bad else "")

    @check("入口脚本可编译")
    def _():
        scripts = ["web_server.py", "update_data.py", "scan_weekly.py",
                   "dashboard.py", "download_bulk.py", "download_margin.py",
                   "check_web.py", "run_screen.py", "run_factor_screen.py",
                   "tools/x_scrape.py", "tools/mediacrawler_adapt.py"]
        bad = [s for s in scripts if s and Path(s).exists() and
               subprocess.run([sys.executable, "-m", "py_compile", s],
                              capture_output=True).returncode != 0]
        return not bad, f"{len(scripts)} 个脚本" + (f"，失败: {bad}" if bad else "")

    # ---------------------------------------------------------- 2 配置
    print("\n【2】配置与密钥")

    @check("rules.yaml 可解析")
    def _():
        from aq.agents.rules import Rules
        r = Rules.load("rules.yaml")
        need = ["hard", "risk", "schedule"]
        miss = [k for k in need if not getattr(r, k)]
        return not miss, f"硬约束 {len(r.hard)} 项、风控 {len(r.risk)} 项" + (
            f"，缺: {miss}" if miss else "")

    @check(".env 权限与 gitignore")
    def _():
        p = Path(".env")
        if not p.exists():
            return True, "无 .env（未配置 key）"
        mode = oct(p.stat().st_mode)[-3:]
        ign = ".env" in Path(".gitignore").read_text(encoding="utf-8") \
            if Path(".gitignore").exists() else False
        ok = mode == "600" and ign
        return ok, f"权限 {mode}（应 600），gitignore {'✓' if ign else '✗'}"

    @check("LLM key 已配置")
    def _():
        from aq.agents.llm import LLM
        llm = LLM()
        return not llm.dry_run, (f"{llm.provider}/{llm.model}" if not llm.dry_run
                                 else "无 key，扫描会输出占位内容")

    @check("买得起门槛自洽")
    def _():
        from aq.agents.rules import Rules
        r = Rules.load("rules.yaml").risk
        cash = float(r.get("init_cash") or 0)
        pos = float(r.get("position_size") or 0)
        n = int(r.get("max_positions") or 5)
        if cash <= 0 or pos <= 0:
            return False, "init_cash 或 position_size 未设置"
        total = pos * n
        note = f"单只 {cash*pos:,.0f} 元（≤{cash*pos/100:.0f} 元/股），总仓位 {total:.0%}"
        return total <= 1.0, note + ("" if total <= 1.0 else " —— 总仓位超 100%！")

    # ---------------------------------------------------------- 3 数据
    print("\n【3】数据完整性")

    @check("行情缓存")
    def _():
        import pandas as pd
        fs = [f for f in glob.glob("data_cache/*.csv")
              if os.path.basename(f)[:6].isdigit()]
        if not fs:
            return False, "无行情数据"
        import random
        random.seed(0)
        bad, dup = [], []
        latest = {}
        need = ["open", "close", "raw_open", "raw_close", "volume", "amount", "tradable"]
        for f in random.sample(fs, min(120, len(fs))):
            s = os.path.basename(f)[:-4]
            d = pd.read_csv(f, parse_dates=["date"], index_col="date")
            if any(c not in d.columns for c in need):
                bad.append(s)
                continue
            if d.index.duplicated().any():
                dup.append(s)
            if (d[["close", "raw_close"]] <= 0).any().any():
                bad.append(s)
            latest[s] = d.index.max()
        newest = max(latest.values())
        stale = sum(1 for v in latest.values() if (newest - v).days > 7)
        ok = not bad and not dup and stale < len(latest) * 0.1
        return ok, (f"{len(fs)} 只，抽查 {len(latest)}：损坏 {len(bad)}、"
                    f"重复 {len(dup)}、超一周未更新 {stale}，最新 {newest.date()}")

    @check("另类数据")
    def _():
        import pandas as pd
        n_margin = len(glob.glob("data_cache/alt/margin_*.csv"))
        parts = [f"两融 {n_margin} 只"]
        for f, nm in [("data_cache/alt/reports.csv", "研报"),
                      ("data_cache/alt/holders.csv", "股东户数")]:
            parts.append(f"{nm} {len(pd.read_csv(f)):,} 条" if os.path.exists(f)
                         else f"{nm} 缺失")
        return n_margin > 0, "、".join(parts)

    @check("行业字段可用")
    def _():
        import pandas as pd
        f = "data_cache/_symbols.csv"
        if not os.path.exists(f):
            return False, "股票列表缺失"
        d = pd.read_csv(f, dtype={"symbol": str})
        has = "industry" in d.columns
        n = int((d["industry"].astype(str).str.len() > 0).sum()) if has else 0
        return has and n > 0, (f"{n}/{len(d)} 只有行业、{d['industry'].nunique()} 个分类"
                               if has else "无 industry 列 —— 行业黑名单会静默失效")

    @check("指数与个股日期同步")
    def _():
        import pandas as pd
        fs = [f for f in glob.glob("data_cache/*.csv")
              if os.path.basename(f)[:6].isdigit()]
        if not fs or not os.path.exists("data_cache/idx_000300.csv"):
            return False, "数据缺失"
        px = pd.read_csv(fs[0], parse_dates=["date"])["date"].max()
        ix = pd.read_csv("data_cache/idx_000300.csv", parse_dates=["date"])["date"].max()
        gap = abs((px - ix).days)
        return gap <= 1, f"个股 {px.date()}、指数 {ix.date()}，差 {gap} 天"

    if quick:
        return report()

    # ---------------------------------------------------------- 4 流水线
    print("\n【4】流水线（需加载全量行情，较慢）")
    import warnings
    warnings.filterwarnings("ignore")
    from aq.config import BENCHMARK
    from aq.data.loader import load_index, load_many
    from aq.data.universe import from_file
    from aq.engine.panel import build_panel
    uni = "universe_full.txt" if Path("universe_full.txt").exists() else "universe.txt"
    panel = build_panel(load_many(from_file(uni), verbose=False))
    idx = load_index(BENCHMARK)

    @check("面板构建")
    def _():
        return len(panel.symbols) > 100 and len(panel.dates) > 500, \
            f"{len(panel.symbols)} 只 × {len(panel.dates)} 日"

    @check("硬过滤链")
    def _():
        from aq.agents.rules import Rules, apply_hard_filters
        from aq.data.loader import fetch_all_symbols
        meta = fetch_all_symbols()
        passed, reasons = apply_hard_filters(panel, Rules.load("rules.yaml"),
                                             panel.dates[-1], meta=meta)
        ok = 50 < len(passed) < len(panel.symbols)
        return ok, f"{len(panel.symbols)} → {len(passed)}，{len(reasons)} 条规则生效"

    @check("因子打分无未来函数")
    def _():
        import numpy as np
        from scan_weekly import factor_score
        sc = factor_score(panel, {"close": panel.close, "amount": panel.amount})
        # 截断数据重算，比较重叠段是否一致 —— 不一致就说明用了未来数据
        cut = panel.dates[-30]
        sc2 = factor_score(panel.slice(None, cut),
                           {"close": panel.close.loc[:cut],
                            "amount": panel.amount.loc[:cut]})
        a = sc.loc[cut].dropna()
        b = sc2.loc[cut].reindex(a.index)
        diff = float(np.nanmax(np.abs(a - b)))
        return diff < 1e-9, f"截断重算最大偏差 {diff:.2e}（>0 即有未来函数）"

    @check("模拟盘")
    def _():
        import yaml
        from aq.paper import pending_actions, simulate, summarize
        R = (yaml.safe_load(Path("rules.yaml").read_text(encoding="utf-8")) or {}).get("risk") or {}
        res = simulate(panel, R)
        s = summarize(res, panel, idx["close"])
        if s.get("empty"):
            return True, "尚无可执行推荐（正常，等下一交易日）"
        acts = pending_actions(res, panel, R)
        # 对账：每只推荐必须落在 建仓/待执行/受阻 三者之一
        from aq.paper import _load_journal
        n_rec = len(_load_journal())
        n_acc = len(res.trades) + len(res.pending) + len(res.blocked)
        ok = n_acc >= n_rec
        return ok, (f"权益 {s['equity']:,.0f}（{s['total_return']:+.2%}）、"
                    f"{s['n_trades']} 笔、提醒 {len(acts)} 条、"
                    f"对账 {n_acc}≥{n_rec} {'✓' if ok else '✗'}")

    @check("历史类比预测")
    def _():
        from aq.agents.forecast import AnalogForecaster
        fc = AnalogForecaster(panel, horizon=15)
        sym = panel.symbols[len(panel.symbols) // 2]
        r = fc.forecast(sym, panel.dates[-1])
        if not r:
            return False, "无法生成"
        if not r.get("enough"):
            return True, f"样本不足（{r.get('n_analogs')}），已正确降级"
        return 0 < r["p_hit_15"] < 1, f"{r['n_analogs']:,} 个类比样本"

    @check("静态报告生成")
    def _():
        r = subprocess.run([sys.executable, "dashboard.py", "--no-open"],
                           capture_output=True, text=True)
        ok = r.returncode == 0 and Path("dashboard.html").exists()
        size = Path("dashboard.html").stat().st_size if ok else 0
        return ok, f"{size:,} 字节"

    return report()


def report() -> int:
    print("\n" + "=" * 74)
    bad = [(n, m) for n, ok, m in _results if not ok]
    print(f"结果：{len(_results) - len(bad)}/{len(_results)} 项通过")
    if bad:
        print("\n未通过：")
        for n, m in bad:
            print(f"  ✗ {n}: {m}")
    print("=" * 74)
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过需加载全量行情的检查")
    sys.exit(run_all(ap.parse_args().quick))
