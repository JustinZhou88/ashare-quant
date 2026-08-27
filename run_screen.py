"""主流程：424 个策略 → 回测 → 反过拟合验证 → 报告。

    python run_screen.py                       # 默认全套
    python run_screen.py --start 2016-01-01    # 指定区间
    python run_screen.py --no-random           # 跳过随机基准（省时间）

产出全部写进 results/：
    leaderboard.csv     全样本排行榜（**不要直接用第一名**）
    wfa_picks.csv       每折样本内选了谁、样本外表现如何
    trades_best.csv     决赛策略的逐笔交易（喂给 Claude 分析用）
    report.md           完整报告
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aq import analysis as A
from aq.config import BENCHMARK, TRADING_DAYS, BacktestConfig
from aq.data.loader import load_index, load_many
from aq.data.universe import from_file
from aq.engine import metrics as M
from aq.engine.panel import build_panel, liquidity_mask, run
from aq.strategies import external as EX
from aq.strategies.grid import build_grid
from aq.validation import deflated as D
from aq.validation import random_bench as RB
from aq.validation.wfa import make_folds, walk_forward

OUT = Path("results")


def _fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.2%}"


def _fmt(x, nd=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--universe", default="universe.txt")
    ap.add_argument("--min-amount", type=float, default=3e7, help="日均成交额下限")
    ap.add_argument("--is-years", type=float, default=4.0)
    ap.add_argument("--oos-years", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=3, help="每折选几个策略等权组合")
    ap.add_argument("--min-trades", type=int, default=40)
    ap.add_argument("--stop-loss", type=float, default=None)
    ap.add_argument("--max-positions", type=int, default=10,
                    help="最多同时持有几只，每只占 1/K 资金")
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--external", default=None,
                    help="外部信号目录（每个 csv 一个策略），见 aq/strategies/external.py")
    ap.add_argument("--knowledge-cutoff", default=None,
                    help="LLM 知识截止日，如 2025-01-01；早于此日的外部信号会被警告")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    cfg = BacktestConfig(stop_loss=args.stop_loss, max_positions=args.max_positions)
    log: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        log.append(s)

    # ---------------------------------------------------------------- 数据
    t0 = time.time()
    syms = from_file(args.universe)
    data = load_many(syms, verbose=False)
    panel = build_panel(data).slice(args.start, args.end)
    idx = load_index(BENCHMARK)
    mask = liquidity_mask(panel, min_amount=args.min_amount)

    say("# A股策略筛选报告")
    say()
    say(f"- 区间: **{panel.dates[0].date()} ~ {panel.dates[-1].date()}** "
        f"（{len(panel.dates)} 个交易日，约 {len(panel.dates)/TRADING_DAYS:.1f} 年）")
    say(f"- 股票池: **{len(panel.symbols)} 只**（市值分层随机抽样）")
    say(f"- 平均每日可交易: **{mask.sum(axis=1).mean():.0f} 只**"
        f"（成交额 ≥ {args.min_amount/1e4:.0f} 万 + 上市满 250 日 + 未停牌）")
    say(f"- 成交模型: T日收盘出信号 → **T+1 开盘成交**；涨停不买入、跌停不卖出；"
        f"T+1 制度；双边成本约 {(cfg.cost.slippage_rate*2 + cfg.cost.commission_rate*2 + cfg.cost.stamp_tax_rate)*1e4:.0f} 个基点")
    say()

    # ---------------------------------------------------------------- 回测
    grid = build_grid()
    if args.external:
        ext = EX.discover(args.external)
        grid = grid + ext
        if ext:
            say(f"另接入 **{len(ext)}** 个外部信号: "
                + ", ".join(f"`{e.name}`" for e in ext))
            if args.knowledge_cutoff:
                for e in ext:
                    w = EX.warn_knowledge_cutoff(EX.load_signal_csv(e.path),
                                                 args.knowledge_cutoff)
                    if w:
                        say(f"- `{e.name}`: {w}")
            say()
    say(f"## 1. 回测 {len(grid)} 个策略")
    say()
    rets, entries, rows = {}, {}, []
    for i, s in enumerate(grid, 1):
        try:
            sig = s.signal(panel)
            res = run(panel, sig, cfg, mask=mask, with_trades=False)
        except Exception as e:                                   # noqa: BLE001
            print(f"  ! {s.id} 失败: {e}")
            continue
        rets[s.id] = res.ret
        entries[s.id] = res.entries
        m = M.summarize(res.ret, exposure=res.exposure)
        # 注意顺序：m 里也有 n_trades（无交易明细时为 0），必须让引擎的真实计数覆盖它
        row = {"strategy": s.id, "family": s.family, **m,
               "n_trades": res.n_trades(), "blocked": res.blocked_entries}
        row.update({f"p_{k}": v for k, v in s.params})
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(grid)}  ({time.time()-t0:.0f}s)", flush=True)

    ret_mat = pd.DataFrame(rets)
    trd_mat = pd.DataFrame(entries)
    lb = pd.DataFrame(rows).set_index("strategy")
    lb.to_csv(OUT / "leaderboard.csv")
    say(f"完成，用时 {time.time()-t0:.0f} 秒。")
    say()

    # 有效交易数过滤
    live = lb[lb["n_trades"] >= args.min_trades]
    say(f"其中 **{len(live)}** 个策略交易数 ≥ {args.min_trades}，进入后续分析。")
    say()
    if live.empty:
        say("没有策略达到最低交易数，无法继续。请调低 --min-trades 或延长回测区间。")
        (OUT / "report.md").write_text("\n".join(log), encoding="utf-8")
        return

    # ---------------------------------------------------------------- 基准
    bench_ret = M.benchmark_returns(idx, panel.dates)
    eq_ret = (panel.close.pct_change().where(mask) ).mean(axis=1).fillna(0.0)
    bm = M.summarize(bench_ret)
    eqm = M.summarize(eq_ret)

    # ---------------------------------------------------------------- 排行榜
    top = live.nlargest(10, "sharpe")
    say("## 2. 全样本排行榜（前 10）")
    say()
    say("> ⚠️ **这张表是本项目里最没用的一张表。** 它是在同一段历史上做了 "
        f"{len(grid)} 次搜索之后挑出来的最大值，天然包含选择偏差。"
        "下面第 3、4 节才是判断依据。")
    say()
    say("| 策略 | 年化 | 夏普 | 最大回撤 | 卡玛 | 交易数 | 胜率 | 仓位 |")
    say("|---|---|---|---|---|---|---|---|")
    for sid, r in top.iterrows():
        say(f"| `{sid}` | {_fmt_pct(r['cagr'])} | {_fmt(r['sharpe'])} | "
            f"{_fmt_pct(r['max_dd'])} | {_fmt(r['calmar'])} | {int(r['n_trades'])} | "
            f"— | {_fmt_pct(r['exposure'])} |")
    say()
    say(f"| **沪深300** | {_fmt_pct(bm['cagr'])} | {_fmt(bm['sharpe'])} | "
        f"{_fmt_pct(bm['max_dd'])} | {_fmt(bm['calmar'])} | — | — | 100% |")
    say(f"| **股票池等权日再平衡** | {_fmt_pct(eqm['cagr'])} | {_fmt(eqm['sharpe'])} | "
        f"{_fmt_pct(eqm['max_dd'])} | {_fmt(eqm['calmar'])} | — | — | 100% |")
    say()
    say("> 「股票池等权日再平衡」不扣交易成本，且股票池取自**今天仍在市**的公司，"
        "含幸存者偏差 —— 它比沪深300高出这么多，主要是这两个原因，不是可实现收益。"
        "任何策略要证明自己有用，至少得跑赢这一行。")
    say()

    # ------------------------------------------------- 多重检验校正
    say("## 3. 多重检验校正：这个夏普里有多少是运气？")
    say()
    sub = ret_mat[live.index]
    n_eff = D.effective_trials(sub)
    sr_daily = (sub.mean() / sub.std(ddof=1)).replace([np.inf, -np.inf], np.nan).dropna()
    sr_var = float(sr_daily.var(ddof=1))
    best_id = top.index[0]

    # 以名义试验次数为准（保守）。用特征值折算的"有效试验数"会把门槛压低，
    # 方向是放水的 —— 长仓策略被市场 beta 主导，相关矩阵必然只有一个大特征值，
    # 折算出来的 N_eff 接近 1，那等于假装自己只试了一次。只作敏感性参考。
    dsr = D.deflated_sharpe(ret_mat[best_id].to_numpy(), n_trials=len(live),
                            sr_variance=sr_var)
    dsr_loose = D.deflated_sharpe(ret_mat[best_id].to_numpy(), n_trials=n_eff,
                                  sr_variance=sr_var)

    say(f"- 试验次数: **{len(live)}**（按名义次数算，保守口径）")
    say(f"- 候选策略年化夏普的横截面标准差: **{_fmt(np.sqrt(sr_var) * np.sqrt(TRADING_DAYS))}**")
    say(f"- **纯噪声能达到的夏普门槛 SR₀: {_fmt(dsr['sr0_ann'])}**（年化）"
        f" —— 搜了 {len(live)} 次，光靠运气就能挑出这个水平")
    say(f"- 排行榜第一 `{best_id}` 的实际夏普: **{_fmt(dsr['sr_ann'])}**（年化）")
    say(f"- **Deflated Sharpe Ratio: {dsr['dsr']:.3f}** "
        f"→ {'✅ 通过（>0.95）' if dsr['passed'] else '❌ 未通过（需 >0.95）'}")
    say()
    say(f"> 敏感性：若按相关矩阵折算的有效试验数 {n_eff:.1f} 计算，门槛降到 "
        f"{_fmt(dsr_loose['sr0_ann'])}，DSR = {dsr_loose['dsr']:.3f}。"
        f"但长仓策略高度共线，这个折算会严重低估实际搜索次数，**不作为判据**。")
    say()
    if not dsr["passed"]:
        say("> 结论：搜索了这么多次之后，排行榜第一名的夏普"
            "**无法与运气区分**。这是绝大多数策略搜索的正常结局，不是 bug。")
        say()

    # ------------------------------------------------- 滚动前推
    say("## 4. 滚动前推验证（真正的答案）")
    say()
    folds = make_folds(panel.dates, is_years=args.is_years, oos_years=args.oos_years)
    wfa = walk_forward(ret_mat[live.index], trd_mat[live.index], folds,
                       top_k=args.top_k, min_trades_is=args.min_trades // 2)
    say(f"切分方式: 样本内 {args.is_years:.0f} 年 → 样本外 {args.oos_years:.0f} 年，"
        f"滚动 {len(folds)} 折；每折取样本内夏普前 {args.top_k} 名等权。")
    say()

    if len(wfa.picks):
        wfa.picks.to_csv(OUT / "wfa_picks.csv", index=False)
        say("| 折 | 样本内 | 样本外 | 选中策略 | IS夏普 | OOS夏普 | OOS收益 |")
        say("|---|---|---|---|---|---|---|")
        for _, r in wfa.picks.iterrows():
            say(f"| {r['fold']} | {r['is_period']} | {r['oos_period']} | "
                f"`{r['strategy']}` | {_fmt(r['is_sharpe'])} | {_fmt(r['oos_sharpe'])} | "
                f"{_fmt_pct(r['oos_return'])} |")
        say()

    wm = M.summarize(wfa.oos_ret)
    wfa_bench = bench_ret.reindex(wfa.oos_ret.index).fillna(0.0)
    wbm = M.summarize(wfa_bench)
    say("**拼接后的样本外表现** —— 这是你真正能指望的数字：")
    say()
    say("| | 年化 | 夏普 | 最大回撤 | 卡玛 | 盈利月占比 |")
    say("|---|---|---|---|---|---|")
    say(f"| 前推策略 | {_fmt_pct(wm['cagr'])} | {_fmt(wm['sharpe'])} | "
        f"{_fmt_pct(wm['max_dd'])} | {_fmt(wm['calmar'])} | {_fmt_pct(wm['pos_month_pct'])} |")
    say(f"| 同期沪深300 | {_fmt_pct(wbm['cagr'])} | {_fmt(wbm['sharpe'])} | "
        f"{_fmt_pct(wbm['max_dd'])} | {_fmt(wbm['calmar'])} | {_fmt_pct(wbm['pos_month_pct'])} |")
    say()

    c = wfa.mean_is_oos_corr
    say(f"**样本内/样本外夏普秩相关: {_fmt(c, 3)}**")
    say()
    if not np.isfinite(c):
        say("> 样本不足，无法判断。")
    elif c < 0.1:
        say("> ⚠️ 接近 0（甚至为负）意味着：**「用历史表现挑策略」这个动作本身没有预测力。**")
        say("> 这是整个流程里最重要的一个数字。它为 0，说明再怎么调参数、加策略都是徒劳 ——")
        say("> 该做的是换一类信号（基本面、资金流、行业轮动），而不是继续在技术指标里搜。")
    elif c < 0.3:
        say("> 弱正相关。历史排名有一点点信息量，但远不足以支撑"
            "「挑第一名」这种激进做法，用前 k 名等权更稳。")
    else:
        say("> ✅ 明显正相关，说明筛选流程确实有预测力，可以继续深挖这个方向。")
    say()

    # ------------------------------------------------- 随机基准
    wpct = None
    if not args.no_random:
        say("## 5. 随机基准：瞎买能赚多少？")
        say()
        exp = float(live.loc[best_id, "exposure"])
        hold = max(float(len(panel.dates)) * exp / max(live.loc[best_id, "n_trades"], 1)
                   * len(panel.symbols), 1.0)
        print("  跑随机基准...", flush=True)
        rb = RB.random_benchmark(panel, cfg, exposure=exp, hold_days=min(hold, 60),
                                 n_sims=args.n_random, mask=mask, verbose=True)
        rb.to_csv(OUT / "random_bench.csv", index=False)
        pct = RB.percentile_of(dsr["sr_ann"], rb["sharpe"].to_numpy())
        wpct = RB.percentile_of(wm["sharpe"], rb["sharpe"].to_numpy())
        say(f"用同样的仓位（{exp:.1%}）和持仓周期随机买卖，跑 {len(rb)} 次：")
        say()
        say(f"- 随机策略夏普: 中位数 **{_fmt(rb['sharpe'].median())}**，"
            f"95分位 **{_fmt(rb['sharpe'].quantile(0.95))}**")
        say(f"- 排行榜第一名在随机分布中的百分位: **{_fmt(pct, 1)}%**")
        say(f"- 前推样本外结果的百分位: **{_fmt(wpct, 1)}%**")
        say()
        if np.isfinite(wpct) and wpct < 95:
            say("> ⚠️ 样本外表现没有超过随机策略的 95 分位 —— "
                "在统计上和「按同样频率瞎买」没有区别。")
            say()

    # ------------------------------------------------- 决赛策略归因
    say("## 6. 交易归因（喂给 Claude 分析的原料）")
    say()
    final_id = best_id
    if len(wfa.picks):
        # 用前推中出现次数最多的策略作为决赛选手，比全样本第一名更可信
        vc = wfa.picks["strategy"].value_counts()
        if len(vc) and vc.iloc[0] > 1:
            final_id = vc.index[0]
            say(f"决赛策略选的是**在 {len(folds)} 折前推里被选中 {vc.iloc[0]} 次**的 "
                f"`{final_id}`，而不是全样本第一名 —— 跨折稳定性比单次最大值可信。")
            say()

    strat = next(s for s in grid if s.id == final_id)
    res = run(panel, strat.signal(panel), cfg, mask=mask, with_trades=True)
    regimes = A.label_regimes(idx)
    tr = A.attach_context(res.trades, regimes, idx)
    tr.to_csv(OUT / "trades_best.csv", index=False)
    say(f"`{final_id}` 共 {len(tr)} 笔交易，已导出 `results/trades_best.csv`。")
    say()

    conc = A.concentration(tr)
    if conc:
        say("**利润集中度**（识别「靠几笔运气」）：")
        say()
        for k, v in conc.items():
            if isinstance(v, float) and np.isfinite(v):
                say(f"- {k}: {v:.1%}" if "占" in k else f"- {k}: {v:.4f}")
            else:
                say(f"- {k}: {v}")
        say()

    reg_tbl = A.by_regime(tr)
    if len(reg_tbl):
        say("**分市场环境表现**：")
        say()
        say("```")
        say(reg_tbl.to_string())
        say("```")
        say()

    yr_tbl = A.by_year(tr)
    if len(yr_tbl):
        say("**分年度表现**：")
        say()
        say("```")
        say(yr_tbl.to_string())
        say("```")
        say()

    mm = A.mae_mfe(tr)
    if mm:
        say("**浮盈/浮亏分析**（止损位参考）：")
        say()
        for k, v in mm.items():
            say(f"- {k}: {v:.2%}" if np.isfinite(v) else f"- {k}: n/a")
        say()

    ps = A.plateau_score(live.reset_index(), strat.family, metric="sharpe")
    say(f"**参数高原度**（`{strat.family}` 族）: {_fmt(ps, 2)} "
        f"— 越接近 1 说明一整片参数都有效（稳健），接近 0 则是孤峰（过拟合）。")
    say()

    # ------------------------------------------------- 结论
    say("## 7. 结论")
    say()
    verdict = []
    verdict.append(("多重检验校正 (DSR>0.95)", dsr["passed"]))
    if wpct is not None:
        verdict.append(("样本外跑赢随机策略的95分位（最硬的一关）",
                        bool(np.isfinite(wpct) and wpct >= 95)))
    verdict.append(("样本外年化跑赢沪深300（最软的一关）", wm["cagr"] > wbm["cagr"]))
    verdict.append(("样本内外排名有相关性 (>0.1)", bool(np.isfinite(c) and c > 0.1)))
    if conc:
        verdict.append(("剔除最赚的5笔后仍盈利", bool(conc.get("剔除前5笔后仍为正"))))
    for name, ok in verdict:
        say(f"- {'✅' if ok else '❌'} {name}")
    say()
    n_pass = sum(1 for _, ok in verdict if ok)
    if n_pass == len(verdict):
        say(f"**{n_pass}/{len(verdict)} 项通过。** 值得进入小资金实盘验证阶段"
            "（仍建议先纸上交易 3 个月）。")
    else:
        say(f"**{n_pass}/{len(verdict)} 项通过 —— 不建议投入实盘。**")
        say()
        say("这不是失败。在 424 个技术指标策略里找不到稳健 alpha 是**预期结果**：")
        say("这些指标全市场都在用，早被交易掉了。真正的下一步是换信息源"
            "（财务因子、资金流向、行业景气度、分析师预期修正），")
        say("而不是继续在均线参数里搜索 —— 那只会让过拟合更严重。")
    say()
    say("---")
    say()
    say("*本报告由回测系统自动生成，仅供研究使用，不构成投资建议。"
        "历史回测不代表未来收益。*")

    (OUT / "report.md").write_text("\n".join(log), encoding="utf-8")
    ret_mat.to_csv(OUT / "returns_matrix.csv")
    pd.DataFrame({"wfa_oos": wfa.oos_ret, "benchmark": wfa_bench}).to_csv(
        OUT / "wfa_oos_returns.csv")
    print(f"\n报告已写入 {OUT/'report.md'}（总用时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
