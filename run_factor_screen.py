"""A股原生因子筛选：两融资金面因子 vs 纯价格对照组。

    python run_factor_screen.py

和 run_screen.py 的区别：那个跑技术指标（时序策略），这个跑截面因子（选股）。
两者共用同一套引擎和同一套反过拟合检验 —— 这样两条路线的结论才可比。

核心问题：**两融资金面因子，能不能跑赢纯价格因子？**
如果不能，说明"另类数据"在这个池子里没提供价格之外的信息，
那就没必要为它增加数据依赖和运维成本。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aq.config import BENCHMARK, TRADING_DAYS, BacktestConfig
from aq.data.altdata import build_margin_panel, load_margin_many
from aq.data.loader import load_index, load_many
from aq.data.universe import from_file
from aq.engine import metrics as M
from aq.engine.panel import build_panel, liquidity_mask, run
from aq.strategies.factors import build_factor_grid
from aq.validation import deflated as D_
from aq.validation.wfa import make_folds, walk_forward

OUT = Path("results_factor")


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.2%}"


def _f(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--universe", default="universe.txt")
    ap.add_argument("--min-amount", type=float, default=3e7)
    ap.add_argument("--is-years", type=float, default=4.0)
    ap.add_argument("--oos-years", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-trades", type=int, default=40)
    ap.add_argument("--margin-lag", type=int, default=1,
                    help="两融数据滞后交易日数（交易所次日才公布，别调成0）")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    log: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        log.append(s)

    t0 = time.time()
    syms = from_file(args.universe)
    data = load_many(syms, verbose=False)
    panel = build_panel(data).slice(args.start, args.end)
    idx = load_index(BENCHMARK)
    mask = liquidity_mask(panel, min_amount=args.min_amount)

    margin = load_margin_many(panel.symbols, start=args.start, verbose=False)
    mp = build_margin_panel(margin, panel.dates, panel.symbols, lag=args.margin_lag)

    # 因子层的输入面板
    D = {**mp, "close": panel.close, "amount": panel.amount}
    has_margin = mp["rz_balance"].notna().any()
    cover = float(mp["rz_balance"].notna().mean().mean())

    say("# A股原生因子筛选报告（两融资金面 vs 纯价格）")
    say()
    say(f"- 区间: **{panel.dates[0].date()} ~ {panel.dates[-1].date()}**"
        f"（{len(panel.dates)} 个交易日）")
    say(f"- 股票池: **{len(panel.symbols)} 只**，其中 **{int(has_margin.sum())} 只**是两融标的")
    say(f"- 两融数据覆盖率: **{cover:.1%}**（面板中非空比例）")
    say(f"- 两融滞后: **{args.margin_lag} 个交易日**（交易所次一交易日才公布 T 日明细）")
    say(f"- 组合: 最多同时持 **{BacktestConfig().max_positions}** 只，每只 1/K 资金，"
        f"名额不足时按**因子分数**挑（不是流动性）")
    say()

    cfg = BacktestConfig()
    grid = build_factor_grid()
    say(f"## 1. 回测 {len(grid)} 个因子策略")
    say()

    rets, entries, rows = {}, {}, []
    for i, s in enumerate(grid, 1):
        try:
            sig, score = s.build(D)
            res = run(panel, sig, cfg, mask=mask, with_trades=False, score=score)
        except Exception as e:                                  # noqa: BLE001
            print(f"  ! {s.id} 失败: {type(e).__name__}: {e}")
            continue
        rets[s.id] = res.ret
        entries[s.id] = res.entries
        m = M.summarize(res.ret, exposure=res.exposure)
        # 注意顺序：**m 必须在前。summarize() 在没有逐笔交易时会返回 n_trades=0，
        # 放在后面会把真实交易数覆盖成 0，导致所有策略都被过滤掉。
        rows.append({**m, **{f"p_{k}": v for k, v in s.params},
                     "strategy": s.id, "family": s.family,
                     "n_trades": res.n_trades()})
        if i % 40 == 0:
            print(f"  {i}/{len(grid)}  ({time.time()-t0:.0f}s)", flush=True)

    ret_mat = pd.DataFrame(rets)
    trd_mat = pd.DataFrame(entries)
    lb = pd.DataFrame(rows).set_index("strategy")
    lb.to_csv(OUT / "leaderboard.csv")
    say(f"完成，用时 {time.time()-t0:.0f} 秒。")
    say()

    live = lb[lb["n_trades"] >= args.min_trades]
    say(f"交易数 ≥ {args.min_trades} 的有 **{len(live)}** 个。")
    say()
    if len(live) < 10:
        say("⚠️ 有效策略太少，无法做统计检验 —— 通常意味着信号生成有 bug，"
            "先检查 `sig.sum(axis=1)` 是不是恒为 0。")
        (OUT / "report.md").write_text("\n".join(log), encoding="utf-8")
        raise SystemExit(1)

    # ---------------------------------------------- 两融 vs 价格 对照
    say("## 2. 关键对照：两融因子 vs 纯价格因子")
    say()
    is_price = live["family"].str.startswith("price_")
    grp = live.groupby(live["family"].where(~is_price, "【对照组】纯价格"))
    tbl = grp.agg(策略数=("sharpe", "size"), 夏普中位数=("sharpe", "median"),
                  夏普最大=("sharpe", "max"), 年化中位数=("cagr", "median"))
    tbl = tbl.sort_values("夏普中位数", ascending=False)
    CTRL = "【对照组】纯价格"
    price_med = float(tbl.loc[CTRL, "夏普中位数"]) if CTRL in tbl.index else np.nan
    say("按**中位数**比，不是最大值 —— 最大值是运气，中位数才是这一族的真实水平：")
    say()
    say("| 因子族 | 策略数 | 夏普中位数 | 夏普最大 | 年化中位数 |")
    say("|---|---|---|---|---|")
    for fam, r in tbl.iterrows():
        star = " ⭐" if (not str(fam).startswith("【") and np.isfinite(price_med)
                        and r["夏普中位数"] > price_med) else ""
        say(f"| {fam}{star} | {int(r['策略数'])} | {_f(r['夏普中位数'])} | "
            f"{_f(r['夏普最大'])} | {_pct(r['年化中位数'])} |")
    say()

    better = ([f for f in tbl.index if not str(f).startswith("【")
               and tbl.loc[f, "夏普中位数"] > price_med]
              if np.isfinite(price_med) else [])
    if better:
        say(f"**{len(better)} 个两融因子族的中位数夏普高于纯价格对照组**：{', '.join(better)}")
    else:
        say("⚠️ **没有任何两融因子族跑赢纯价格对照组。** "
            "在这个股票池和区间上，两融数据没有提供价格之外的信息。")
    say()

    # ---------------------------------------------- 多重检验
    say("## 3. 多重检验校正")
    say()
    sub = ret_mat[live.index]
    sr_daily = (sub.mean() / sub.std(ddof=1)).replace([np.inf, -np.inf], np.nan).dropna()
    sr_var = float(sr_daily.var(ddof=1))
    best_id = live["sharpe"].idxmax()
    n_nom = len(live)
    n_eff = D_.effective_trials(sub)
    dsr_nom = D_.deflated_sharpe(ret_mat[best_id].to_numpy(), n_nom, sr_var)
    dsr_eff = D_.deflated_sharpe(ret_mat[best_id].to_numpy(), int(n_eff), sr_var)

    say(f"排行榜第一: `{best_id}`，夏普 **{_f(dsr_nom['sr_ann'])}**（年化）")
    say()
    say("| 试验次数口径 | N | 噪声门槛 SR₀ | DSR | 结论 |")
    say("|---|---|---|---|---|")
    say(f"| 名义（保守，主判据） | {n_nom} | {_f(dsr_nom['sr0_ann'])} | "
        f"{dsr_nom['dsr']:.3f} | {'✅ 通过' if dsr_nom['passed'] else '❌ 未通过'} |")
    say(f"| 有效独立（放水，参考） | {n_eff:.0f} | {_f(dsr_eff['sr0_ann'])} | "
        f"{dsr_eff['dsr']:.3f} | {'✅ 通过' if dsr_eff['passed'] else '❌ 未通过'} |")
    say()
    say("> 因子网格只有 270 个，比技术指标的 424 个小 —— 搜索空间小，"
        "噪声门槛也低，这是设计上刻意的取舍。")
    say()

    # ---------------------------------------------- 滚动前推
    say("## 4. 滚动前推验证")
    say()
    folds = make_folds(panel.dates, is_years=args.is_years, oos_years=args.oos_years)
    wfa = walk_forward(ret_mat[live.index], trd_mat[live.index], folds,
                       top_k=args.top_k, min_trades_is=args.min_trades // 2)
    if len(wfa.picks):
        wfa.picks.to_csv(OUT / "wfa_picks.csv", index=False)
        say("| 折 | 样本外 | 选中策略 | IS夏普 | OOS夏普 | OOS收益 |")
        say("|---|---|---|---|---|---|")
        for _, r in wfa.picks.iterrows():
            say(f"| {r['fold']} | {r['oos_period']} | `{r['strategy']}` | "
                f"{_f(r['is_sharpe'])} | {_f(r['oos_sharpe'])} | {_pct(r['oos_return'])} |")
        say()

    wm = M.summarize(wfa.oos_ret)
    bench = M.benchmark_returns(idx, panel.dates).reindex(wfa.oos_ret.index).fillna(0.0)
    wbm = M.summarize(bench)
    say("| | 年化 | 夏普 | 最大回撤 | 卡玛 |")
    say("|---|---|---|---|---|")
    say(f"| 前推策略（样本外） | {_pct(wm['cagr'])} | {_f(wm['sharpe'])} | "
        f"{_pct(wm['max_dd'])} | {_f(wm['calmar'])} |")
    say(f"| 同期沪深300 | {_pct(wbm['cagr'])} | {_f(wbm['sharpe'])} | "
        f"{_pct(wbm['max_dd'])} | {_f(wbm['calmar'])} |")
    say()
    c = wfa.mean_is_oos_corr
    say(f"**样本内/样本外夏普秩相关: {_f(c, 3)}**"
        f" —— {'筛选流程有预测力' if np.isfinite(c) and c > 0.1 else '筛选流程没有预测力'}")
    say()

    # ---------------------------------------------- 结论
    say("## 5. 结论")
    say()
    checks = [
        ("两融因子中位数跑赢纯价格对照组", bool(better)),
        ("多重检验（名义口径）通过", dsr_nom["passed"]),
        ("样本外跑赢沪深300", wm["cagr"] > wbm["cagr"]),
        ("样本内外排名有相关性", bool(np.isfinite(c) and c > 0.1)),
    ]
    for name, ok in checks:
        say(f"- {'✅' if ok else '❌'} {name}")
    say()
    n_pass = sum(1 for _, ok in checks if ok)
    say(f"**{n_pass}/{len(checks)} 项通过。**")
    say()
    say("---")
    say()
    say("*仅供研究，不构成投资建议。回测不代表未来收益。*")

    (OUT / "report.md").write_text("\n".join(log), encoding="utf-8")
    ret_mat.to_csv(OUT / "returns_matrix.csv")
    print(f"\n报告已写入 {OUT/'report.md'}（用时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
