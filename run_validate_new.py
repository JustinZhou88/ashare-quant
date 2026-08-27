"""新因子的正式验收：股东户数（筹码集中） + 波动率倾斜反转。

    python run_validate_new.py --universe universe_full.txt

这两个因子在初测里表现都很好，但初测是**全样本 + 小池子 + 简化模拟**。
本脚本用完整引擎（涨停买不进 / T+1 / 真实成本）+ 滚动前推 + 多重检验，
决定它们有没有资格进实盘打分。

判定标准（四项全过才接进 scan_weekly）：
  1. 滚动前推样本外跑赢沪深300
  2. 控制价格反转后的增量 alpha 显著（|t| > 1.96）
  3. Bonferroni 校正后仍显著
  4. 样本外夏普 > 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aq.config import BENCHMARK, TRADING_DAYS, BacktestConfig
from aq.data.altdata import (build_margin_panel, load_holders, load_margin_many,
                             quarter_ends, to_pit_panel)
from aq.data.loader import load_index, load_many
from aq.data.universe import from_file
from aq.engine import metrics as M
from aq.engine.panel import build_panel, liquidity_mask, run
from aq.strategies import factors as F
from aq.validation import deflated as D_
from aq.validation.stats import norm_ppf
from aq.validation.wfa import make_folds

OUT = Path("results_factor")


def _p(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.2%}"


def _f(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_full.txt")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--is-years", type=float, default=4.0)
    ap.add_argument("--oos-years", type=float, default=1.0)
    ap.add_argument("--min-amount", type=float, default=5e7)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    log: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        log.append(s)

    uni = args.universe if Path(args.universe).exists() else "universe.txt"
    syms = from_file(uni)
    print(f"加载 {len(syms)} 只股票 ...", flush=True)
    data = load_many(syms, verbose=False)
    panel = build_panel(data).slice(args.start, None)
    idx = load_index(BENCHMARK)
    mask = liquidity_mask(panel, min_amount=args.min_amount)

    print("加载两融与股东户数 ...", flush=True)
    margin = load_margin_many(panel.symbols, start="2016-01-01", verbose=False)
    mp = build_margin_panel(margin, panel.dates, panel.symbols, lag=1)
    h = load_holders(quarter_ends("2016-03-31", "2026-06-30"))
    hc = to_pit_panel(h, "holders_chg_pct", panel.dates, panel.symbols)
    D = {**mp, "close": panel.close, "amount": panel.amount, "holders_chg": hc}

    cfg = BacktestConfig()
    say("# 新因子验收报告")
    say()
    say(f"- 股票池 **{len(panel.symbols)} 只**，区间 {panel.dates[0].date()} ~ "
        f"{panel.dates[-1].date()}")
    say(f"- 股东户数面板覆盖率 **{hc.notna().mean().mean():.1%}**")
    say(f"- 日均可交易 **{mask.sum(axis=1).mean():.0f} 只**")
    say(f"- 引擎：T+1、涨停买不进、跌停卖不出、真实成本、最多持 "
        f"{cfg.max_positions} 只")
    say()

    # ------------------------------------------------ 候选因子
    cands = {
        "股东户数_筹码集中": (F.holders_concentration, {"k": 10, "rebal": 10}),
        "股东户数_筹码分散": (F.holders_dispersion, {"k": 10, "rebal": 10}),
        "波动倾斜反转_w0.5": (F.vol_tilted_reversal, {"n": 20, "k": 10, "rebal": 5,
                                                     "vol_weight": 0.5}),
        "波动倾斜反转_w0.3": (F.vol_tilted_reversal, {"n": 20, "k": 10, "rebal": 5,
                                                     "vol_weight": 0.3}),
        "【对照】价格反转20": (F.price_reversal_cs, {"n": 20, "k": 10, "rebal": 5}),
        "【对照】价格反转60": (F.price_reversal_cs, {"n": 60, "k": 10, "rebal": 5}),
    }

    rets, stats = {}, {}
    for name, (fn, kw) in cands.items():
        try:
            sig, score = fn(D, **kw)
            r = run(panel, sig, cfg, mask=mask, score=score)
        except Exception as e:                                 # noqa: BLE001
            print(f"  ! {name} 失败: {type(e).__name__}: {e}")
            continue
        rets[name] = r.ret
        m = M.summarize(r.ret, exposure=r.exposure)
        m["n_trades"] = r.n_trades()
        m["blocked"] = r.blocked_entries
        stats[name] = m

    df = pd.DataFrame(rets).dropna(how="all")
    bench = M.benchmark_returns(idx, panel.dates)
    mb = M.summarize(bench)

    say("## 1. 全样本表现（完整引擎）")
    say()
    say("| 因子 | 年化 | 夏普 | 最大回撤 | 交易数 | 涨停买不进 |")
    say("|---|---|---|---|---|---|")
    for n, m in stats.items():
        say(f"| {n} | {_p(m['cagr'])} | {_f(m['sharpe'])} | {_p(m['max_dd'])} | "
            f"{m['n_trades']} | {m['blocked']} |")
    say(f"| 沪深300 | {_p(mb['cagr'])} | {_f(mb['sharpe'])} | {_p(mb['max_dd'])} | — | — |")
    say()
    say("> 「涨停买不进」这一列是诊断值：数字越大说明策略越依赖追涨停，"
        "而那部分收益在实盘拿不到。")
    say()

    # ------------------------------------------------ 滚动前推
    say("## 2. 滚动前推（样本外）")
    say()
    folds = make_folds(panel.dates, is_years=args.is_years, oos_years=args.oos_years)
    oos = {}
    for name in rets:
        chunks = [df.loc[f.oos_start:f.oos_end, name] for f in folds]
        if chunks:
            s = pd.concat(chunks).sort_index()
            oos[name] = s[~s.index.duplicated(keep="first")]
    oos_df = pd.DataFrame(oos).dropna(how="all")
    ob = bench.reindex(oos_df.index).fillna(0.0)
    mob = M.summarize(ob)

    say(f"共 {len(folds)} 折，样本外 {len(oos_df)} 个交易日"
        f"（{oos_df.index[0].date()} ~ {oos_df.index[-1].date()}）")
    say()
    say("| 因子 | 样本外年化 | 夏普 | 最大回撤 |")
    say("|---|---|---|---|")
    oos_stats = {}
    for n in oos_df.columns:
        m = M.summarize(oos_df[n].dropna())
        oos_stats[n] = m
        say(f"| {n} | {_p(m['cagr'])} | {_f(m['sharpe'])} | {_p(m['max_dd'])} |")
    say(f"| 沪深300 | {_p(mob['cagr'])} | {_f(mob['sharpe'])} | {_p(mob['max_dd'])} |")
    say()

    # ------------------------------------------------ 增量 alpha
    say("## 3. 增量 alpha（样本外，控制价格反转）")
    say()
    ctrl = [c for c in oos_df.columns if c.startswith("【对照】")]
    tests = [c for c in oos_df.columns if not c.startswith("【对照】")]
    n_tests = max(len(tests) * max(len(ctrl), 1), 1)
    bonf = norm_ppf(1 - 0.05 / (2 * n_tests))
    say(f"共 {n_tests} 次检验，Bonferroni 校正后的 t 门槛 = **{bonf:.2f}**（原始 1.96）")
    say()
    say("| 因子 | 对照 | 相关性 | 年化 alpha | t 值 | 原始显著 | 校正后显著 |")
    say("|---|---|---|---|---|---|---|")
    alpha_res = {}
    for tname in tests:
        for cname in ctrl:
            d = oos_df[[tname, cname]].dropna()
            if len(d) < 60:
                continue
            y = d[tname].to_numpy()
            X = np.column_stack([np.ones(len(d)), d[cname].to_numpy()])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            nn, kk = X.shape
            s2 = np.sum(resid ** 2) / (nn - kk)
            se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
            t = beta[0] / se if se > 0 else 0.0
            a = beta[0] * TRADING_DAYS
            corr = np.corrcoef(y, d[cname].to_numpy())[0, 1]
            alpha_res.setdefault(tname, []).append(t)
            say(f"| {tname} | {cname} | {_f(corr, 3)} | {_p(a)} | {_f(t)} | "
                f"{'✅' if abs(t) > 1.96 else '❌'} | "
                f"{'✅' if abs(t) > bonf else '❌'} |")
    say()

    # ------------------------------------------------ 判定
    say("## 4. 验收判定")
    say()
    say("| 因子 | 样本外跑赢基准 | 夏普>0.5 | 增量显著 | 校正后显著 | 结论 |")
    say("|---|---|---|---|---|---|")
    approved = []
    for tname in tests:
        m = oos_stats.get(tname, {})
        ts = alpha_res.get(tname, [0.0])
        c1 = m.get("cagr", -1) > mob["cagr"]
        c2 = m.get("sharpe", 0) > 0.5
        c3 = min(abs(t) for t in ts) > 1.96 if ts else False
        c4 = min(abs(t) for t in ts) > bonf if ts else False
        ok = c1 and c2 and c3 and c4
        if ok:
            approved.append(tname)
        say(f"| {tname} | {'✅' if c1 else '❌'} | {'✅' if c2 else '❌'} | "
            f"{'✅' if c3 else '❌'} | {'✅' if c4 else '❌'} | "
            f"{'**通过**' if ok else '不通过'} |")
    say()
    if approved:
        say(f"**通过验收：{', '.join(approved)}** —— 可以接进 scan_weekly 的打分。")
    else:
        say("**没有因子通过全部四项。** 不接进实盘打分。")
        say()
        say("> 这不是失败。初测里表现好、正式验收挂掉，正是这套流程存在的意义 —— "
            "它拦掉的每一个因子，都是一次本来会发生的实盘亏损。")
    say()
    say("---")
    say()
    say("*仅供研究，不构成投资建议。*")

    (OUT / "new_factors_report.md").write_text("\n".join(log), encoding="utf-8")
    oos_df.to_csv(OUT / "new_factors_oos.csv")
    print(f"\n报告写入 {OUT/'new_factors_report.md'}")


if __name__ == "__main__":
    main()
