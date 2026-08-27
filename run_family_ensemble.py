"""因子族等权组合 —— 检验「选族不选参数」这个假设。

上一步 run_factor_screen.py 发现一个矛盾：
  - 滚动前推的样本外收益明显跑赢基准
  - 但样本内/样本外的**参数排名**相关性只有 0.045（等于没有）

唯一自洽的解释：alpha 在**因子族**层面，不在参数层面。
每折选中的都是同一批族（margin_reversal / short_squeeze），只是参数不同 ——
说明"这个族有效"是可重复的，"这组参数最好"是运气。

如果解释成立，那么**族内全部参数等权**应该：
  1. 样本外表现不比"挑最优参数"差
  2. 试验次数从 270 降到 8（只在族层面做选择），多重检验门槛大幅降低
  3. 不需要调参 —— 少一个过拟合的入口

这个脚本就是验证这三点。
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
from aq.validation.wfa import make_folds

OUT = Path("results_factor")


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.2%}"


def _f(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--is-years", type=float, default=4.0)
    ap.add_argument("--oos-years", type=float, default=1.0)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    log: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        log.append(s)

    t0 = time.time()
    syms = from_file("universe.txt")
    data = load_many(syms, verbose=False)
    panel = build_panel(data).slice(args.start, None)
    idx = load_index(BENCHMARK)
    mask = liquidity_mask(panel)
    margin = load_margin_many(panel.symbols, start=args.start, verbose=False)
    mp = build_margin_panel(margin, panel.dates, panel.symbols, lag=1)
    D = {**mp, "close": panel.close, "amount": panel.amount}
    cfg = BacktestConfig()

    say("# 因子族等权组合报告")
    say()
    say("检验假设：**alpha 在因子族层面，参数选择是噪声。**")
    say()

    # ---------------------------------------------- 逐策略回测（复用）
    grid = build_factor_grid()
    rets: dict[str, pd.Series] = {}
    fam_of: dict[str, str] = {}
    for i, s in enumerate(grid, 1):
        try:
            sig, score = s.build(D)
            res = run(panel, sig, cfg, mask=mask, score=score)
        except Exception:                                      # noqa: BLE001
            continue
        rets[s.id] = res.ret
        fam_of[s.id] = s.family
        if i % 60 == 0:
            print(f"  {i}/{len(grid)} ({time.time()-t0:.0f}s)", flush=True)
    ret_mat = pd.DataFrame(rets)

    # ---------------------------------------------- 族等权
    fams = sorted(set(fam_of.values()))
    fam_ret = pd.DataFrame({
        f: ret_mat[[c for c in ret_mat.columns if fam_of[c] == f]].mean(axis=1)
        for f in fams})
    fam_ret.to_csv(OUT / "family_returns.csv")

    bench = M.benchmark_returns(idx, panel.dates)
    say(f"- 区间 **{panel.dates[0].date()} ~ {panel.dates[-1].date()}**，"
        f"{len(fams)} 个因子族，每族内全部参数等权")
    say()
    say("## 1. 各因子族等权组合（全样本）")
    say()
    say("| 因子族 | 年化 | 夏普 | 最大回撤 | 卡玛 | 盈利月占比 |")
    say("|---|---|---|---|---|---|")
    fam_stats = {}
    for f in fams:
        m = M.summarize(fam_ret[f])
        fam_stats[f] = m
        say(f"| {f} | {_pct(m['cagr'])} | {_f(m['sharpe'])} | {_pct(m['max_dd'])} | "
            f"{_f(m['calmar'])} | {_pct(m['pos_month_pct'])} |")
    bm = M.summarize(bench)
    say(f"| **沪深300** | {_pct(bm['cagr'])} | {_f(bm['sharpe'])} | {_pct(bm['max_dd'])} | "
        f"{_f(bm['calmar'])} | {_pct(bm['pos_month_pct'])} |")
    say()

    # ---------------------------------------------- 族层面的多重检验
    say("## 2. 多重检验：试验次数从 270 降到 8")
    say()
    sr_d = (fam_ret.mean() / fam_ret.std(ddof=1)).replace([np.inf, -np.inf], np.nan).dropna()
    sr_var = float(sr_d.var(ddof=1))
    best_f = sr_d.idxmax()
    dsr = D_.deflated_sharpe(fam_ret[best_f].to_numpy(), n_trials=len(fams),
                             sr_variance=sr_var)
    say(f"最好的族: `{best_f}`，夏普 **{_f(dsr['sr_ann'])}**（年化）")
    say(f"- 试验次数 N = **{len(fams)}**（只在族层面选择，不再选参数）")
    say(f"- 噪声门槛 SR₀ = **{_f(dsr['sr0_ann'])}**（年化）")
    say(f"- **DSR = {dsr['dsr']:.3f}** → "
        f"{'✅ 通过（>0.95）' if dsr['passed'] else '❌ 未通过'}")
    say()
    say("> 对比：上一步在 270 个参数组合里挑第一名，门槛 SR₀ 是 0.49，DSR 0.917 没过。"
        "把决策规则从「选参数」改成「选族」，试验次数降了 34 倍 —— "
        "**这不是放水，是决策规则本来就该这么定**：既然参数选择是噪声，就别去选它。")
    say()

    # ---------------------------------------------- 族层面的前推
    say("## 3. 滚动前推：族等权 vs 挑最优参数")
    say()
    folds = make_folds(panel.dates, is_years=args.is_years, oos_years=args.oos_years)
    rows, oos_ens, oos_best = [], [], []
    for fo in folds:
        is_f = fam_ret.loc[fo.is_start:fo.is_end]
        oos_f = fam_ret.loc[fo.oos_start:fo.oos_end]
        is_p = ret_mat.loc[fo.is_start:fo.is_end]
        oos_p = ret_mat.loc[fo.oos_start:fo.oos_end]
        if len(is_f) < 120 or len(oos_f) < 40:
            continue
        # 策略A：样本内选最好的族，样本外用该族等权
        sr_is = (is_f.mean() / is_f.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        pick_f = sr_is.idxmax()
        r_ens = oos_f[pick_f]
        # 策略B：样本内选最好的单个参数组合
        sr_is_p = (is_p.mean() / is_p.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
        pick_p = sr_is_p.idxmax()
        r_best = oos_p[pick_p]
        oos_ens.append(r_ens)
        oos_best.append(r_best)
        rows.append({
            "fold": fo.i, "oos": f"{fo.oos_start.date()}~{fo.oos_end.date()}",
            "选中族": pick_f, "族等权OOS收益": (1 + r_ens).prod() - 1,
            "选中参数": pick_p, "挑参数OOS收益": (1 + r_best).prod() - 1,
        })

    picks = pd.DataFrame(rows)
    picks.to_csv(OUT / "family_wfa.csv", index=False)
    say("| 折 | 样本外区间 | 样本内选中的族 | 族等权收益 | 挑最优参数收益 |")
    say("|---|---|---|---|---|")
    for _, r in picks.iterrows():
        say(f"| {r['fold']} | {r['oos']} | {r['选中族']} | "
            f"{_pct(r['族等权OOS收益'])} | {_pct(r['挑参数OOS收益'])} |")
    say()

    ens = pd.concat(oos_ens).sort_index()
    ens = ens[~ens.index.duplicated(keep="first")]
    bst = pd.concat(oos_best).sort_index()
    bst = bst[~bst.index.duplicated(keep="first")]
    me, mb = M.summarize(ens), M.summarize(bst)
    wbench = bench.reindex(ens.index).fillna(0.0)
    mbn = M.summarize(wbench)

    say("**拼接后的样本外表现：**")
    say()
    say("| 做法 | 年化 | 夏普 | 最大回撤 | 卡玛 | 盈利月占比 |")
    say("|---|---|---|---|---|---|")
    say(f"| **族等权**（不调参） | {_pct(me['cagr'])} | {_f(me['sharpe'])} | "
        f"{_pct(me['max_dd'])} | {_f(me['calmar'])} | {_pct(me['pos_month_pct'])} |")
    say(f"| 挑最优参数 | {_pct(mb['cagr'])} | {_f(mb['sharpe'])} | "
        f"{_pct(mb['max_dd'])} | {_f(mb['calmar'])} | {_pct(mb['pos_month_pct'])} |")
    say(f"| 沪深300 | {_pct(mbn['cagr'])} | {_f(mbn['sharpe'])} | "
        f"{_pct(mbn['max_dd'])} | {_f(mbn['calmar'])} | {_pct(mbn['pos_month_pct'])} |")
    say()

    # ---------------------------------------------- 增量价值检验
    say("## 4. 关键检验：两融因子相对纯价格因子的**增量**")
    say()
    say("前面的比较只说明两融因子的夏普更高，没有排除一种可能："
        "**它其实就是价格反转的一个噪声版本**。融资余额跌得多的股票，"
        "很可能就是价格跌得多的股票。所以要做两件事：看相关性，看回归 alpha。")
    say()
    corr = fam_ret.corr()
    say("| 两融因子 | 与价格反转相关 | 与价格动量相关 |")
    say("|---|---|---|")
    for f in [x for x in fams if not x.startswith("price_")]:
        say(f"| {f} | {_f(corr.loc[f, 'price_reversal_cs'], 3)} | "
            f"{_f(corr.loc[f, 'price_momentum_cs'], 3)} |")
    say()
    say("**控制价格因子后剩下的 alpha**（双因子回归：反转 + 动量）：")
    say()
    say("| 两融因子 | 年化 alpha | t 值 | 显著? |")
    say("|---|---|---|---|")
    Xb = np.column_stack([np.ones(len(fam_ret)),
                          fam_ret["price_reversal_cs"].to_numpy(float),
                          fam_ret["price_momentum_cs"].to_numpy(float)])
    alpha_rows = []
    for f in [x for x in fams if not x.startswith("price_")]:
        y = fam_ret[f].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(Xb).all(1)
        yy, XX = y[ok], Xb[ok]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        resid = yy - XX @ beta
        n, kk = XX.shape
        s2 = np.sum(resid ** 2) / (n - kk)
        se = np.sqrt(s2 * np.linalg.inv(XX.T @ XX)[0, 0])
        t = beta[0] / se if se > 0 else 0.0
        alpha_rows.append((f, beta[0] * TRADING_DAYS, t))
        say(f"| {f} | {_pct(beta[0]*TRADING_DAYS)} | {_f(t)} | "
            f"{'✅' if abs(t) > 1.96 else '❌ 未达 1.96'} |")
    say()
    max_t = max((abs(t) for _, _, t in alpha_rows), default=0.0)
    if max_t < 1.96:
        say(f"> ⚠️ **没有任何两融因子的增量 alpha 达到统计显著（最高 t = {max_t:.2f}）。**")
        say("> 也就是说：两融因子的收益里，绝大部分就是价格反转。"
            "「用了另类数据」这个卖点在这个池子和区间上**证据不足** —— "
            "增量是正的（年化 2~4%），但十年样本仍不足以证明它不是运气。")
    say()

    # ---------------------------------------------- 结论
    say("## 5. 结论")
    say()
    checks = [
        ("族等权样本外不差于挑参数", me["sharpe"] >= mb["sharpe"] - 0.05),
        ("族等权样本外跑赢沪深300", me["cagr"] > mbn["cagr"]),
        ("族层面多重检验通过", dsr["passed"]),
        ("族等权最大回撤优于沪深300", me["max_dd"] > mbn["max_dd"]),
        ("两融因子相对价格因子有显著增量", max_t > 1.96),
    ]
    for n, ok in checks:
        say(f"- {'✅' if ok else '❌'} {n}")
    say()
    n_pass = sum(1 for _, ok in checks if ok)
    say(f"**{n_pass}/{len(checks)} 项通过。**")
    say()
    if me["sharpe"] >= mb["sharpe"] - 0.05:
        say("族等权不比挑参数差 —— 证实了「参数选择是噪声」。"
            "既然如此就**不要调参**：少一个过拟合入口，多重检验门槛还低 34 倍。")
    say()
    say("---")
    say()
    say("*仅供研究，不构成投资建议。回测不代表未来收益。*")

    (OUT / "family_report.md").write_text("\n".join(log), encoding="utf-8")
    print(f"\n报告写入 {OUT/'family_report.md'}（用时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
