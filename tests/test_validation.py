"""验证层测试 —— 核心是用蒙特卡洛检查 Deflated Sharpe 的公式对不对。

跑法： .venv/bin/python tests/test_validation.py

最关键的一项：造 424 个**真实夏普为 0** 的随机策略，看观测到的最大夏普
是多少。如果 expected_max_sharpe 预测得准，就证明"424 个策略里挑第一名，
即使全是噪声也能挑出高夏普"这件事被正确量化了。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aq.config import TRADING_DAYS                              # noqa: E402
from aq.validation.deflated import (                            # noqa: E402
    benjamini_hochberg, deflated_sharpe, effective_trials,
    expected_max_sharpe, probabilistic_sharpe)
from aq.validation.stats import norm_cdf, norm_ppf              # noqa: E402
from aq.validation.wfa import make_folds                        # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def test_norm() -> None:
    for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.999):
        check(f"norm_cdf(norm_ppf({p})) 往返一致", abs(norm_cdf(norm_ppf(p)) - p) < 1e-6)
    check("norm_ppf(0.975) ≈ 1.96", abs(norm_ppf(0.975) - 1.959964) < 1e-4,
          f"{norm_ppf(0.975):.6f}")


def test_expected_max_is_calibrated() -> None:
    """蒙特卡洛：424 个纯噪声策略，实测最大夏普 vs 公式预测。"""
    rng = np.random.default_rng(0)
    n_trials, n_days, n_rep = 424, 2500, 40
    daily_vol = 0.01
    observed = []
    for _ in range(n_rep):
        r = rng.normal(0.0, daily_vol, size=(n_days, n_trials))   # 真实夏普 = 0
        sr = r.mean(axis=0) / r.std(axis=0, ddof=1)
        observed.append(sr.max())
    obs_mean = float(np.mean(observed))

    # 公式需要的是这批夏普估计量的方差；理论上 ≈ 1/T
    sr_var = 1.0 / n_days
    pred = expected_max_sharpe(n_trials, sr_var)

    rel = abs(pred - obs_mean) / obs_mean
    check("E[max Sharpe] 公式与蒙特卡洛一致（误差<10%）", rel < 0.10,
          f"预测 {pred*np.sqrt(TRADING_DAYS):.2f} vs 实测 "
          f"{obs_mean*np.sqrt(TRADING_DAYS):.2f}（年化），相对误差 {rel:.1%}")

    ann = obs_mean * np.sqrt(TRADING_DAYS)
    check("10年样本 × 424 个纯噪声策略 -> 最大年化夏普 ≈ 0.9", 0.7 < ann < 1.3,
          f"实测 {ann:.2f} —— 即：10年回测里挑出的夏普 0.9 策略，可能什么都不是")

    # 样本越短，噪声天花板越高。这是"两年回测夏普 2"毫无意义的原因。
    for days, label in [(500, "2年"), (1250, "5年")]:
        pred_ann = expected_max_sharpe(n_trials, 1.0 / days) * np.sqrt(TRADING_DAYS)
        check(f"{label}样本的噪声夏普天花板 = {pred_ann:.2f}", True,
              f"{label}回测 × 424 个策略，纯噪声就能挑出年化夏普 {pred_ann:.2f}")


def test_deflated_rejects_noise() -> None:
    """把噪声里挑出的最优策略喂给 DSR，应该判为不通过。"""
    rng = np.random.default_rng(7)
    n_trials, n_days = 424, 2500
    r = rng.normal(0.0, 0.01, size=(n_days, n_trials))
    sr = r.mean(axis=0) / r.std(axis=0, ddof=1)
    best = r[:, int(np.argmax(sr))]

    df = pd.DataFrame(r)
    res = deflated_sharpe(best, n_trials=effective_trials(df),
                          sr_variance=float(np.var(sr, ddof=1)))
    check("噪声中的最优策略被 DSR 拒绝", not res["passed"],
          f"DSR={res['dsr']:.3f}, 夏普={res['sr_ann']:.2f}, 门槛={res['sr0_ann']:.2f}")
    check("被拒绝的策略夏普为正（不是靠「看起来差」才拒绝的）", res["sr_ann"] > 0.5,
          f"夏普 {res['sr_ann']:.2f} 看着还行，但扣掉选择偏差后不显著")


def test_deflated_accepts_real_edge() -> None:
    """真有 alpha 的策略（年化夏普 ~2.5，且只试了 5 次）应该通过。"""
    rng = np.random.default_rng(11)
    n_days = 2500
    target_ann = 2.5
    mu = target_ann / np.sqrt(TRADING_DAYS) * 0.01
    good = rng.normal(mu, 0.01, size=n_days)
    res = deflated_sharpe(good, n_trials=5, sr_variance=1.0 / n_days)
    check("真实 alpha（试验次数少）能通过 DSR", res["passed"],
          f"DSR={res['dsr']:.3f}, 夏普={res['sr_ann']:.2f}, 门槛={res['sr0_ann']:.2f}")


def test_effective_trials() -> None:
    rng = np.random.default_rng(3)
    n = 500
    indep = pd.DataFrame(rng.normal(size=(n, 50)))
    check("50 个独立策略 -> 有效试验数接近 50",
          40 <= effective_trials(indep) <= 50, f"{effective_trials(indep):.1f}")

    base = rng.normal(size=n)
    dup = pd.DataFrame({f"s{i}": base + rng.normal(0, 1e-6, n) for i in range(50)})
    check("50 个几乎相同的策略 -> 有效试验数接近 1",
          effective_trials(dup) < 2.0, f"{effective_trials(dup):.2f}")


def test_psr() -> None:
    rng = np.random.default_rng(5)
    r = rng.normal(0.0008, 0.01, 2000)          # 年化夏普 ~1.25
    check("PSR(基准=0) 接近 1", probabilistic_sharpe(r, 0.0) > 0.99,
          f"{probabilistic_sharpe(r, 0.0):.4f}")
    high = 0.2                                   # 日频门槛极高
    check("PSR(基准极高) 接近 0", probabilistic_sharpe(r, high) < 0.01,
          f"{probabilistic_sharpe(r, high):.4f}")


def test_bh() -> None:
    p = np.array([0.001, 0.008, 0.02, 0.3, 0.7])
    out = benjamini_hochberg(p, alpha=0.05)
    check("BH 挑出显著项", bool(out[0] and out[1]) and not bool(out[3] or out[4]),
          f"{out}")


def test_folds_no_leakage() -> None:
    """滚动窗口必须严格 样本内 < 样本外，且各折样本外不重叠。"""
    dates = pd.bdate_range("2015-01-01", "2026-07-31")
    folds = make_folds(dates, is_years=4, oos_years=1)
    check("生成了多折", len(folds) >= 4, f"{len(folds)} 折")

    ok_order = all(f.is_end < f.oos_start for f in folds)
    check("每折样本内结束早于样本外开始（无泄漏）", ok_order)

    ok_disjoint = all(folds[i].oos_end <= folds[i + 1].oos_start
                      for i in range(len(folds) - 1))
    check("各折样本外区间不重叠", ok_disjoint)

    ok_forward = all(folds[i].oos_start < folds[i + 1].oos_start
                     for i in range(len(folds) - 1))
    check("样本外窗口随时间前推", ok_forward)


def main() -> int:
    for fn in [test_norm, test_expected_max_is_calibrated, test_deflated_rejects_noise,
               test_deflated_accepts_real_edge, test_effective_trials, test_psr,
               test_bh, test_folds_no_leakage]:
        try:
            fn()
        except Exception as e:                                  # noqa: BLE001
            check(f"{fn.__name__} 抛异常", False, f"{type(e).__name__}: {e}")

    ok = sum(1 for _, p, _ in CHECKS if p)
    print(f"\n{'='*72}\n反过拟合验证层测试\n{'='*72}")
    for name, passed, detail in CHECKS:
        mark = "✅" if passed else "❌"
        print(f"{mark} {name}" + (f"\n     {detail}" if detail else ""))
    print(f"{'='*72}\n{ok}/{len(CHECKS)} 通过\n")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
