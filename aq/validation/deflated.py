"""多重检验校正 —— 这是整个项目最重要的一个文件。

问题：跑 424 个策略，取最好的那个，它的夏普比率**必然**被高估。
即使 424 个策略全是纯噪声（真实夏普 = 0），最大值的期望也接近 3（年化）。
拿这个数去实盘，就是那个"回测年化 80%、实盘半年亏 30%"的经典故事。

本模块实现 Bailey & López de Prado (2014) 的 Deflated Sharpe Ratio：
  1. 先算「N 次独立试验下，纯噪声能达到的最大夏普期望」= SR₀
  2. 再问「观察到的夏普显著超过 SR₀ 的概率是多少」= DSR
DSR > 0.95 才算通过。绝大多数排行榜第一名过不了这一关 —— 这是正常的。

另外处理了一个常被忽略的细节：424 个策略高度相关（同一族只是参数微调），
不是 424 次独立试验。`effective_trials` 用相关矩阵特征值估计有效试验数，
避免过度惩罚。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS
from .stats import norm_cdf, norm_ppf, skew_kurt

_EULER = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """N 次试验下，真实夏普全为 0 时，观测到的最大夏普的期望值（日频单位）。

    E[max] ≈ √V · [(1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e))]
    """
    n = max(int(n_trials), 2)
    v = max(float(sr_variance), 1e-18)
    z1 = norm_ppf(1.0 - 1.0 / n)
    z2 = norm_ppf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(v) * ((1 - _EULER) * z1 + _EULER * z2)


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR：观测夏普真的高于 sr_benchmark 的概率（考虑偏度和峰度）。

    returns / sr_benchmark 均为**日频**单位。
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 30 or r.std(ddof=1) <= 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    sk, ku = skew_kurt(r)
    denom = 1.0 - sk * sr + (ku - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)
    return norm_cdf(z)


def effective_trials(ret_matrix: pd.DataFrame) -> float:
    """有效独立试验数（**仅供敏感性参考，不要当默认判据**）。

    出发点是合理的：`ma_cross(5,20)` 和 `ma_cross(5,30)` 几乎是同一个策略，
    按 424 次独立检验来罚似乎太狠。这里用相关矩阵特征值的参与率
    (Σλ)² / Σλ² 估计"实际做了多少次独立试验"。

    ⚠️ 但对**只做多的股票策略**，这个估计会严重偏低：它们的收益被市场 beta
    主导，相关矩阵只有一个大特征值，参与率往往只有 2~3。用它当试验次数
    等于假装自己只搜索了 3 次，会把噪声门槛压到很低 —— **方向是放水的**。

    实践建议：判据用名义试验次数（保守），把这个值作为区间的另一端来看。

    ret_matrix: 列 = 策略，行 = 日期，值 = 日收益。
    """
    m = ret_matrix.dropna(axis=1, how="all").fillna(0.0)
    m = m.loc[:, m.std() > 1e-12]
    if m.shape[1] <= 1:
        return float(max(m.shape[1], 1))
    c = np.corrcoef(m.to_numpy().T)
    c = np.nan_to_num(c, nan=0.0)
    lam = np.linalg.eigvalsh(c)
    lam = np.clip(lam, 0, None)
    s1, s2 = lam.sum(), (lam ** 2).sum()
    if s2 <= 0:
        return 1.0
    return float(max(1.0, s1 ** 2 / s2))


def deflated_sharpe(returns: np.ndarray, n_trials: int,
                    sr_variance: float) -> dict:
    """完整的 DSR 检验。

    returns:     入选策略的日收益序列
    n_trials:    试验次数（建议传 effective_trials 的结果）
    sr_variance: 所有候选策略**日频**夏普的方差
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 30 or r.std(ddof=1) <= 0:
        return {"dsr": 0.0, "sr_daily": 0.0, "sr_ann": 0.0,
                "sr0_daily": 0.0, "sr0_ann": 0.0, "n_trials": n_trials, "passed": False}

    sr_d = float(r.mean() / r.std(ddof=1))
    sr0_d = expected_max_sharpe(n_trials, sr_variance)
    dsr = probabilistic_sharpe(r, sr0_d)
    k = math.sqrt(TRADING_DAYS)
    return {
        "dsr": dsr,
        "sr_daily": sr_d, "sr_ann": sr_d * k,
        "sr0_daily": sr0_d, "sr0_ann": sr0_d * k,   # 噪声能达到的门槛（年化）
        "n_trials": n_trials,
        "passed": bool(dsr > 0.95),
    }


def bonferroni_threshold(n_trials: int, alpha: float = 0.05) -> float:
    """Bonferroni 校正后的 t 值门槛 —— 比 DSR 粗暴，但一句话能讲明白。"""
    n = max(int(n_trials), 1)
    return norm_ppf(1.0 - alpha / (2 * n))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH 控制 FDR，返回布尔数组表示哪些策略通过。"""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed_sorted = ranked <= thresh
    k = np.flatnonzero(passed_sorted)
    out = np.zeros(n, bool)
    if len(k):
        cut = k.max()
        out[order[:cut + 1]] = True
    return out
