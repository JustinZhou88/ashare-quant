"""随机策略基准 —— 「瞎买也能赚这么多吗？」

多数 A股策略的收益其实来自两件与策略无关的事：
  1. 长期持有本身有正的市场 beta；
  2. 在牛市里持仓时间长。

所以真正的对照组不是"0 收益"，而是**交易频率、持仓周期、平均仓位都相同的
随机策略**。如果你的策略排不进随机组的前 5%，它就没有 alpha。

这一步比夏普比率更能说服人：它把"运气"具体量化成了一个分布。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS, BacktestConfig
from ..engine.panel import Panel, run


def random_signal(panel: Panel, exposure: float, hold_days: int,
                  rng: np.random.Generator) -> pd.DataFrame:
    """生成平均仓位≈exposure、平均持仓≈hold_days 的随机信号。"""
    n_t, n_s = panel.close.shape
    hold = max(int(round(hold_days)), 1)
    # 持仓 hold 天，则入场概率 p 满足 1-(1-p)^hold ≈ exposure
    p = 1.0 - (1.0 - min(max(exposure, 1e-4), 0.99)) ** (1.0 / hold)
    entries = rng.random((n_t, n_s)) < p
    e = pd.DataFrame(entries, index=panel.dates, columns=panel.symbols)
    # 入场后持有 hold 天 = 过去 hold 天内出现过入场
    return e.rolling(hold, min_periods=1).max().astype(float)


def random_benchmark(panel: Panel, cfg: BacktestConfig, exposure: float,
                     hold_days: float, n_sims: int = 100, seed: int = 0,
                     mask: pd.DataFrame | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """跑 n_sims 个随机策略，返回它们的年化夏普 / 年化收益分布。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sims):
        sig = random_signal(panel, exposure, hold_days, rng)
        res = run(panel, sig, cfg, mask=mask, with_trades=False)
        r = res.ret
        sd = r.std(ddof=1)
        rows.append({
            "sim": i,
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
            "cagr": float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1) if len(r) else 0.0,
            "exposure": res.exposure,
        })
        if verbose and (i + 1) % 20 == 0:
            print(f"    随机基准 {i + 1}/{n_sims}", flush=True)
    return pd.DataFrame(rows)


def percentile_of(value: float, dist: np.ndarray) -> float:
    """value 在分布中的百分位（0~100）。"""
    d = np.asarray(dist, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    return float((d < value).mean() * 100)
