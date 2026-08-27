"""绩效指标。

除了常规的收益/回撤，这里刻意加了几个**识别过拟合**的指标：
  - `top5_share`：前 5 笔交易贡献了多少利润。>60% 说明策略靠几次运气。
  - `t_stat`：日收益均值的 t 统计量。样本外还需要经过多重检验校正（见 validation）。
  - `pos_month_pct`：盈利月份占比。真策略应该稳定分布，不是全靠某一个月。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, int]:
    """返回 (最大回撤, 谷底日期, 最长水下天数)。"""
    if equity.empty:
        return 0.0, None, 0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin() if len(dd) else None
    underwater = (equity < peak).astype(int)
    longest, cur = 0, 0
    for u in underwater.to_numpy():
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return float(dd.min()), trough, int(longest)


def summarize(ret: pd.Series, trades: pd.DataFrame | None = None,
              exposure: float | None = None) -> dict:
    """把日收益序列 + 交易记录压成一行指标。"""
    ret = ret.dropna()
    n = len(ret)
    if n < 20:
        return {"n_days": n, "cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "n_trades": 0}

    equity = (1 + ret).cumprod()
    years = n / TRADING_DAYS
    total = float(equity.iloc[-1])
    cagr = total ** (1 / years) - 1 if total > 0 and years > 0 else -1.0
    vol = float(ret.std(ddof=1)) * np.sqrt(TRADING_DAYS)
    mu = float(ret.mean())
    sharpe = mu / float(ret.std(ddof=1)) * np.sqrt(TRADING_DAYS) if ret.std(ddof=1) > 0 else 0.0
    downside = ret[ret < 0].std(ddof=1)
    sortino = mu / float(downside) * np.sqrt(TRADING_DAYS) if downside and downside > 0 else 0.0
    mdd, trough, uw = max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < -1e-9 else 0.0
    t_stat = mu / (float(ret.std(ddof=1)) / np.sqrt(n)) if ret.std(ddof=1) > 0 else 0.0

    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) if n > 40 else pd.Series(dtype=float)
    pos_month = float((monthly > 0).mean()) if len(monthly) else 0.0

    out = {
        "n_days": n, "years": round(years, 2),
        "total_return": total - 1, "cagr": cagr, "vol": vol,
        "sharpe": sharpe, "sortino": sortino,
        "max_dd": mdd, "calmar": calmar,
        "t_stat": t_stat, "pos_month_pct": pos_month,
        "longest_underwater_days": uw,
        "exposure": exposure if exposure is not None else np.nan,
    }

    if trades is not None and len(trades):
        r = trades["ret_net"].to_numpy(float)
        wins, losses = r[r > 0], r[r <= 0]
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        top5 = float(np.sort(r)[-5:].sum())
        out.update({
            "n_trades": int(len(r)),
            "win_rate": float((r > 0).mean()),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": gross_win / gross_loss if gross_loss > 1e-12 else np.inf,
            "expectancy": float(r.mean()),
            "avg_hold_days": float(trades["hold_days"].mean()),
            # 利润集中度：前 5 笔占总盈利的比例，越高越像运气
            "top5_share": top5 / gross_win if gross_win > 1e-12 else np.nan,
        })
    else:
        out.update({"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                    "expectancy": 0.0, "avg_hold_days": 0.0, "top5_share": np.nan})
    return out


def benchmark_returns(index_df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """基准（如沪深300）在给定交易日上的日收益。"""
    px = index_df["close"].reindex(dates).ffill()
    return px.pct_change().fillna(0.0)
