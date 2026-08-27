"""技术指标 —— 全部对 DataFrame（日期 × 股票）整体运算，一次算完所有股票。

所有指标都只用当前及历史数据（rolling / ewm 天然后视），不存在未来函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def ema(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder RSI。"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ru = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rd = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(rd != 0, 100.0)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    pc = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - pc).abs(), (low - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.DataFrame, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9):
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=sig, adjust=False, min_periods=sig).mean()
    return dif, dea, (dif - dea) * 2


def kdj(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 9):
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    rsv = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=n).mean()
    d = k.ewm(alpha=1 / 3, adjust=False, min_periods=n).mean()
    return k, d, 3 * k - 2 * d


def roc(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close.pct_change(n)


def hold_state(entry: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """把「入场条件 / 出场条件」变成「持仓状态」：入场后一直持有到出场。

    同一天同时触发时**出场优先**（保守）。全向量化，无 Python 循环。
    """
    s = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
    s = s.mask(entry.fillna(False), 1.0)
    s = s.mask(exit_.fillna(False), 0.0)
    return s.ffill().fillna(0.0)
