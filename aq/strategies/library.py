"""策略库：14 个策略族。

每个策略是一个函数 `fn(P, **params) -> DataFrame(0/1)`，
返回值语义是「**t 日收盘时希望持有的仓位**」，引擎负责 t+1 开盘执行。

策略只允许使用 t 日及之前的数据。所有指标都是 rolling/ewm，天然满足。
新增策略照抄任意一个函数的形状即可，注册进 FAMILIES 就能进入筛选流程。
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from . import indicators as ta


# ---------------------------------------------------------------- 趋势跟随
def ma_cross(P, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """均线金叉持有、死叉空仓。最经典也最容易被参数拟合的一族。"""
    c = P.close
    return (ta.sma(c, fast) > ta.sma(c, slow)).astype(float)


def ema_cross(P, fast: int = 10, slow: int = 30) -> pd.DataFrame:
    c = P.close
    return (ta.ema(c, fast) > ta.ema(c, slow)).astype(float)


def ma_stack(P, n1: int = 5, n2: int = 10, n3: int = 20) -> pd.DataFrame:
    """均线多头排列：短 > 中 > 长。"""
    c = P.close
    a, b, d = ta.sma(c, n1), ta.sma(c, n2), ta.sma(c, n3)
    return ((a > b) & (b > d)).astype(float)


def macd_trend(P, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    dif, dea, _ = ta.macd(P.close, fast, slow, sig)
    return ((dif > dea) & (dif > 0)).astype(float)


def roc_momentum(P, n: int = 20, thresh: float = 0.0) -> pd.DataFrame:
    """N 日涨幅超过阈值就持有。"""
    return (ta.roc(P.close, n) > thresh).astype(float)


# ---------------------------------------------------------------- 突破
def donchian(P, entry_n: int = 20, exit_n: int = 10) -> pd.DataFrame:
    """唐奇安通道：创 N 日新高入场，跌破 M 日新低出场。"""
    c, h, l = P.close, P.high, P.low
    hh = h.rolling(entry_n, min_periods=entry_n).max().shift(1)
    ll = l.rolling(exit_n, min_periods=exit_n).min().shift(1)
    return ta.hold_state(entry=c > hh, exit_=c < ll)


def turtle_lite(P, entry_n: int = 20, exit_n: int = 10,
                atr_n: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    """海龟简化版：突破入场 + ATR 跟踪止损出场。"""
    c, h, l = P.close, P.high, P.low
    a = ta.atr(h, l, c, atr_n)
    hh = h.rolling(entry_n, min_periods=entry_n).max().shift(1)
    ll = l.rolling(exit_n, min_periods=exit_n).min().shift(1)
    trail = c.rolling(entry_n, min_periods=1).max() - atr_mult * a
    return ta.hold_state(entry=c > hh, exit_=(c < ll) | (c < trail))


def vol_breakout(P, n: int = 20, vol_mult: float = 2.0, exit_n: int = 10) -> pd.DataFrame:
    """放量突破：价格创 N 日新高 **且** 成交量超过均量 vol_mult 倍。"""
    c, h, l, v = P.close, P.high, P.low, P.volume
    hh = h.rolling(n, min_periods=n).max().shift(1)
    vma = v.rolling(n, min_periods=n).mean().shift(1)
    ll = l.rolling(exit_n, min_periods=exit_n).min().shift(1)
    return ta.hold_state(entry=(c > hh) & (v > vol_mult * vma), exit_=c < ll)


def boll_breakout(P, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """突破布林上轨入场，回落中轨出场。"""
    lower, mid, upper = ta.bollinger(P.close, n, k)
    c = P.close
    return ta.hold_state(entry=c > upper, exit_=c < mid)


def chandelier(P, entry_n: int = 20, atr_n: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """吊灯止损：创新高入场，从持仓期最高价回撤 mult×ATR 出场。"""
    c, h, l = P.close, P.high, P.low
    a = ta.atr(h, l, c, atr_n)
    hh = h.rolling(entry_n, min_periods=entry_n).max().shift(1)
    stop = h.rolling(entry_n, min_periods=1).max() - mult * a
    return ta.hold_state(entry=c > hh, exit_=c < stop)


# ---------------------------------------------------------------- 均值回归
def rsi_reversion(P, n: int = 14, lo: float = 30.0, hi: float = 55.0) -> pd.DataFrame:
    """超卖买入，回到 hi 卖出。"""
    r = ta.rsi(P.close, n)
    return ta.hold_state(entry=r < lo, exit_=r > hi)


def boll_reversion(P, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """跌破布林下轨买入，回到中轨卖出。"""
    lower, mid, _ = ta.bollinger(P.close, n, k)
    c = P.close
    return ta.hold_state(entry=c < lower, exit_=c > mid)


def kdj_reversion(P, n: int = 9, k_lo: float = 20.0, k_hi: float = 80.0) -> pd.DataFrame:
    k, d, _ = ta.kdj(P.high, P.low, P.close, n)
    return ta.hold_state(entry=(k < k_lo) & (k > d), exit_=k > k_hi)


def trend_pullback(P, ma_n: int = 60, rsi_n: int = 5,
                   rsi_lo: float = 30.0, rsi_hi: float = 60.0) -> pd.DataFrame:
    """长期均线之上的短期回调买入 —— 趋势过滤 + 均值回归。"""
    c = P.close
    up_trend = c > ta.sma(c, ma_n)
    r = ta.rsi(c, rsi_n)
    return ta.hold_state(entry=up_trend & (r < rsi_lo), exit_=(r > rsi_hi) | ~up_trend)


def dip_buy(P, ma_n: int = 20, dip: float = 0.05, hold_n: int = 5) -> pd.DataFrame:
    """跌破均线一定幅度买入，持有固定天数（近似：回到均线上方出场）。"""
    c = P.close
    m = ta.sma(c, ma_n)
    entry = c < m * (1 - dip)
    exit_ = c > m
    return ta.hold_state(entry=entry, exit_=exit_)


# ---------------------------------------------------------------- 注册表
FAMILIES: dict[str, Callable] = {
    "ma_cross": ma_cross,
    "ema_cross": ema_cross,
    "ma_stack": ma_stack,
    "macd_trend": macd_trend,
    "roc_momentum": roc_momentum,
    "donchian": donchian,
    "turtle_lite": turtle_lite,
    "vol_breakout": vol_breakout,
    "boll_breakout": boll_breakout,
    "chandelier": chandelier,
    "rsi_reversion": rsi_reversion,
    "boll_reversion": boll_reversion,
    "kdj_reversion": kdj_reversion,
    "trend_pullback": trend_pullback,
    "dip_buy": dip_buy,
}
