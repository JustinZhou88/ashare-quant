"""交易记录归因 —— 对应流程第 ③ 步「把交易记录交给 AI 分析」。

这里生成的不是给人看的漂亮图表，而是**给模型看的结构化事实**：
每一笔交易赚了多少、在什么市场环境下发生、浮盈最高到过多少、
利润集中在哪几笔、参数挪一格会不会塌。

把 report.md + trades.csv 丢给 Claude，问它这四个问题：
  1. 这个策略赚钱的真实来源是什么？是选股，还是单纯的市场 beta？
  2. 亏损交易有没有共同特征（板块、市值、波动率、持仓天数）？
  3. 参数曲面是"高原"还是"孤峰"？孤峰=过拟合。
  4. 如果剔除贡献最大的 5 笔交易，策略还成立吗？
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


# ---------------------------------------------------------------- 市场环境
def label_regimes(index_df: pd.DataFrame) -> pd.DataFrame:
    """给每个交易日打市场环境标签（只用当日及历史数据，无未来函数）。"""
    c = index_df["close"]
    ma60 = c.rolling(60, min_periods=60).mean()
    ret20 = c.pct_change(20)
    vol20 = c.pct_change().rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    # 用过去 250 日的分位排名，避免用到未来信息
    vol_rank = vol20.rolling(250, min_periods=100).rank(pct=True)

    trend = pd.Series("震荡", index=c.index, dtype=object)
    trend[(c > ma60) & (ret20 > 0.02)] = "上涨"
    trend[(c < ma60) & (ret20 < -0.02)] = "下跌"

    vol_bucket = pd.Series("中波动", index=c.index, dtype=object)
    vol_bucket[vol_rank <= 0.33] = "低波动"
    vol_bucket[vol_rank >= 0.67] = "高波动"

    return pd.DataFrame({"trend": trend, "vol_bucket": vol_bucket,
                         "idx_ret20": ret20, "idx_vol20": vol20})


def attach_context(trades: pd.DataFrame, regimes: pd.DataFrame,
                   index_df: pd.DataFrame) -> pd.DataFrame:
    """给每笔交易补上入场时的市场环境 + 同期指数收益（用于算超额）。"""
    if trades.empty:
        return trades
    t = trades.copy()
    reg = regimes.reindex(pd.DatetimeIndex(t["entry_date"]), method="ffill")
    t["trend"] = reg["trend"].to_numpy()
    t["vol_bucket"] = reg["vol_bucket"].to_numpy()

    ic = index_df["close"]
    entry_px = ic.reindex(pd.DatetimeIndex(t["entry_date"]), method="ffill").to_numpy()
    exit_px = ic.reindex(pd.DatetimeIndex(t["exit_date"]), method="ffill").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t["idx_ret"] = exit_px / entry_px - 1.0
    t["excess_ret"] = t["ret_net"] - t["idx_ret"]      # 相对指数的超额
    t["year"] = pd.DatetimeIndex(t["entry_date"]).year
    return t


# ---------------------------------------------------------------- 归因表
def _agg(g: pd.DataFrame) -> pd.Series:
    r = g["ret_net"]
    wins = r[r > 0]
    losses = r[r <= 0]
    gl = -losses.sum()
    return pd.Series({
        "笔数": len(r),
        "胜率": (r > 0).mean(),
        "平均收益": r.mean(),
        "平均盈利": wins.mean() if len(wins) else 0.0,
        "平均亏损": losses.mean() if len(losses) else 0.0,
        "盈亏比": (wins.sum() / gl) if gl > 1e-12 else np.inf,
        "总贡献": r.sum(),
        "平均超额": g["excess_ret"].mean() if "excess_ret" in g else np.nan,
    })


def by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    """分市场环境的表现 —— 回答"哪些环境最好"。"""
    if trades.empty or "trend" not in trades:
        return pd.DataFrame()
    return trades.groupby(["trend", "vol_bucket"], observed=True).apply(
        _agg, include_groups=False).round(4)


def by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "year" not in trades:
        return pd.DataFrame()
    return trades.groupby("year").apply(_agg, include_groups=False).round(4)


def by_hold_bucket(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["持仓区间"] = pd.cut(t["hold_days"], [0, 3, 5, 10, 20, 60, 10_000],
                           labels=["1-3天", "4-5天", "6-10天", "11-20天", "21-60天", "60天+"])
    return t.groupby("持仓区间", observed=True).apply(_agg, include_groups=False).round(4)


def concentration(trades: pd.DataFrame) -> dict:
    """利润集中度 —— 识别"靠几笔运气"的关键指标。"""
    if trades.empty:
        return {}
    r = trades["ret_net"].to_numpy(float)
    order = np.sort(r)[::-1]
    total_win = order[order > 0].sum()
    n_sym = trades["symbol"].nunique()
    per_sym = trades.groupby("symbol")["ret_net"].sum().sort_values(ascending=False)
    return {
        "总交易笔数": int(len(r)),
        "涉及股票数": int(n_sym),
        "前1笔占总盈利": float(order[0] / total_win) if total_win > 0 else np.nan,
        "前5笔占总盈利": float(order[:5].sum() / total_win) if total_win > 0 else np.nan,
        "前10笔占总盈利": float(order[:10].sum() / total_win) if total_win > 0 else np.nan,
        "前3只股票占总收益": float(per_sym.head(3).sum() / r.sum()) if abs(r.sum()) > 1e-12 else np.nan,
        # 下面两项是"各笔收益率之和"，不是组合净值收益 —— 只用来看稳健性
        "各笔收益之和": float(r.sum()),
        "剔除前5笔后各笔收益之和": float(np.sort(r)[:-5].sum()) if len(r) > 5 else np.nan,
        "剔除前5笔后仍为正": bool(np.sort(r)[:-5].sum() > 0) if len(r) > 5 else False,
    }


def mae_mfe(trades: pd.DataFrame) -> dict:
    """浮亏/浮盈分析 —— 止损止盈设在哪里合理。"""
    if trades.empty or "mae" not in trades:
        return {}
    t = trades
    w, l = t[t["ret_net"] > 0], t[t["ret_net"] <= 0]
    return {
        "盈利交易平均最大浮亏(MAE)": float(w["mae"].mean()) if len(w) else np.nan,
        "亏损交易平均最大浮亏(MAE)": float(l["mae"].mean()) if len(l) else np.nan,
        "盈利交易平均最大浮盈(MFE)": float(w["mfe"].mean()) if len(w) else np.nan,
        "亏损交易平均最大浮盈(MFE)": float(l["mfe"].mean()) if len(l) else np.nan,
        "MAE 5%分位(可作止损参考)": float(t["mae"].quantile(0.05)),
        "亏损交易中曾浮盈>5%的比例": float((l["mfe"] > 0.05).mean()) if len(l) else np.nan,
    }


# ---------------------------------------------------------------- 参数敏感度
def sensitivity(leaderboard: pd.DataFrame, family: str,
                metric: str = "oos_sharpe") -> pd.DataFrame:
    """同一策略族内，指标随参数怎么变。

    看的是**形状**不是最大值：邻近参数一起好 = 高原（可信）；
    只有一个点好、旁边都塌 = 孤峰（过拟合，实盘必炸）。
    """
    sub = leaderboard[leaderboard["family"] == family].copy()
    if sub.empty:
        return sub
    keys = [c for c in sub.columns if c.startswith("p_")]
    return sub[keys + [metric, "n_trades"]].sort_values(metric, ascending=False)


def plateau_score(leaderboard: pd.DataFrame, family: str,
                  metric: str = "oos_sharpe") -> float:
    """高原度：该族内 metric 的「前 25% 分位 / 最大值」。

    接近 1 = 一整片参数都好（稳健）；接近 0 = 只有一个孤峰（过拟合）。
    """
    sub = leaderboard[leaderboard["family"] == family][metric].dropna()
    if len(sub) < 4:
        return np.nan
    mx = sub.max()
    # 全族为负时高原度没有意义（除以负数会得到荒谬的值，实测出现过 -3.5e8）。
    # 一个整体亏钱的族根本不该讨论"参数稳不稳"，直接返回 0。
    if mx <= 0:
        return 0.0
    return float(sub.quantile(0.75) / mx)
