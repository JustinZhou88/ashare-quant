"""滚动前推验证（Walk-Forward Analysis）。

这一步回答的问题不是"哪个策略历史表现最好"，而是：
**「在历史上选策略」这件事本身，到底有没有预测力？**

做法：把历史切成若干折，每折用前 N 年（样本内）挑出最好的策略，
拿它在后 1 年（样本外）的真实表现记账。把所有样本外拼起来，
才是你能指望的收益 —— 排行榜上那条漂亮的曲线不是。

最有价值的输出是 `is_oos_corr`：样本内夏普与样本外夏普的**秩相关**。
如果它长期在 0 附近，说明你的整个筛选流程就是在挑噪声，
再优化参数也没用 —— 该换特征，不是换参数。

切片的合法性：引擎是因果的（t 日收益只依赖 t 日及以前的数据），
所以「全样本跑一次、再按窗口切片」等价于「每折单独跑」，但快 N 倍。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


@dataclass
class Fold:
    i: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    def __str__(self) -> str:
        return (f"折{self.i}: 样本内 {self.is_start.date()}~{self.is_end.date()} "
                f"| 样本外 {self.oos_start.date()}~{self.oos_end.date()}")


def make_folds(dates: pd.DatetimeIndex, is_years: float = 4.0,
               oos_years: float = 1.0, anchored: bool = False) -> list[Fold]:
    """切分滚动窗口。anchored=True 时样本内起点固定（累积扩张窗口）。"""
    dates = pd.DatetimeIndex(dates).sort_values()
    start, end = dates[0], dates[-1]
    folds: list[Fold] = []
    i = 0
    is_lo = start
    is_hi = start + pd.DateOffset(days=int(is_years * 365.25))
    while True:
        oos_lo = is_hi + pd.Timedelta(days=1)
        oos_hi = oos_lo + pd.DateOffset(days=int(oos_years * 365.25))
        if oos_lo >= end:
            break
        oos_hi = min(oos_hi, end)
        if len(dates[(dates >= oos_lo) & (dates <= oos_hi)]) < 60:
            break
        folds.append(Fold(i, is_lo, is_hi, oos_lo, oos_hi))
        i += 1
        is_hi = oos_hi
        if not anchored:
            is_lo = is_hi - pd.DateOffset(days=int(is_years * 365.25))
    return folds


def _sharpe(r: pd.Series | pd.DataFrame):
    sd = r.std(ddof=1)
    mu = r.mean()
    out = mu / sd.replace(0, np.nan) * np.sqrt(TRADING_DAYS) if isinstance(sd, pd.Series) \
        else (mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
    return out.fillna(0.0) if isinstance(out, pd.Series) else out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 5:
        return np.nan
    ra, rb = both.iloc[:, 0].rank(), both.iloc[:, 1].rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(ra.corr(rb))


@dataclass
class WFAResult:
    oos_ret: pd.Series                  # 拼接起来的样本外日收益（真正可信的曲线）
    folds: list[Fold]
    picks: pd.DataFrame                 # 每折选了谁、样本内外表现如何
    is_oos_corr: pd.Series              # 每折的 IS/OOS 夏普秩相关
    top_k: int

    @property
    def mean_is_oos_corr(self) -> float:
        return float(np.nanmean(self.is_oos_corr.to_numpy()))


def walk_forward(ret_matrix: pd.DataFrame, trades_matrix: pd.DataFrame,
                 folds: list[Fold], top_k: int = 1,
                 min_trades_is: int = 30) -> WFAResult:
    """ret_matrix / trades_matrix: 行=日期, 列=策略ID（日收益 / 当日开仓笔数）。"""
    rows, corrs, oos_chunks = [], {}, []

    for f in folds:
        is_r = ret_matrix.loc[f.is_start:f.is_end]
        oos_r = ret_matrix.loc[f.oos_start:f.oos_end]
        if len(is_r) < 120 or len(oos_r) < 40:
            continue

        n_tr = trades_matrix.loc[f.is_start:f.is_end].sum()
        eligible = n_tr[n_tr >= min_trades_is].index
        if len(eligible) == 0:
            continue

        is_sr = _sharpe(is_r[eligible])
        oos_sr_all = _sharpe(oos_r[eligible])
        corrs[f.i] = _spearman(is_sr, oos_sr_all)

        picked = is_sr.nlargest(top_k).index.tolist()
        # top_k 个策略等权组合，作为该折的样本外收益
        chunk = oos_r[picked].mean(axis=1)
        oos_chunks.append(chunk)

        for p in picked:
            rows.append({
                "fold": f.i,
                "is_period": f"{f.is_start.date()}~{f.is_end.date()}",
                "oos_period": f"{f.oos_start.date()}~{f.oos_end.date()}",
                "strategy": p,
                "is_sharpe": float(is_sr[p]),
                "oos_sharpe": float(oos_sr_all[p]),
                "is_trades": int(n_tr[p]),
                "oos_return": float((1 + oos_r[p]).prod() - 1),
                "n_eligible": len(eligible),
            })

    oos = pd.concat(oos_chunks).sort_index() if oos_chunks else pd.Series(dtype=float)
    oos = oos[~oos.index.duplicated(keep="first")]
    return WFAResult(
        oos_ret=oos, folds=folds,
        picks=pd.DataFrame(rows),
        is_oos_corr=pd.Series(corrs, name="is_oos_spearman"),
        top_k=top_k,
    )
