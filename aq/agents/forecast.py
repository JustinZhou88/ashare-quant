"""历史类比预测 —— 给推荐标的一个**有数据支撑**的价格区间与持有周期。

为什么不让 LLM 给目标价：它说「目标 12.5 元」背后没有任何模型，
是凭空的虚假精确。分析师提示词里明确禁止了这件事。

这里的做法是**历史类比（analog forecast）**：
在全市场历史上找出「和这只票现在长得像」的样本
（相近的波动率、相近的跌幅深度、相近的 RSI、同样处于因子候选池），
看它们进场后实际发生了什么 —— 最高涨到哪、第几天见顶、多少比例摸到目标。

输出的是**分布**不是点估计。「25%~75% 分位在 +3.2%~+14.8%」比
「目标价 12.5 元」诚实得多，也更有用 —— 你能据此判断风险回报是否值得。

所有类比样本都取自 as_of 之前，不使用未来数据。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _forward_paths(panel, horizon: int) -> dict[str, np.ndarray]:
    """预计算每个 (日期, 股票) 在未来 horizon 日的路径特征。

    执行假设与回测一致：**次日开盘买入**。
    返回的数组在最后 horizon 行是 NaN（未来数据不足）。
    """
    op = panel.open.to_numpy(float)
    hi = panel.high.to_numpy(float)
    lo = panel.low.to_numpy(float)
    cl = panel.close.to_numpy(float)
    n_t, n_s = cl.shape

    entry = np.full((n_t, n_s), np.nan)
    entry[:-1] = op[1:]                      # t 日信号 -> t+1 开盘买入

    mfe = np.full((n_t, n_s), np.nan)        # 期间最大涨幅
    mae = np.full((n_t, n_s), np.nan)        # 期间最大跌幅
    day_peak = np.full((n_t, n_s), np.nan)   # 第几天见顶
    ret_end = np.full((n_t, n_s), np.nan)    # 到期收益

    for t in range(n_t - horizon - 1):
        e = entry[t]
        seg_hi = hi[t + 1: t + 1 + horizon]
        seg_lo = lo[t + 1: t + 1 + horizon]
        # 停牌或未上市的股票整段是 NaN，nanmax/nanmin 会刷 All-NaN 警告。
        # 这是预期内的（不是错误），直接静音，别让它淹没真正该看的输出。
        with np.errstate(invalid="ignore", divide="ignore"), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mfe[t] = np.nanmax(seg_hi, axis=0) / e - 1.0
            mae[t] = np.nanmin(seg_lo, axis=0) / e - 1.0
            ret_end[t] = cl[t + horizon] / e - 1.0
            day_peak[t] = np.nanargmax(
                np.where(np.isfinite(seg_hi), seg_hi, -np.inf), axis=0) + 1
    return {"entry": entry, "mfe": mfe, "mae": mae,
            "day_peak": day_peak, "ret_end": ret_end}


def _features(panel) -> dict[str, pd.DataFrame]:
    cl = panel.close
    ret = cl.pct_change()
    return {
        "vol60": ret.rolling(60, min_periods=40).std() * np.sqrt(244),
        "ret20": cl.pct_change(20),
        "ret60": cl.pct_change(60),
        "pos60": (cl - panel.low.rolling(60, min_periods=40).min())
        / (panel.high.rolling(60, min_periods=40).max()
           - panel.low.rolling(60, min_periods=40).min()).replace(0, np.nan),
    }


class AnalogForecaster:
    """一次性预计算，之后每只股票的查询都很快。"""

    def __init__(self, panel, horizon: int = 15, min_analogs: int = 120):
        self.panel = panel
        self.horizon = horizon
        self.min_analogs = min_analogs
        self.paths = _forward_paths(panel, horizon)
        self.feat = _features(panel)
        self.dates = panel.dates
        self.symbols = panel.symbols

    def forecast(self, sym: str, as_of: pd.Timestamp,
                 tol_vol: float = 0.25, tol_ret: float = 0.15) -> dict:
        """对 sym 在 as_of 这天的状态，找历史类比样本并汇总分布。

        只使用 as_of **之前**结束的样本 —— 类比样本的完整未来路径必须
        在 as_of 之前就已经走完，否则就是未来函数。
        """
        if sym not in self.symbols:
            return {}
        j = self.symbols.index(sym)
        i = int(np.searchsorted(self.dates, as_of))
        if i >= len(self.dates):
            i = len(self.dates) - 1

        cur = {k: float(v.iloc[i, j]) for k, v in self.feat.items()}
        if not all(np.isfinite(list(cur.values()))):
            return {}

        # 类比样本的观察窗口必须完全落在 as_of 之前
        cutoff = max(i - self.horizon - 1, 0)
        vol = self.feat["vol60"].to_numpy()[:cutoff]
        r20 = self.feat["ret20"].to_numpy()[:cutoff]
        pos = self.feat["pos60"].to_numpy()[:cutoff]

        ok = (np.abs(vol - cur["vol60"]) <= tol_vol * max(cur["vol60"], 0.1)) \
            & (np.abs(r20 - cur["ret20"]) <= tol_ret) \
            & (np.abs(pos - cur["pos60"]) <= 0.25)

        mfe = self.paths["mfe"][:cutoff][ok]
        mae = self.paths["mae"][:cutoff][ok]
        dpk = self.paths["day_peak"][:cutoff][ok]
        ren = self.paths["ret_end"][:cutoff][ok]
        m = np.isfinite(mfe) & np.isfinite(mae) & np.isfinite(ren)
        mfe, mae, dpk, ren = mfe[m], mae[m], dpk[m], ren[m]

        if len(mfe) < self.min_analogs:
            return {"n_analogs": int(len(mfe)), "enough": False}

        px = float(self.panel.raw_close.iloc[i, j])
        q = [10, 25, 50, 75, 90]
        return {
            "n_analogs": int(len(mfe)), "enough": True,
            "price_at_scan": px,
            "horizon": self.horizon,
            "mfe_q": {f"p{k}": float(np.percentile(mfe, k)) for k in q},
            "peak_price_q": {f"p{k}": px * (1 + float(np.percentile(mfe, k)))
                             for k in q},
            "mae_q": {f"p{k}": float(np.percentile(mae, k)) for k in q},
            "ret_end_q": {f"p{k}": float(np.percentile(ren, k)) for k in q},
            "days_to_peak_median": float(np.median(dpk)),
            "days_to_peak_q25": float(np.percentile(dpk, 25)),
            "days_to_peak_q75": float(np.percentile(dpk, 75)),
            "p_hit_15": float((mfe >= 0.15).mean()),
            "p_hit_15_before_stop": float(((mfe >= 0.15) & (mae > -0.15)).mean()),
            "p_stop_15": float((mae <= -0.15).mean()),
            "p_positive_end": float((ren > 0).mean()),
            "expected_ret": float(np.mean(ren)),
        }


def to_text(fc: dict, name: str = "") -> str:
    """渲染成人能读的预测块。强调这是**分布**不是承诺。"""
    if not fc:
        return "  （无法生成历史类比：数据不足）"
    if not fc.get("enough"):
        return f"  （历史类比样本仅 {fc.get('n_analogs',0)} 个，不足以给出分布）"

    pk, mf = fc["peak_price_q"], fc["mfe_q"]
    L = [f"  基于 {fc['n_analogs']} 个历史相似样本（{fc['horizon']} 个交易日窗口）：",
         f"    预期最高价:  25分位 {pk['p25']:.2f}({mf['p25']:+.1%})  "
         f"中位 {pk['p50']:.2f}({mf['p50']:+.1%})  "
         f"75分位 {pk['p75']:.2f}({mf['p75']:+.1%})",
         f"    乐观情形(90分位): {pk['p90']:.2f} ({mf['p90']:+.1%})",
         f"    见顶时间:    中位 第 {fc['days_to_peak_median']:.0f} 个交易日"
         f"（25~75分位: 第 {fc['days_to_peak_q25']:.0f}~{fc['days_to_peak_q75']:.0f} 日）",
         f"    摸到 +15% 概率: {fc['p_hit_15']:.1%}"
         f"（且未先触发 -15% 止损: {fc['p_hit_15_before_stop']:.1%}）",
         f"    触发 -15% 止损概率: {fc['p_stop_15']:.1%}",
         f"    持有到期为正的概率: {fc['p_positive_end']:.1%}，"
         f"期望收益 {fc['expected_ret']:+.2%}"]
    return "\n".join(L)
