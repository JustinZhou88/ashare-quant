"""候选股票的信息包构建 —— 喂给 LLM 的**唯一**信息来源。

铁律：**只允许使用 scan_date 及以前的数据**。

这不是洁癖。LLM 的训练语料里包含了 scan_date 之后发生的事，如果你在 prompt 里
再喂给它未来数据，你就同时踩了两个未来函数，回测结果会好得离谱且完全不可实现。
本模块所有取数都以 `.loc[:date]` 结尾，就是为了让这条铁律在代码层面可审计。

输出两份东西：
  - `text`: 给 LLM 看的中文结构化摘要
  - `facts`: 给记账用的机器可读字典（日后做统计检验要用）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategies import indicators as ta


def _safe(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}%}" if pct else f"{x:.{nd}f}"


def build_facts(panel, D: dict, sym: str, date: pd.Timestamp,
                factor_rank: float | None = None,
                meta: pd.DataFrame | None = None) -> dict:
    """抽取单只股票在 date 这天的全部可见事实。"""
    cl = panel.close.loc[:date, sym].dropna()
    if len(cl) < 60:
        return {}
    rc = panel.raw_close.loc[:date, sym].dropna()
    amt = panel.amount.loc[:date, sym]
    hi = panel.high.loc[:date, sym]
    lo = panel.low.loc[:date, sym]

    def ret(n):
        return float(cl.iloc[-1] / cl.iloc[-1 - n] - 1) if len(cl) > n else np.nan

    ma = {n: float(cl.iloc[-n:].mean()) for n in (5, 20, 60, 120) if len(cl) >= n}
    last = float(cl.iloc[-1])
    vol60 = float(cl.pct_change().iloc[-60:].std() * np.sqrt(244))
    amt20 = float(amt.iloc[-20:].mean())
    amt5 = float(amt.iloc[-5:].mean())
    hi60, lo60 = float(hi.iloc[-60:].max()), float(lo.iloc[-60:].min())
    rsi = float(ta.rsi(cl.to_frame(), 14).iloc[-1, 0]) if len(cl) > 20 else np.nan

    f = {
        "symbol": sym,
        "name": panel.names.get(sym, ""),
        "date": str(pd.Timestamp(date).date()),
        "price": float(rc.iloc[-1]) if len(rc) else np.nan,
        "ret_5d": ret(5), "ret_20d": ret(20), "ret_60d": ret(60), "ret_250d": ret(250),
        "vol_60d_ann": vol60,
        "amount_20d": amt20,
        "volume_ratio_5v20": amt5 / amt20 if amt20 > 0 else np.nan,
        "pos_in_60d_range": (last - lo60) / (hi60 - lo60) if hi60 > lo60 else np.nan,
        "drawdown_from_60d_high": last / hi60 - 1 if hi60 > 0 else np.nan,
        "rsi_14": rsi,
        "factor_rank": factor_rank,
    }
    for n, v in ma.items():
        f[f"px_vs_ma{n}"] = last / v - 1 if v > 0 else np.nan

    # 两融（已在 build_margin_panel 里滞后过，这里直接取）
    for key, label in [("rz_balance", "rz"), ("rq_volume", "rq")]:
        s = D.get(key)
        if s is None or sym not in s.columns:
            continue
        ss = s.loc[:date, sym].dropna()
        if len(ss) > 21:
            f[f"{label}_chg_20d"] = float(ss.iloc[-1] / ss.iloc[-21] - 1) \
                if ss.iloc[-21] else np.nan
            f[f"{label}_latest"] = float(ss.iloc[-1])

    # 分析师预期修正 —— 价值陷阱排查的核心依据。
    # 评级/EPS 下修是「基本面真的在恶化」的硬证据，
    # 能把「情绪超跌」和「价值陷阱」区分开。
    for key, label in [("upgrade_net", "analyst_upgrade_net"),
                       ("eps_revision", "analyst_eps_revision"),
                       ("coverage", "analyst_coverage")]:
        panel_df = D.get(key)
        if panel_df is None or sym not in panel_df.columns:
            continue
        ss = panel_df.loc[:date, sym].dropna()
        if len(ss):
            f[label] = float(ss.iloc[-1])
            if len(ss) > 60:
                f[f"{label}_chg60"] = float(ss.iloc[-1] - ss.iloc[-61])

    if meta is not None and not meta.empty:
        row = meta[meta["symbol"] == sym]
        if len(row):
            f["industry"] = str(row.iloc[0].get("industry", "") or "")
            f["mcap"] = float(row.iloc[0].get("mcap", np.nan))
    return f


def facts_to_text(f: dict) -> str:
    """把事实字典渲染成 LLM 可读的中文块。数值全部保留，不做主观描述。"""
    if not f:
        return ""
    L = [f"### {f['symbol']} {f.get('name','')}"]
    if f.get("industry"):
        L.append(f"行业: {f['industry']}"
                 + (f"  总市值: {f['mcap']/1e8:.0f}亿" if np.isfinite(f.get("mcap", np.nan)) else ""))
    L.append(f"现价: {_safe(f.get('price'))}")
    L.append("涨跌幅: " + "  ".join(
        f"{k}={_safe(f.get(f'ret_{k}'), 1, True)}" for k in ("5d", "20d", "60d", "250d")))
    L.append(f"年化波动率(60日): {_safe(f.get('vol_60d_ann'), 1, True)}"
             f"   RSI(14): {_safe(f.get('rsi_14'), 0)}")
    L.append("均线位置: " + "  ".join(
        f"距MA{n}={_safe(f.get(f'px_vs_ma{n}'), 1, True)}" for n in (5, 20, 60, 120)
        if f'px_vs_ma{n}' in f))
    L.append(f"60日区间位置: {_safe(f.get('pos_in_60d_range'), 2)} "
             f"(0=最低 1=最高)   距60日高点: {_safe(f.get('drawdown_from_60d_high'),1,True)}")
    L.append(f"20日均成交额: {f.get('amount_20d', 0)/1e8:.2f}亿   "
             f"近5日量比: {_safe(f.get('volume_ratio_5v20'))}")
    if "rz_chg_20d" in f:
        L.append(f"融资余额20日变化: {_safe(f.get('rz_chg_20d'), 1, True)}"
                 f"   融券余量20日变化: {_safe(f.get('rq_chg_20d'), 1, True)}")
    if "analyst_coverage" in f or "analyst_upgrade_net" in f:
        parts = []
        if "analyst_coverage" in f:
            parts.append(f"近60日研报数 {f['analyst_coverage']:.0f}")
            if "analyst_coverage_chg60" in f:
                parts.append(f"较前60日 {f['analyst_coverage_chg60']:+.0f}")
        if "analyst_upgrade_net" in f:
            parts.append(f"评级净上调 {f['analyst_upgrade_net']:+.0f}")
        if "analyst_eps_revision" in f and np.isfinite(f.get("analyst_eps_revision", np.nan)):
            parts.append(f"EPS预测修正 {f['analyst_eps_revision']:+.1%}")
        L.append("分析师覆盖: " + "  ".join(parts))
    if f.get("factor_rank") is not None:
        L.append(f"量化因子排名: 候选池第 {int(f['factor_rank'])} 名")
    return "\n".join(L)


def market_context(idx: pd.DataFrame, date: pd.Timestamp) -> str:
    """大盘环境 —— A股个股 beta 高，大盘状态必须进 prompt。"""
    c = idx["close"].loc[:date].dropna()
    if len(c) < 60:
        return ""
    ma60 = float(c.iloc[-60:].mean())
    r20 = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else np.nan
    r60 = float(c.iloc[-1] / c.iloc[-61] - 1) if len(c) > 61 else np.nan
    vol = float(c.pct_change().iloc[-20:].std() * np.sqrt(244))
    trend = "多头（收盘价在60日线上方）" if c.iloc[-1] > ma60 else "空头（收盘价在60日线下方）"
    return (f"【大盘环境 沪深300 @ {pd.Timestamp(date).date()}】\n"
            f"点位 {c.iloc[-1]:.0f}，{trend}\n"
            f"近20日 {r20:+.2%}，近60日 {r60:+.2%}，年化波动率 {vol:.1%}")
