"""股票池构建。

⚠️ 幸存者偏差（Survivorship Bias）说明 —— 这是 A股回测最容易被忽略的坑：

东财只能拉到**当前在市**的股票。用今天的名单回测过去 10 年，等于事先
知道了"这些公司活下来了、没退市、没被 ST"。这会系统性高估收益，
尤其对持股周期长的策略。

本模块的缓解措施：
  1. `stratified_sample` 按市值分层随机抽样，避免只挑今天的大白马；
  2. 回测时用 `point_in_time_filter` 按**当时可得**的滚动成交额筛选，
     不用今天的市值排名（那是明确的未来函数）；
  3. 残留偏差（已退市股票完全缺失）无法用免费数据源修复 —— 请把
     筛出来的收益率打个折看，不要当成可实现收益。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import fetch_all_symbols


def _exclude_junk(df: pd.DataFrame) -> pd.DataFrame:
    """剔除 ST、退市整理、B股、北交所（涨跌幅规则不同且流动性差）。"""
    bad = df["name"].str.contains("ST|退|B股", case=False, regex=True, na=False)
    is_bj = df["symbol"].str.startswith(("4", "8", "920"))
    return df[~bad & ~is_bj].copy()


def top_mcap(n: int = 300, exclude_star: bool = False) -> list[str]:
    """今日市值前 N。最快，但幸存者偏差最重 —— 只建议做冒烟测试。"""
    df = _exclude_junk(fetch_all_symbols())
    if exclude_star:
        df = df[~df["symbol"].str.startswith("688")]
    return df.nlargest(n, "mcap")["symbol"].tolist()


def stratified_sample(n: int = 300, seed: int = 42,
                      n_strata: int = 5, min_mcap: float = 3e9) -> list[str]:
    """按市值分层随机抽样，各层等量。比 top_mcap 更接近真实可交易全集。"""
    df = _exclude_junk(fetch_all_symbols())
    df = df[df["mcap"] >= min_mcap]
    if df.empty:
        raise RuntimeError("没有满足市值下限的股票")

    df = df.sort_values("mcap").reset_index(drop=True)
    bounds = np.linspace(0, len(df), n_strata + 1).astype(int)
    per = max(1, n // n_strata)
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        stratum = df.iloc[lo:hi]
        if stratum.empty:
            continue
        k = min(per, len(stratum))
        idx = rng.choice(len(stratum), size=k, replace=False)
        picked += stratum.iloc[idx]["symbol"].tolist()
    return sorted(picked)[:n]


def from_file(path: str) -> list[str]:
    """自定义股票池：每行一个 6 位代码（# 开头为注释）。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                out.append(line.zfill(6))
    return out


# 逐日的可交易域筛选（流动性 + 上市时长 + 停牌）见
# aq.engine.panel.liquidity_mask —— 那里是全向量化版本，语义相同但快得多。
