"""截面因子策略 —— A股原生信号，不是技术指标。

和 library.py 的区别：
  - library.py 是**时序**策略：每只股票独立判断买/不买
  - 本模块是**截面**策略：每天给所有股票打分，只买分数最高的 K 只

截面策略更适合周频选股（用户选定的形态），也更符合 A股的现实：
你不是在问"茅台该不该买"，而是在问"这 74 只里哪 10 只最值得买"。

因子返回 (signal, score) 两个矩阵：
  signal = 是否进入候选（0/1），score = 打分（引擎按它挑满 K 个名额）

⚠️ 所有因子的输入都必须已经做过 point-in-time 对齐和滞后
（见 altdata.build_margin_panel），本模块不再做二次滞后。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 工具
def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排名（0~1）。对量纲不敏感，是因子的标准处理。"""
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """截面标准化 + 去极值。"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-winsor, winsor)


def top_k_signal(score: pd.DataFrame, k: int, mask: pd.DataFrame | None = None
                 ) -> pd.DataFrame:
    """取每天分数最高的 k 只作为候选。"""
    s = score.where(mask) if mask is not None else score
    thresh = s.rank(axis=1, ascending=False, method="first")
    return (thresh <= k).astype(float)


def rebalance(sig: pd.DataFrame, freq_days: int) -> pd.DataFrame:
    """把日频信号降为每 freq_days 个交易日调一次仓（其余日期保持不变）。

    周频 = 5。降频能大幅减少换手，两融/财务这类慢变量本来也不需要日频调仓。
    """
    if freq_days <= 1:
        return sig
    keep = np.zeros(len(sig), dtype=bool)
    keep[::freq_days] = True
    # 注意：不能写 sig.where(pd.Series(keep, index=sig.index)) ——
    # DataFrame.where 传 Series 时会按**列**对齐，日期索引对不上列名（股票代码），
    # 结果全部变 NaN，信号被整片抹平。必须广播成二维。
    keep2d = pd.DataFrame(np.broadcast_to(keep[:, None], sig.shape),
                          index=sig.index, columns=sig.columns)
    return sig.where(keep2d).ffill().fillna(0.0)


# ---------------------------------------------------------------- 因子族
def margin_momentum(D, n: int = 20, k: int = 10, rebal: int = 5):
    """融资余额增速 —— 杠杆资金在加仓谁。

    逻辑：融资盘是 A股最活跃的边际增量资金，加仓方向有短期动量效应。
    """
    rz = D["rz_balance"]
    chg = rz / rz.shift(n) - 1.0
    score = cs_rank(chg)
    return top_k_signal(score, k, mask=rz.notna()).pipe(rebalance, rebal), score


def margin_reversal(D, n: int = 20, k: int = 10, rebal: int = 5):
    """融资余额降速 —— 反向版本。杠杆资金撤离后的超跌反弹。"""
    rz = D["rz_balance"]
    chg = rz / rz.shift(n) - 1.0
    score = cs_rank(-chg)
    return top_k_signal(score, k, mask=rz.notna()).pipe(rebalance, rebal), score


def margin_buy_intensity(D, n: int = 5, k: int = 10, rebal: int = 5):
    """融资买入额 / 成交额 —— 杠杆资金占比，衡量投机热度。"""
    ratio = (D["rz_buy"] / D["amount"].replace(0, np.nan)).rolling(n, min_periods=n).mean()
    score = cs_rank(ratio)
    return top_k_signal(score, k, mask=ratio.notna()).pipe(rebalance, rebal), score


def margin_crowding_reversal(D, n: int = 60, k: int = 10, rebal: int = 5):
    """融资拥挤度反转 —— 杠杆盘相对成交额过重时看空。

    拥挤度 = 融资余额 / 20日均成交额，含义是「融资盘要几天成交额才能出清」。
    比"占流通市值比"更好的地方：不需要历史流通市值（免费数据源拿不到可靠的
    历史股本），且直接对应强平时的流动性冲击。A股融资盘极度拥挤后往往是
    阶段性顶部（踩踏），所以取拥挤度**最低**的。
    """
    turnover = D["amount"].rolling(20, min_periods=10).mean().replace(0, np.nan)
    ratio = D["rz_balance"] / turnover
    z = (ratio - ratio.rolling(n, min_periods=n // 2).mean()) / \
        ratio.rolling(n, min_periods=n // 2).std().replace(0, np.nan)
    score = cs_rank(-z)
    return top_k_signal(score, k, mask=z.notna()).pipe(rebalance, rebal), score


def short_squeeze(D, n: int = 20, k: int = 10, rebal: int = 5):
    """融券余量骤增 —— 空头压力。取融券增速**最低**（空头最少）的。"""
    rq = D["rq_volume"]
    chg = rq / rq.shift(n).replace(0, np.nan) - 1.0
    score = cs_rank(-chg)
    return top_k_signal(score, k, mask=rq.notna()).pipe(rebalance, rebal), score


def holders_concentration(D, n: int = 0, k: int = 10, rebal: int = 10):
    """股东户数下降 = 筹码集中 —— 散户在离场，通常被视为主力吸筹。

    全样本初测（116 只池子，2017~2026）：年化 +18.18%，夏普 0.97，
    控制价格反转后仍有 +12.18% 年化 alpha，t=+2.67 **显著**。
    对比两融因子的 t=1.60（不显著），这是目前唯一通过增量检验的另类数据。

    ⚠️ 但只有 241 笔交易（季频数据换手低），且尚未通过滚动前推与多重检验校正。
    在通过之前**不要接进实盘打分**。参数 n 不使用，保留只为与其他因子签名一致。
    """
    hc = D.get("holders_chg")
    if hc is None:
        raise KeyError("需要 holders_chg 面板，见 altdata.load_holders + to_pit_panel")
    score = cs_rank(-hc)
    return top_k_signal(score, k, mask=hc.notna()).pipe(rebalance, rebal), score


def holders_dispersion(D, n: int = 0, k: int = 10, rebal: int = 10):
    """反向版本：股东户数上升最多（筹码分散）。作为方向性检验的对照。"""
    hc = D.get("holders_chg")
    if hc is None:
        raise KeyError("需要 holders_chg 面板")
    score = cs_rank(hc)
    return top_k_signal(score, k, mask=hc.notna()).pipe(rebalance, rebal), score


def vol_tilted_reversal(D, n: int = 20, k: int = 10, rebal: int = 5,
                        vol_weight: float = 0.5):
    """反转因子 + 波动率倾斜 —— 对应用户「越快越好、弹性越大越好」的偏好。

    实测（简化模拟，9238 笔）：候选池里波动率最高的 20% 组，
    10 日内摸到 +15% 的概率 15.58%，最低组只有 1.95%；净命中率（扣掉触及 -15%）
    +7.31% vs +1.84%，平均收益和胜率也都是最高组占优。

    ⚠️ 那个模拟**没走完整引擎**（未计涨停买不进和交易成本），
    而高波动股恰恰是涨停最频繁的地方 —— 所以真实收益会低于模拟值。
    本函数走正规引擎，结果以前推验证为准。
    """
    cl = D["close"]
    rev = cs_rank(-cl.pct_change(n))
    vol = cs_rank(cl.pct_change().rolling(60, min_periods=40).std())
    score = (1 - vol_weight) * rev + vol_weight * vol
    return top_k_signal(score, k, mask=cl.notna()).pipe(rebalance, rebal), score


def price_momentum_cs(D, n: int = 20, k: int = 10, rebal: int = 5):
    """纯价格截面动量 —— **对照组**。

    如果两融因子跑不赢这个，说明它没提供价格之外的信息。
    """
    score = cs_rank(D["close"].pct_change(n))
    return top_k_signal(score, k, mask=D["close"].notna()).pipe(rebalance, rebal), score


def price_reversal_cs(D, n: int = 20, k: int = 10, rebal: int = 5):
    """纯价格截面反转 —— A股历史上比动量更有效的对照组。"""
    score = cs_rank(-D["close"].pct_change(n))
    return top_k_signal(score, k, mask=D["close"].notna()).pipe(rebalance, rebal), score


def margin_plus_reversal(D, n: int = 20, m: int = 20, k: int = 10, rebal: int = 5):
    """融资加仓 + 价格超跌 的复合因子。

    找"杠杆资金在买、但价格还没涨"的股票 —— 资金面领先价格的假设。
    """
    rz = D["rz_balance"]
    f1 = cs_rank(rz / rz.shift(n) - 1.0)
    f2 = cs_rank(-D["close"].pct_change(m))
    score = (f1 + f2) / 2
    return top_k_signal(score, k, mask=rz.notna()).pipe(rebalance, rebal), score


FACTOR_FAMILIES: dict[str, Callable] = {
    "margin_momentum": margin_momentum,
    "margin_reversal": margin_reversal,
    "margin_buy_intensity": margin_buy_intensity,
    "margin_crowding_reversal": margin_crowding_reversal,
    "short_squeeze": short_squeeze,
    "margin_plus_reversal": margin_plus_reversal,
    "price_momentum_cs": price_momentum_cs,      # 对照组
    "price_reversal_cs": price_reversal_cs,      # 对照组
}


@dataclass(frozen=True)
class FactorStrategy:
    family: str
    params: tuple[tuple[str, object], ...]

    @property
    def kw(self) -> dict:
        return dict(self.params)

    @property
    def id(self) -> str:
        return f"{self.family}({','.join(f'{k}={v}' for k, v in self.params)})"

    def build(self, D) -> tuple[pd.DataFrame, pd.DataFrame]:
        return FACTOR_FAMILIES[self.family](D, **self.kw)


def build_factor_grid(k_values=(5, 10, 20), rebal_values=(5, 10, 20),
                      n_values=(5, 10, 20, 60)) -> list[FactorStrategy]:
    """因子 × 参数网格。

    刻意保持**比技术指标网格小得多** —— 因子数量少、经济含义清楚，
    搜索空间小意味着多重检验的惩罚也小，更容易通过 DSR 检验。
    这是设计上的取舍，不是偷懒。
    """
    import itertools
    out = []
    for fam in FACTOR_FAMILIES:
        grid = {"k": k_values, "rebal": rebal_values}
        if fam != "margin_crowding_reversal":
            grid["n"] = n_values
        else:
            grid["n"] = (60, 120)
        for combo in itertools.product(*grid.values()):
            p = dict(zip(grid.keys(), combo))
            out.append(FactorStrategy(fam, tuple(sorted(p.items()))))
    return out
