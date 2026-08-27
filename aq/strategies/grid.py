"""参数网格 —— 把 15 个策略族展开成几百个候选策略。

⚠️ 这里就是过拟合的源头。展开 400 个候选，等于做 400 次假设检验：
即使全是随机噪声，也必然有几个"看起来夏普 2"。
所以本项目**禁止**直接取排行榜第一名 —— 必须过 validation/ 那一关。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .library import FAMILIES


@dataclass(frozen=True)
class Strategy:
    family: str
    params: tuple[tuple[str, Any], ...]

    @property
    def kw(self) -> dict:
        return dict(self.params)

    @property
    def id(self) -> str:
        ps = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.family}({ps})"

    def signal(self, panel) -> pd.DataFrame:
        return FAMILIES[self.family](panel, **self.kw)


def _mk(family: str, **grid) -> list[Strategy]:
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        d = dict(zip(keys, combo))
        if not _valid(family, d):
            continue
        out.append(Strategy(family, tuple(sorted(d.items()))))
    return out


def _valid(family: str, p: dict) -> bool:
    """剔除无意义的参数组合（快线必须快于慢线等）。"""
    if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
        return False
    if family == "ma_stack" and not (p["n1"] < p["n2"] < p["n3"]):
        return False
    if "entry_n" in p and "exit_n" in p and p["exit_n"] > p["entry_n"]:
        return False
    if "lo" in p and "hi" in p and p["lo"] >= p["hi"]:
        return False
    if "k_lo" in p and "k_hi" in p and p["k_lo"] >= p["k_hi"]:
        return False
    if "rsi_lo" in p and "rsi_hi" in p and p["rsi_lo"] >= p["rsi_hi"]:
        return False
    return True


def build_grid() -> list[Strategy]:
    """全部候选策略。"""
    S: list[Strategy] = []

    S += _mk("ma_cross", fast=[3, 5, 8, 10, 20, 30], slow=[10, 20, 30, 60, 120, 250])
    S += _mk("ema_cross", fast=[3, 5, 8, 10, 20, 30], slow=[10, 20, 30, 60, 120, 250])
    S += _mk("ma_stack", n1=[3, 5, 10], n2=[8, 10, 20], n3=[20, 30, 60])
    S += _mk("macd_trend", fast=[8, 12, 16], slow=[17, 26, 35], sig=[5, 9])
    S += _mk("roc_momentum", n=[5, 10, 20, 40, 60, 120], thresh=[0.0, 0.02, 0.05, 0.10, 0.20])

    S += _mk("donchian", entry_n=[10, 20, 40, 55, 120], exit_n=[5, 10, 20, 40])
    S += _mk("turtle_lite", entry_n=[20, 55], exit_n=[10, 20],
             atr_n=[14, 20], atr_mult=[1.5, 2.0, 3.0])
    S += _mk("vol_breakout", n=[10, 20, 40], vol_mult=[1.5, 2.0, 3.0], exit_n=[5, 10, 20])
    S += _mk("boll_breakout", n=[10, 20, 30, 40, 60], k=[1.5, 2.0, 2.5])
    S += _mk("chandelier", entry_n=[10, 20, 40], atr_n=[14, 22], mult=[2.0, 2.5, 3.0, 4.0])

    S += _mk("rsi_reversion", n=[5, 9, 14, 21, 30], lo=[15, 20, 25, 30], hi=[50, 55, 70])
    S += _mk("boll_reversion", n=[10, 20, 40, 60], k=[1.5, 2.0, 2.5, 3.0])
    S += _mk("kdj_reversion", n=[9, 14, 21], k_lo=[15, 20, 25], k_hi=[70, 80, 90])
    S += _mk("trend_pullback", ma_n=[20, 60, 120, 250], rsi_n=[3, 5, 9],
             rsi_lo=[20, 30], rsi_hi=[50, 60, 70])
    S += _mk("dip_buy", ma_n=[10, 20, 40, 60], dip=[0.03, 0.05, 0.08, 0.12])

    return S


if __name__ == "__main__":
    g = build_grid()
    from collections import Counter
    print(f"候选策略总数: {len(g)}")
    for fam, c in Counter(s.family for s in g).most_common():
        print(f"  {fam:16s} {c:4d}")
