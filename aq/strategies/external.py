"""接入外部信号 —— 让别的系统当"选手"，本项目当"裁判"。

任何能输出「每天该持有哪些股票」的系统，都可以塞进同一套引擎和验证层：
LLM 多智能体（如 TradingAgents）、Pine Script 导出的信号、因子模型、
你自己的选股表格 —— 只要变成一张 0/1 矩阵。

CSV 格式：第一列是日期，其余每列一个 6 位股票代码，值为 0/1
（1 = 该日**收盘时**希望持有）。引擎会在次日开盘执行。

    date,600519,000651,300750
    2024-01-02,1,0,0
    2024-01-03,1,1,0

⚠️ 用 LLM 生成历史信号时必须知道的事
--------------------------------------------------
LLM 回测有一个技术指标策略没有的、**更严重的未来函数**：
未来已经在模型权重里了。

让模型基于 2020 年 3 月的新闻做决策，它早就知道疫情后市场 V 型反弹、
知道哪些公司后来暴雷、哪些赛道后来崩了。这不是参数过拟合，是数据泄漏，
而且 **walk-forward 修不了** —— 泄漏不在你的数据切分里，在模型的训练语料里。

唯一干净的做法：只在**模型知识截止日之后**的区间做前推验证。
样本必然很短（几个月到一两年），那就老实承认样本短，
而不是拿一段模型"背过答案"的历史去证明策略有效。

本模块的 `warn_knowledge_cutoff` 会在信号区间早于给定截止日时提示这一点。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def load_signal_csv(path: str | Path) -> pd.DataFrame:
    """读入外部信号矩阵，做基本合法性检查。"""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [str(c).strip().zfill(6) for c in df.columns]

    vals = df.to_numpy(dtype=float, na_value=0.0)
    bad = np.isfinite(vals) & (vals != 0) & (vals != 1)
    if bad.any():
        raise ValueError(f"{path}: 信号矩阵只能是 0/1，发现 {int(bad.sum())} 个其它取值")
    if df.index.duplicated().any():
        raise ValueError(f"{path}: 存在重复日期")
    return df.fillna(0.0)


def warn_knowledge_cutoff(signals: pd.DataFrame, cutoff: str) -> str | None:
    """信号区间早于模型知识截止日时给出警告文本，否则返回 None。"""
    if signals.empty:
        return None
    start = signals.index.min()
    cut = pd.Timestamp(cutoff)
    if start >= cut:
        return None
    n_before = int((signals.index < cut).sum())
    return (f"⚠️ 信号起始于 {start.date()}，早于知识截止日 {cut.date()}，"
            f"其中 {n_before} 个交易日（占 {n_before/len(signals):.0%}）落在模型"
            f"「已知答案」的区间内。这部分回测结果不可信 —— "
            f"请用 --start {cut.date()} 单独看截止日之后的表现。")


@dataclass(frozen=True)
class ExternalStrategy:
    """和 grid.Strategy 同接口，可以直接混进筛选流程一起被检验。"""
    name: str
    path: str
    family: str = "external"

    @property
    def params(self) -> tuple:
        return (("source", self.name),)

    @property
    def id(self) -> str:
        return f"external({self.name})"

    def signal(self, panel) -> pd.DataFrame:
        raw = load_signal_csv(self.path)
        # 对齐到面板：缺失的日期/股票按不持仓处理；日期用前值填充，
        # 这样"每周决策一次"的系统也能正确表达"这一周一直持有"
        out = raw.reindex(columns=panel.symbols).reindex(panel.dates, method="ffill")
        return out.fillna(0.0)


def discover(directory: str | Path) -> list[ExternalStrategy]:
    """把目录下所有 *.csv 当作外部策略。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    return [ExternalStrategy(name=p.stem, path=str(p))
            for p in sorted(d.glob("*.csv"))]
