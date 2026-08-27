"""外部情绪 / 事件源接入层。

架构决定：**爬虫与分析解耦**。
爬虫（MediaCrawler、Playwright 抓 X）由你单独运行，把结果落到 `sentiment_feed/`；
本模块只负责**读取 + 归一化 + 记账**。

为什么这么设计：
  1. 爬虫需要你的登录态、会被封、会因为对方改版而随时挂掉。
     把它塞进扫描主流程，等于让最脆弱的环节决定整条流水线能不能跑。
  2. 这类数据**拿不到历史，无法回测**。它们唯一诚实的验证方式是
     从今天开始前瞻记账（见 aq/journal.py）。既然如此，就不该让它们
     进入决策路径，而应该先作为"实验变量"记录 6~12 个月。

所以：**外部情绪只写进 LLM 的上下文供参考，并被记账；不进入因子打分。**
等积累够样本，用 journal 里的增量 alpha 回归来决定要不要提拔它。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FEED_DIR = Path("sentiment_feed")
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0"


@dataclass
class Reading:
    source: str
    as_of: str
    score: float                  # 归一化到 [-1, 1]，正=乐观/看多
    symbol: str | None = None     # None 表示市场级读数
    note: str = ""
    raw: dict = field(default_factory=dict)


# ================================================================ Polymarket
class PolymarketRisk:
    """地缘尾部风险温度计。

    实测：Polymarket 活跃市场里**没有任何 A股/中国经济/央行的市场**，
    只有地缘政治（台海、领导人更替、台湾选举）。
    所以它只能做一件事 —— 当尾部风险概率异动时，调整**总仓位**。
    它不提供任何选股信息，别指望。
    """

    name = "polymarket_risk"
    URL = "https://gamma-api.polymarket.com/markets"
    KEYWORDS = ("taiwan", "china", "chinese", "xi jinping", "tariff", "rare earth")

    def fetch(self, as_of: str | None = None) -> list[Reading]:
        as_of = as_of or datetime.now().strftime("%Y-%m-%d")
        out: list[Reading] = []
        try:
            # 必须按成交量降序取 —— 默认排序会漏掉成交量最大的市场
            # （实测台海那个 $39M 的市场不在默认排序的前 500 里）
            markets: list[dict] = []
            seen: set[str] = set()
            for off in (0, 500, 1000):
                r = requests.get(self.URL, headers={"User-Agent": _UA}, timeout=25,
                                 params={"active": "true", "closed": "false",
                                         "limit": 500, "offset": off,
                                         "order": "volumeNum", "ascending": "false"})
                b = [x for x in r.json() if isinstance(x, dict)]
                if not b:
                    break
                for x in b:
                    k = str(x.get("id") or x.get("question") or "")
                    if k and k not in seen:
                        seen.add(k)
                        markets.append(x)
        except Exception as e:                                 # noqa: BLE001
            return [Reading(self.name, as_of, 0.0, note=f"拉取失败: {type(e).__name__}")]

        for m in markets:
            q = str(m.get("question") or "")
            if not any(k in q.lower() for k in self.KEYWORDS):
                continue
            vol = float(m.get("volumeNum") or 0)
            if vol < 1e5:                       # 成交量太小的市场没有信息含量
                continue
            prob = _first_price(m)
            if prob is None:
                continue
            # 尾部风险概率越高，score 越负（越该减仓）
            out.append(Reading(self.name, as_of, score=-float(prob),
                               note=q[:120], raw={"prob": prob, "volume": vol}))
        return out or [Reading(self.name, as_of, 0.0, note="无相关活跃市场")]


def _first_price(m: dict) -> float | None:
    p = m.get("outcomePrices")
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:                                      # noqa: BLE001
            return None
    if isinstance(p, list) and p:
        try:
            return float(p[0])
        except Exception:                                      # noqa: BLE001
            return None
    return None


# ================================================================ 文件适配器
class FeedFileSource:
    """通用文件适配器 —— 读取外部爬虫落盘的结果。

    约定格式（JSON Lines 或 CSV，放在 sentiment_feed/<source>/）：
        {"date": "2026-08-01", "symbol": "600519", "score": 0.6,
         "count": 128, "note": "微博提及量环比+80%"}

    symbol 可省略（市场级读数）。score 若缺失，会用 count 做截面 z-score 归一化。
    这样无论你用 MediaCrawler、Playwright 还是手工整理，只要吐成这个格式就能接进来。
    """

    def __init__(self, source: str) -> None:
        self.name = source
        self.dir = FEED_DIR / source

    def fetch(self, as_of: str | None = None) -> list[Reading]:
        if not self.dir.exists():
            return []
        frames = []
        for f in sorted(self.dir.glob("*.json")) + sorted(self.dir.glob("*.jsonl")):
            try:
                if f.suffix == ".jsonl":
                    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
                else:
                    rows = json.loads(f.read_text(encoding="utf-8"))
                    rows = rows if isinstance(rows, list) else [rows]
                frames.append(pd.DataFrame(rows))
            except Exception:                                  # noqa: BLE001
                continue
        for f in sorted(self.dir.glob("*.csv")):
            try:
                frames.append(pd.read_csv(f, dtype={"symbol": str}))
            except Exception:                                  # noqa: BLE001
                continue
        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True)
        if "date" not in df:
            return []
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if as_of:
            df = df[df["date"] <= pd.Timestamp(as_of)]        # 严禁使用未来数据
        if df.empty:
            return []
        latest = df["date"].max()
        df = df[df["date"] == latest]

        if "score" not in df and "count" in df:
            c = pd.to_numeric(df["count"], errors="coerce")
            sd = c.std(ddof=0)
            df["score"] = np.tanh((c - c.mean()) / sd) if sd and sd > 0 else 0.0

        out = []
        for _, r in df.iterrows():
            sym = str(r["symbol"]).zfill(6) if pd.notna(r.get("symbol")) else None
            out.append(Reading(self.name, str(latest.date()),
                               score=float(r.get("score", 0) or 0), symbol=sym,
                               note=str(r.get("note", ""))[:120],
                               raw={k: r[k] for k in df.columns if k not in ("date",)}))
        return out


# 具名适配器 —— 只是给上面这个通用适配器起了好认的名字
def mediacrawler_source() -> FeedFileSource:
    """MediaCrawler 情绪。

    使用方式：单独跑 MediaCrawler（需要你自己的登录态），把结果整理成
    上面的约定格式，落到 sentiment_feed/mediacrawler/。
    注意该项目声明仅供学习研究、禁止商业用途，且不覆盖雪球/东财股吧 ——
    A股散户主要聚集地它抓不到，微博的股票讨论质量相对有限。
    """
    return FeedFileSource("mediacrawler")


def x_influencer_source() -> FeedFileSource:
    """X 博主提及。

    使用方式：用 Playwright 单独抓取（脚本见 tools/x_scrape.py），落到
    sentiment_feed/x_influencer/。

    强烈建议**只把它当反向拥挤指标**：某只票被集中推荐往往意味着
    流动性出口已经打开，而不是机会开始。这个判断会写进 LLM 的提示词，
    并由 journal 前瞻验证 —— 到底是正是负，让数据说话。
    """
    return FeedFileSource("x_influencer")


# ================================================================ 汇总
def collect(as_of: str | None = None, enable: tuple[str, ...] = (),
            symbols: list[str] | None = None) -> list[Reading]:
    """按配置拉取全部启用的情绪源。任何一个失败都不影响其他源。"""
    out: list[Reading] = []
    srcs = {"polymarket": PolymarketRisk,
            "mediacrawler": mediacrawler_source,
            "x_influencer": x_influencer_source}
    # 股吧要传候选股票列表，签名和别的源不同，单独处理
    if "guba" in enable and symbols:
        try:
            from .guba import to_readings as guba_readings
            out += guba_readings(list(symbols))
        except Exception as e:                                 # noqa: BLE001
            out.append(Reading("guba", as_of or "", 0.0,
                               note=f"失败: {type(e).__name__}"))
    for key in enable:
        maker = srcs.get(key)
        if maker is None:
            continue
        try:
            src = maker() if key != "polymarket" else maker()
            out += src.fetch(as_of)
        except Exception as e:                                 # noqa: BLE001
            out.append(Reading(key, as_of or "", 0.0, note=f"失败: {type(e).__name__}"))
    return out


def geopolitical_gate(readings: list[Reading], base_positions: int,
                      spike_prob: float = 0.25, hard_prob: float = 0.45) -> tuple[int, str]:
    """地缘风险熔断：概率超阈值就砍总仓位。

    定位说明（很重要，别搞混）：
    实测 232 个交易日，台海概率变化与沪深300 的**同步**相关性是 -0.025，
    领先 1/5/10 日全部约等于 0，20 日 -0.119（窗口重叠，真实 t 约 -0.5）。
    **作为收益预测信号，它没有证据支持。**

    但这个检验对尾部风险不公平 —— 样本期内没有真正的事件发生，
    就像用没着火的日子去评估火警的价值。所以它在这里的定位是
    **熔断器**而不是信号：不预测涨跌，只在尾部概率抬升时降低暴露。
    代价是偶尔少赚，收益是极端情形下不被打穿。这个不对称是划算的。
    """
    probs = [(-r.score, r.note) for r in readings
             if r.source == "polymarket_risk" and r.score < 0]
    if not probs:
        return base_positions, ""
    p, note = max(probs)
    if p >= hard_prob:
        return max(1, base_positions // 3), f"地缘风险熔断（{note[:40]} = {p:.0%}）：仓位降至 1/3"
    if p >= spike_prob:
        return max(2, base_positions // 2), f"地缘风险预警（{note[:40]} = {p:.0%}）：仓位减半"
    return base_positions, ""


def to_text(readings: list[Reading]) -> str:
    """渲染成 LLM 上下文。明确标注这些是**未经验证**的实验性信息。"""
    if not readings:
        return ""
    L = ["【外部情绪 / 事件源（实验性，尚未通过统计验证，仅供参考，不可作为主要依据）】"]
    by_src: dict[str, list[Reading]] = {}
    for r in readings:
        by_src.setdefault(r.source, []).append(r)
    for src, rs in by_src.items():
        L.append(f"— {src} —")
        for r in sorted(rs, key=lambda x: -abs(x.score))[:8]:
            tag = f"[{r.symbol}] " if r.symbol else "[市场级] "
            L.append(f"  {tag}score={r.score:+.2f}  {r.note}")
    return "\n".join(L)
