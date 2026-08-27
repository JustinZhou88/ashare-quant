"""东方财富股吧 —— A股散户情绪的最佳可得来源。

为什么用它而不是 X：
  - **X 上的中文 A股社区很小**。中文财经推特主体是币圈、美股、宏观时政，
    A股实盘讨论极少。抓 X 时间线还要猜哪条在说哪只票。
  - 股吧是**按个股聚合**的：一只票一个页面，帖子数/点击/评论直接对应关注度。
  - 公开页面，**无需登录、无封号风险**，且每次扫描只抓 20 只候选，成本极低。

指标含义：
  post_count   最近 N 天发帖数        —— 关注度绝对水平
  click_sum    帖子点击总量           —— 阅读热度
  comment_sum  评论总数               —— 参与深度
  heat_z       相对该股自身历史的 z 分  —— **这个才是信号**

为什么看 z 分而不是绝对值：茅台的股吧天然比小盘股热闹十倍，
横向比绝对值只会选出大盘股。真正有信息的是「**这只票现在比它平时热多少**」。

⚠️ 局限：股吧只能拿到当前页面，**没有历史**，所以无法回测。
和 X / MediaCrawler 一样，它只进 LLM 上下文并被 journal 记账，
不进因子打分。等积累 3 个月再用增量 alpha 回归判断它有没有用。

⚠️ 默认解释是**反向拥挤指标**：散户热度飙升往往意味着流动性出口正在打开。
是正是负最终由记账数据决定，不由这个假设决定。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HIST = Path("sentiment_feed/guba/history.csv")
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_CLICK = re.compile(r'"post_click_count":(\d+)')
_COMMENT = re.compile(r'"post_comment_count":(\d+)')
_PUBTIME = re.compile(r'"post_publish_time":"([^"]{10})')


def fetch_one(symbol: str, session: requests.Session | None = None,
              days: int = 7) -> dict:
    """抓单只股票的股吧热度快照。失败返回空 dict，不抛异常。"""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", _UA)
    try:
        r = s.get(f"https://guba.eastmoney.com/list,{symbol}.html", timeout=20)
        if r.status_code != 200:
            return {}
        t = r.text
    except Exception:                                          # noqa: BLE001
        return {}

    clicks = [int(x) for x in _CLICK.findall(t)]
    comments = [int(x) for x in _COMMENT.findall(t)]
    times = _PUBTIME.findall(t)
    if not clicks:
        return {}

    # 只统计最近 days 天的帖子（股吧首页混有置顶老帖）
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    recent = sum(1 for x in times if x[:10] >= cutoff) if times else len(clicks)

    return {
        "symbol": symbol,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "post_count": int(recent),
        "click_sum": int(np.sum(clicks)),
        "click_max": int(np.max(clicks)),
        "comment_sum": int(np.sum(comments)) if comments else 0,
        "n_posts_page": len(clicks),
    }


def fetch_many(symbols: list[str], days: int = 7, pause: float = 0.4,
               verbose: bool = False) -> pd.DataFrame:
    """抓一批股票（扫描时只抓候选池，通常 20 只）。"""
    s = requests.Session()
    s.headers.update({"User-Agent": _UA})
    rows = []
    for i, sym in enumerate(symbols, 1):
        d = fetch_one(sym, s, days=days)
        if d:
            rows.append(d)
        if verbose and i % 10 == 0:
            print(f"    股吧 {i}/{len(symbols)}", flush=True)
        time.sleep(pause)
    return pd.DataFrame(rows)


def append_history(df: pd.DataFrame) -> pd.DataFrame:
    """把今天的快照追加进历史 —— 这是日后能算 z 分和做验证的前提。

    股吧没有历史接口，所以**必须从今天开始自己积累**。
    每周扫描一次，一年就有 50 个观测点，足以算出每只票的热度基线。
    """
    HIST.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return pd.read_csv(HIST, dtype={"symbol": str}) if HIST.exists() \
            else pd.DataFrame()
    old = pd.read_csv(HIST, dtype={"symbol": str}) if HIST.exists() else pd.DataFrame()
    if not old.empty:
        key = set(zip(df["date"], df["symbol"]))
        old = old[~old.apply(lambda r: (r["date"], r["symbol"]) in key, axis=1)]
    out = pd.concat([old, df], ignore_index=True).sort_values(["date", "symbol"])
    out.to_csv(HIST, index=False)
    return out


def heat_scores(today: pd.DataFrame, hist: pd.DataFrame,
                min_obs: int = 4) -> dict[str, float]:
    """相对该股自身历史的热度 z 分。

    横向比绝对值没意义（茅台天生比小盘股热闹），
    真正有信息的是「这只票现在比它平时热多少」。
    历史观测不足 min_obs 次的返回 0（不是 NaN，避免下游误判为异常）。
    """
    if today.empty:
        return {}
    out: dict[str, float] = {}
    for _, r in today.iterrows():
        sym = str(r["symbol"])
        cur = float(r["click_sum"])
        h = hist[(hist["symbol"] == sym) & (hist["date"] < r["date"])] \
            if not hist.empty else pd.DataFrame()
        if len(h) < min_obs:
            out[sym] = 0.0
            continue
        base = h["click_sum"].astype(float)
        sd = base.std(ddof=0)
        out[sym] = float(np.tanh((cur - base.mean()) / sd)) if sd > 0 else 0.0
    return out


def to_readings(symbols: list[str], days: int = 7, verbose: bool = False) -> list:
    """抓取 + 记录历史 + 算 z 分，返回 sentiment.Reading 列表。"""
    from .sentiment import Reading
    df = fetch_many(symbols, days=days, verbose=verbose)
    if df.empty:
        return []
    hist = append_history(df)
    z = heat_scores(df, hist)
    today = datetime.now().strftime("%Y-%m-%d")
    out = []
    for _, r in df.iterrows():
        sym = str(r["symbol"])
        n_obs = int((hist["symbol"] == sym).sum()) if not hist.empty else 0
        note = (f"近{days}日发帖{r['post_count']}  点击{r['click_sum']:,}  "
                f"评论{r['comment_sum']}")
        if n_obs < 5:
            note += f"（历史仅{n_obs}次观测，z分暂不可靠）"
        out.append(Reading("guba", today, score=float(z.get(sym, 0.0)),
                           symbol=sym, note=note, raw=r.to_dict()))
    return out
