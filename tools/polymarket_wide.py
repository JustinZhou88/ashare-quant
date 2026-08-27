#!/usr/bin/env python3
"""宽口径 Polymarket 赔率采集器 —— 只落盘，不参与任何决策。

定位说明（很重要，别搞混）：
`aq/data/sentiment.py` 里的 PolymarketRisk 是**熔断器** —— 它只认 6 个中国
尾部风险关键词，用 `score = -prob` 的约定去砍仓位。那个约定成立的前提正是
关键词的狭窄性：每一个都描述"对中国股市不利的事件"。

本脚本覆盖面宽得多（利率/大宗/地缘/贸易/美国政策…），**因此绝不能套用同一
个符号约定** —— "美联储降息 85%" 套进去会被当成极端利空，把仓位砍到 1/3。

所以它现在只做一件事：把赔率和周变化存成时间序列。等积累够样本
（作者验证台海用了 232 个交易日），再用 aq/validation 那套方法检验哪些
类别真的对 A 股有领先性，有证据了才谈接入。

零 LLM 调用，零外部依赖（只用 requests）。

用法:
    python tools/polymarket_wide.py                 # 采集当日快照
    python tools/polymarket_wide.py --dry-run       # 只打印不落盘
    python tools/polymarket_wide.py --top 30        # 每类只留前 30
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT_DIR = Path("sentiment_feed/polymarket_wide")
API = "https://gamma-api.polymarket.com/markets"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 分类关键词。命中多个类别时全部记录 —— 交叉类（比如"中国+关税"）本身就是信息。
# 分类关键词。用**词边界**匹配 —— 裸子串会把 "war" 匹配进 Warnock / Stewart，
# 把一堆 2028 年总统提名的彩票盘错判成地缘政治。命中多类时全部记录，
# 交叉类（比如"中国+关税"）本身就是信息。
CATEGORIES: dict[str, tuple[str, ...]] = {
    "rates_recession": ("fed", "federal reserve", "interest rate", "rate cut",
                        "rate hike", "recession", "inflation", "cpi", "jobs report",
                        "unemployment", "gdp", "soft landing", "yield curve", "fomc"),
    "commodities": ("oil", "crude", "opec", "gold", "copper", "natural gas",
                    "wheat", "gasoline", "brent", "wti"),
    "geopolitics": ("war", "invade", "invasion", "ceasefire", "nuclear", "missile",
                    "airstrike", "nato", "ukraine", "russia", "iran", "israel",
                    "north korea", "taiwan", "coup", "annex"),
    "trade_tech": ("tariff", "tariffs", "trade war", "export control", "sanction",
                   "sanctions", "chip", "chips", "semiconductor", "rare earth",
                   "tiktok", "huawei", "nvidia", "asml", "tsmc"),
    "us_politics": ("government shutdown", "shutdown", "impeach", "supreme court",
                    "debt ceiling", "midterm"),
    "china": ("china", "chinese", "xi jinping", "yuan", "renminbi", "pboc",
              "hong kong", "beijing"),
    "crypto_risk": ("bitcoin", "ethereum", "crypto"),
}

# 这些词一出现就基本是长尾彩票盘（2028 提名、名人梗），排除掉
NOISE = ("presidential nomination", "presidential election", "win the 2028",
         "next prime minister", "jesus christ", "nobel", "time person of the year",
         "super bowl", "world cup", "oscar")

_PAT = {c: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b")
        for c, kws in CATEGORIES.items()}
_NOISE = re.compile("|".join(re.escape(x) for x in NOISE))


def _prob(m: dict) -> float | None:
    """取第一个 outcome 的价格 = 该事件发生的隐含概率。"""
    p = m.get("outcomePrices")
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:                                      # noqa: BLE001
            return None
    if isinstance(p, list) and p:
        try:
            v = float(p[0])
            return v if 0.0 <= v <= 1.0 else None
        except Exception:                                      # noqa: BLE001
            return None
    return None


def _classify(question: str) -> list[str]:
    q = question.lower()
    if _NOISE.search(q):
        return []
    return [c for c, pat in _PAT.items() if pat.search(q)]


def fetch(pages: int = 4, page_size: int = 500, timeout: float = 30.0) -> list[dict]:
    """按成交量降序翻页。必须显式指定 order —— 默认排序会漏掉大成交量市场。"""
    out: list[dict] = []
    seen: set[str] = set()
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    for i in range(pages):
        r = s.get(API, timeout=timeout,
                  params={"active": "true", "closed": "false",
                          "limit": page_size, "offset": i * page_size,
                          "order": "volumeNum", "ascending": "false"})
        r.raise_for_status()
        batch = [x for x in r.json() if isinstance(x, dict)]
        if not batch:
            break
        for x in batch:
            k = str(x.get("slug") or x.get("id") or x.get("question") or "")
            if k and k not in seen:
                seen.add(k)
                out.append(x)
    return out


def _prev_snapshot() -> dict[str, float]:
    """上一份快照的 slug -> prob，用来算变化量。"""
    if not OUT_DIR.exists():
        return {}
    files = sorted(OUT_DIR.glob("20*.jsonl"))
    if not files:
        return {}
    prev: dict[str, float] = {}
    for line in files[-1].read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            if r.get("slug") and r.get("prob") is not None:
                prev[r["slug"]] = float(r["prob"])
        except Exception:                                      # noqa: BLE001
            continue
    return prev


def build(markets: list[dict], min_volume: float, top_per_cat: int) -> list[dict]:
    prev = _prev_snapshot()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    rows: list[dict] = []
    for m in markets:
        q = str(m.get("question") or "")
        cats = _classify(q)
        if not cats:
            continue
        vol = float(m.get("volumeNum") or 0)
        if vol < min_volume:
            continue
        p = _prob(m)
        if p is None:
            continue
        # 概率贴边的市场已无信息含量：0.1% 的"LeBron 当选总统"和 99% 的既定事实
        # 都不会再动，只会污染样本
        if p < 0.02 or p > 0.98:
            continue
        slug = str(m.get("slug") or "")
        rows.append({
            "date": stamp,
            "slug": slug,
            "question": q[:200],
            "categories": cats,
            "prob": round(p, 4),
            # 变化量比水平值更有信息含量：99% 的"特朗普仍是总统"永远是 99%，
            # 但从 12% 跳到 30% 的市场才说明发生了什么。
            "d_prob": (round(p - prev[slug], 4) if slug in prev else None),
            "volume": round(vol, 0),
            "volume_1wk": round(float(m.get("volume1wk") or 0), 0),
            "volume_24hr": round(float(m.get("volume24hr") or 0), 0),
            "spread": m.get("spread"),
            "end_date": m.get("endDateIso") or m.get("endDate"),
        })
    # 每个类别只留成交量前 N —— 控制文件大小，也过滤掉长尾噪音
    keep: dict[str, dict] = {}
    for c in CATEGORIES:
        sub = [r for r in rows if c in r["categories"]]
        # 按**近一周**成交量排序，而不是累计成交量 —— 累计量会把常年挂着的
        # 彩票盘顶到前面，近期成交量才反映当下的关注度
        sub.sort(key=lambda r: (r["volume_1wk"], r["volume"]), reverse=True)
        for r in sub[:top_per_cat]:
            keep[r["slug"]] = r
    return sorted(keep.values(), key=lambda r: r["volume_1wk"], reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=4, help="翻几页（每页 500）")
    ap.add_argument("--min-volume", type=float, default=1e5, help="成交量下限，低于此值无信息含量")
    ap.add_argument("--top", type=int, default=25, help="每个类别保留前 N 个")
    ap.add_argument("--dry-run", action="store_true", help="只打印不落盘")
    args = ap.parse_args()

    try:
        markets = fetch(pages=args.pages)
    except Exception as e:                                     # noqa: BLE001
        print(f"! 拉取失败: {type(e).__name__}: {str(e)[:120]}")
        raise SystemExit(1)
    print(f"拉取 {len(markets)} 个活跃市场")

    rows = build(markets, args.min_volume, args.top)
    print(f"命中分类 {len(rows)} 个（成交量 >= ${args.min_volume:,.0f}，每类前 {args.top}）\n")

    for c in CATEGORIES:
        sub = [r for r in rows if c in r["categories"]]
        if not sub:
            continue
        print(f"— {c} （{len(sub)} 个）")
        for r in sub[:6]:
            d = r["d_prob"]
            ds = f" Δ{d:+.1%}" if d is not None else ""
            print(f"    {r['prob']:5.1%}{ds:>9}  周量 ${r['volume_1wk']/1e6:5.2f}M  {r['question'][:70]}")
        print()

    if args.dry_run:
        print("(--dry-run，未落盘)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = OUT_DIR / f"{rows[0]['date'] if rows else datetime.now().strftime('%Y-%m-%d')}.jsonl"
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")
    print(f"已写入 {f}（{len(rows)} 条）")
    n = len(list(OUT_DIR.glob('20*.jsonl')))
    print(f"累计快照 {n} 份 —— 需要几个月的样本才谈得上做相关性检验。")


if __name__ == "__main__":
    main()
