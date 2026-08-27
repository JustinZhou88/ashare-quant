"""把 MediaCrawler 的输出转成本项目的情绪数据格式。

    # 1) 单独跑 MediaCrawler（需要你自己的登录态），例如抓微博关键词
    #    python main.py --platform wb --lt qrcode --type search
    # 2) 把它的输出目录喂给这个脚本
    python tools/mediacrawler_adapt.py --input ~/MediaCrawler/data/weibo --platform weibo

MediaCrawler 支持小红书/抖音/快手/B站/微博/贴吧/知乎，**不支持雪球和东财股吧** ——
而 A股散户主要聚集在后两个地方，这是它用于股票情绪的最大局限。
该项目也声明仅供学习研究、禁止商业用途。

输出：sentiment_feed/mediacrawler/YYYY-MM-DD.jsonl
格式：{"date","symbol","count","score","note"}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("sentiment_feed/mediacrawler")
# 不能用 \b：Python 的 \b 把中文也算词字符，"688386值得关注" 里
# "6" 和 "值" 之间没有词边界，会漏掉紧挨中文的股票代码。
# 改用数字前后瞻，既避开中文问题，又不会把长数字串的一部分误当代码。
CODE_RE = re.compile(r"(?<!\d)(6\d{5}|00\d{4}|30\d{4}|68[89]\d{3})(?!\d)")

# 极简中文情感词典。想要更准可以换成模型打分，但对"情绪强度"这个用途，
# 提及量(count)本身通常比情感极性更有信息量 —— 散户情绪主要体现在关注度上。
POS = ["涨停", "大涨", "利好", "突破", "牛", "抄底", "满仓", "加仓", "看好", "翻倍"]
NEG = ["跌停", "大跌", "利空", "破位", "熊", "套牢", "割肉", "减仓", "看空", "暴雷"]


def iter_texts(root: Path):
    """遍历 MediaCrawler 的输出（json/jsonl/csv 都试一遍）。"""
    for f in root.rglob("*"):
        if f.suffix not in (".json", ".jsonl", ".csv"):
            continue
        try:
            if f.suffix == ".jsonl":
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        yield json.loads(line)
            elif f.suffix == ".json":
                d = json.loads(f.read_text(encoding="utf-8"))
                yield from (d if isinstance(d, list) else [d])
            else:
                import pandas as pd
                for _, r in pd.read_csv(f).iterrows():
                    yield r.to_dict()
        except Exception:                                      # noqa: BLE001
            continue


def text_of(rec: dict) -> str:
    """MediaCrawler 各平台字段名不统一，把可能的正文字段都拼起来。"""
    keys = ("content", "desc", "title", "text", "note_desc", "content_text")
    return " ".join(str(rec.get(k, "")) for k in keys if rec.get(k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="MediaCrawler 输出目录")
    ap.add_argument("--platform", default="weibo")
    ap.add_argument("--min-count", type=int, default=2,
                    help="提及次数下限，过滤偶然提及")
    args = ap.parse_args()

    root = Path(args.input).expanduser()
    if not root.exists():
        print(f"找不到目录 {root}")
        return

    name_map = {}
    try:
        import pandas as pd
        f = Path("data_cache/_symbols.csv")
        if f.exists():
            d = pd.read_csv(f, dtype={"symbol": str})
            name_map = {str(r["name"]).strip(): str(r["symbol"]).zfill(6)
                        for _, r in d.iterrows()
                        if isinstance(r["name"], str) and len(str(r["name"])) >= 2}
    except Exception:                                          # noqa: BLE001
        pass

    cnt: Counter[str] = Counter()
    senti: dict[str, list[int]] = {}
    n = 0
    for rec in iter_texts(root):
        t = text_of(rec)
        if not t:
            continue
        n += 1
        hits = set(CODE_RE.findall(t))
        for name, code in name_map.items():
            if name in t:
                hits.add(code)
        if not hits:
            continue
        s = sum(1 for w in POS if w in t) - sum(1 for w in NEG if w in t)
        for c in hits:
            cnt[c] += 1
            senti.setdefault(c, []).append(s)

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for c, k in cnt.most_common():
        if k < args.min_count:
            continue
        ss = senti.get(c, [0])
        avg = sum(ss) / len(ss)
        rows.append({"date": today, "symbol": c, "count": k,
                     "score": max(-1.0, min(1.0, avg / 3.0)),
                     "note": f"{args.platform}提及{k}次 情感均值{avg:+.1f}"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{today}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                   encoding="utf-8")
    print(f"扫描 {n} 条内容，识别 {len(rows)} 只股票 -> {out}")
    for r in rows[:10]:
        print(f"  {r['symbol']}  提及{r['count']}次  score={r['score']:+.2f}")


if __name__ == "__main__":
    main()
