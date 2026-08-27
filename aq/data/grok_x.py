"""通过 Grok 的 x_search 工具获取 X 上的 A股讨论热度。

比自己爬 X 好在三点：
  1. **不违反服务条款** —— xAI 官方接口，Grok 服务端自己去搜
  2. **不需要登录** —— 只要 API key，不碰你的浏览器和账号
  3. **不会被封号** —— 爬虫迟早会因为对方改版或风控失效

⚠️ 接口版本：旧的 Live Search（`search_parameters`）已于 2026-01-12 下线，
调用会返回 410。本模块用的是 Responses API 的 `x_search` 工具
（`POST https://api.x.ai/v1/responses`），服务端自主执行搜索循环并返回带引用的结果。

输出写进 sentiment_feed/x_influencer/，**和爬虫的格式完全一致** ——
两条路可以互换，也可以同时用来交叉印证。

⚠️ 和爬虫一样的定位：拿不到可靠历史，**无法回测**。
所以只进 LLM 上下文并被 journal 记账，不进因子打分。
默认按**反向拥挤指标**解释：集中提及往往意味着流动性出口正在打开。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

OUT_DIR = Path("sentiment_feed/x_influencer")
API = "https://api.x.ai/v1/responses"
CODE_RE = re.compile(r"(?<!\d)(6\d{5}|00\d{4}|30\d{4}|68[89]\d{3})(?!\d)")

_PROMPT = """搜索 X 上最近 {days} 天关于中国 A股个股的讨论，找出**被讨论最多的具体股票**。

要求：
1. 只统计有明确指向的个股（提到 6 位代码或公司简称），忽略只谈大盘、指数、政策的内容
2. 按被提及的帖子数量排序，给出前 {n} 只
3. 对每只标注：股票代码、名称、大致提及次数、整体情绪倾向（看多/看空/分歧）
4. 如果某只票的讨论明显集中在少数几个账号，请指出——这对判断是真热度还是单人刷屏很重要

**只输出 JSON，不要任何其他文字：**
{{"stocks":[{{"symbol":"600519","name":"贵州茅台","mentions":42,
  "sentiment":"看多","concentrated":false,"note":"讨论要点一句话"}}],
  "period":"最近{days}天","caveat":"数据可得性或覆盖度的限制说明"}}"""


def fetch(days: int = 7, top_n: int = 20, model: str = "grok-4.5",
          api_key: str | None = None, timeout: float = 180.0) -> dict:
    """调用 Grok 搜索 X 上的 A股讨论。返回解析后的 dict。"""
    key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        return {"error": "未设置 XAI_API_KEY。去 https://console.x.ai 申请后："
                         "export XAI_API_KEY=..."}

    today = datetime.now()
    payload = {
        "model": model,
        "input": [{"role": "user",
                   "content": _PROMPT.format(days=days, n=top_n)}],
        "tools": [{
            "type": "x_search",
            "from_date": (today - timedelta(days=days)).strftime("%Y-%m-%d"),
            "to_date": today.strftime("%Y-%m-%d"),
        }],
    }
    try:
        r = requests.post(API, json=payload, timeout=timeout,
                          headers={"Authorization": f"Bearer {key}",
                                   "Content-Type": "application/json"})
        if r.status_code == 410:
            return {"error": "410 —— 用到了已下线的旧 Live Search 接口，"
                             "本模块应走 /v1/responses 的 x_search 工具"}
        r.raise_for_status()
        js = r.json()
    except Exception as e:                                     # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    text = _extract_text(js)
    data = _parse_json(text)
    if data is None:
        return {"error": "模型输出无法解析为 JSON", "raw": text[:600]}
    data["_citations"] = js.get("citations") or []
    data["_model"] = model
    return data


def _extract_text(js: dict) -> str:
    """从 Responses API 的返回里取出正文。字段结构随版本有变化，逐层兜底。"""
    if isinstance(js.get("output_text"), str):
        return js["output_text"]
    parts = []
    for item in (js.get("output") or []):
        for c in (item.get("content") or []):
            t = c.get("text")
            if isinstance(t, str):
                parts.append(t)
    if parts:
        return "\n".join(parts)
    ch = js.get("choices") or []
    if ch:
        return str((ch[0].get("message") or {}).get("content", ""))
    return json.dumps(js, ensure_ascii=False)[:2000]


def _parse_json(text: str):
    t = (text or "").strip()
    if "```" in t:
        for seg in t.split("```"):
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            if seg.startswith("{"):
                try:
                    return json.loads(seg)
                except Exception:                              # noqa: BLE001
                    continue
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:                                      # noqa: BLE001
            return None
    return None


def to_feed(data: dict) -> Path | None:
    """写成与爬虫一致的 jsonl 格式，供 sentiment.FeedFileSource 读取。"""
    if not data or data.get("error") or not data.get("stocks"):
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    counts = [float(s.get("mentions") or 0) for s in data["stocks"]]
    mu = float(np.mean(counts)) if counts else 0.0
    sd = float(np.std(counts)) if counts else 0.0
    for s in data["stocks"]:
        sym = str(s.get("symbol", "")).zfill(6)
        if not CODE_RE.fullmatch(sym):
            continue
        n = float(s.get("mentions") or 0)
        # 热度 z 分：横向比绝对提及数没意义，比的是「在这批里有多热」
        z = float(np.tanh((n - mu) / sd)) if sd > 0 else 0.0
        rows.append({"date": today, "symbol": sym, "count": int(n), "score": z,
                     "note": f"X提及{int(n)}次 {s.get('sentiment','')}"
                             f"{' 集中于少数账号' if s.get('concentrated') else ''}"
                             f" | {str(s.get('note',''))[:50]}"})
    if not rows:
        return None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{today}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                   encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="用 Grok 搜 X 上的 A股讨论热度")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--model", default="grok-4.5")
    a = ap.parse_args()

    d = fetch(days=a.days, top_n=a.top, model=a.model)
    if d.get("error"):
        print("失败:", d["error"])
        if d.get("raw"):
            print("原始输出:", d["raw"][:400])
        raise SystemExit(1)
    for s in d.get("stocks", [])[:15]:
        print(f"  {s.get('symbol','?'):8s} {str(s.get('name','')):10s} "
              f"提及{s.get('mentions','?'):>5} 次  {s.get('sentiment','')}"
              f"{'  [集中于少数账号]' if s.get('concentrated') else ''}")
    if d.get("caveat"):
        print(f"\n模型自述的局限: {d['caveat']}")
    p = to_feed(d)
    print(f"\n已写入 {p}" if p else "\n没有可用数据，未写入")
    print(f"引用来源 {len(d.get('_citations') or [])} 条")
