"""全市场扩池下载（多线程，可断点续传）。

    python download_bulk.py --min-amount 5e7 --workers 6

瓶颈是网络往返不是 CPU，所以用线程池。每个线程有独立的 requests.Session
（见 loader._get_session）—— Session 不是线程安全的，共用会随机报 SSL 错误。

断点续传：已缓存的股票直接跳过，中断后重跑即可，不会重复下载。

⚠️ 关于选池的偏差：这里用**今日成交额**过滤掉微盘股。
对「向前扫描选股」这个用途没问题；但用于回测时，
它会漏掉"当年活跃、如今萎缩"的股票（轻微幸存者偏差）。
回测里真正起作用的是 engine.panel.liquidity_mask 那个逐日的 point-in-time 过滤。
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from aq.config import BENCHMARK, DATA_DIR
from aq.data.loader import fetch_all_symbols, load_index, load_stock

_lock = threading.Lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--min-amount", type=float, default=5e7,
                    help="今日成交额下限，默认 5000 万（过滤微盘股）")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pages", type=int, default=30, help="股票列表拉多少页(200只/页)")
    ap.add_argument("--refresh-list", action="store_true")
    args = ap.parse_args()

    print("拉取全市场股票列表 ...")
    meta = fetch_all_symbols(refresh=args.refresh_list, max_pages=args.pages)
    print(f"  全市场 {len(meta)} 只")

    m = meta.copy()
    bad = m["name"].str.contains("ST|退|B股", case=False, na=False, regex=True)
    m = m[~bad & ~m["symbol"].str.startswith(("4", "8", "920"))]
    print(f"  剔除 ST/退市/北交所后 {len(m)} 只")
    m = m[m["amount"].fillna(0) >= args.min_amount]
    print(f"  今日成交额 ≥ {args.min_amount/1e4:.0f} 万: {len(m)} 只")
    syms = m.sort_values("mcap", ascending=False)["symbol"].tolist()
    if args.max_symbols:
        syms = syms[:args.max_symbols]

    cache = Path(DATA_DIR)
    todo = [s for s in syms if not (cache / f"{s}.csv").exists()]
    print(f"\n已缓存 {len(syms)-len(todo)} 只，待下载 {len(todo)} 只，"
          f"{args.workers} 线程并发")
    if todo:
        est = len(todo) * 16 / args.workers / 60
        print(f"预计约 {est:.0f} 分钟\n")

    done = {"ok": 0, "fail": 0, "n": 0}
    t0 = time.time()

    def fetch(sym: str):
        try:
            df = load_stock(sym, start=args.start)
            ok = not df.empty
        except Exception:                                      # noqa: BLE001
            ok = False
        with _lock:
            done["n"] += 1
            done["ok" if ok else "fail"] += 1
            n = done["n"]
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / el if el > 0 else 0
                eta = (len(todo) - n) / rate / 60 if rate > 0 else 0
                print(f"  {n}/{len(todo)}  成功 {done['ok']} 失败 {done['fail']}  "
                      f"{rate*60:.0f}只/分  剩余约 {eta:.0f} 分钟", flush=True)
        return ok

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch, s) for s in todo]
            for _ in as_completed(futs):
                pass

    have = [s for s in syms if (cache / f"{s}.csv").exists()]
    Path("universe_full.txt").write_text("\n".join(have), encoding="utf-8")
    print(f"\n完成：{len(have)} 只可用，写入 universe_full.txt "
          f"（用时 {(time.time()-t0)/60:.1f} 分钟）")

    load_index(BENCHMARK, start=args.start)
    print(f"基准 {BENCHMARK} 已就绪")


if __name__ == "__main__":
    main()
