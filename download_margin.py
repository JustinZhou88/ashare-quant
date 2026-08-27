"""全市场两融数据下载（多线程，可断点续传）。

    python download_margin.py --workers 6

和 download_bulk.py 同样的模式。串行下载 2800 只要 11 小时，
6 线程约 2 小时。已缓存的自动跳过，中断后重跑即可。

约 1/3 的 A股不是两融标的，这些会返回空表 —— 属于正常，不是失败。
因子层对缺两融数据的股票会退回纯价格打分（见 scan_weekly.factor_score）。
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from aq.config import DATA_DIR
from aq.data.altdata import load_margin
from aq.data.universe import from_file

_lock = threading.Lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_full.txt")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    uni = args.universe if Path(args.universe).exists() else "universe.txt"
    syms = from_file(uni)
    alt = Path(DATA_DIR) / "alt"
    alt.mkdir(parents=True, exist_ok=True)
    todo = [s for s in syms if not (alt / f"margin_{s}.csv").exists()]

    print(f"股票池 {len(syms)} 只，已缓存 {len(syms)-len(todo)} 只，"
          f"待下载 {len(todo)} 只，{args.workers} 线程")
    if not todo:
        print("已全部就绪。")
        return
    print(f"预计约 {len(todo)*14/args.workers/60:.0f} 分钟\n")

    done = {"ok": 0, "empty": 0, "n": 0}
    t0 = time.time()

    def fetch(sym: str):
        try:
            d = load_margin(sym, start=args.start)
            ok = not d.empty
        except Exception:                                      # noqa: BLE001
            ok = False
        with _lock:
            done["n"] += 1
            done["ok" if ok else "empty"] += 1
            n = done["n"]
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / el if el else 0
                print(f"  {n}/{len(todo)}  有数据 {done['ok']}  "
                      f"非两融标的 {done['empty']}  "
                      f"剩余约 {(len(todo)-n)/rate/60 if rate else 0:.0f} 分钟",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(fetch, s) for s in todo]):
            pass

    have = len(list(alt.glob("margin_*.csv")))
    print(f"\n完成：本地共 {have} 只有两融缓存（用时 {(time.time()-t0)/60:.1f} 分钟）")


if __name__ == "__main__":
    main()
