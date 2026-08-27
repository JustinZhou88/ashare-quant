"""下载行情到本地缓存。首次运行必须先跑这个。

    python download.py --n 150 --mode stratified

数据存在 data_cache/*.csv，之后所有回测都从本地读，不再联网。
"""

from __future__ import annotations

import argparse
import time

from aq.config import BENCHMARK
from aq.data import universe as U
from aq.data.loader import load_index, load_many


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="股票池大小")
    ap.add_argument("--mode", default="stratified",
                    choices=["stratified", "top_mcap", "file"])
    ap.add_argument("--file", default="universe.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2015-01-01", help="起始日期（越早请求越多）")
    ap.add_argument("--refresh", action="store_true", help="强制重新下载")
    args = ap.parse_args()

    if args.mode == "stratified":
        syms = U.stratified_sample(args.n, seed=args.seed)
    elif args.mode == "top_mcap":
        syms = U.top_mcap(args.n)
    else:
        syms = U.from_file(args.file)

    print(f"股票池 {len(syms)} 只（{args.mode}），起始 {args.start}")
    t0 = time.time()
    data = load_many(syms, refresh=args.refresh, start=args.start)
    print(f"成功 {len(data)}/{len(syms)} 只，用时 {time.time() - t0:.0f}s")

    idx = load_index(BENCHMARK, refresh=args.refresh, start=args.start)
    print(f"基准 {BENCHMARK}: {len(idx)} 根日线")

    if data:
        lens = sorted((len(d), s) for s, d in data.items())
        print(f"最短历史: {lens[0][1]} {lens[0][0]} 根 / 最长: {lens[-1][1]} {lens[-1][0]} 根")
    with open("universe.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(data)))
    print("股票池已写入 universe.txt")


if __name__ == "__main__":
    main()
