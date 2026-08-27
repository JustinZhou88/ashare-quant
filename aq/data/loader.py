"""A股行情数据层（多数据源 + 本地 CSV 缓存）。

数据源优先级：腾讯 → 东财。
东财单次能拿全历史但对连续请求会封 IP（SSL RECORD_LAYER_FAILURE），
腾讯每次上限 640 根需要翻页，但稳定得多。两个都实现，自动降级。

关键设计：**同时抓不复权和后复权两套价格**。
  - 后复权：算策略信号和收益。不像前复权那样每次分红改写历史，
    不同时间跑回测结果一致。
  - 不复权：判涨跌停。涨跌停按真实成交价算，用复权价判会在除权日误判。

不抓这两套，动量/突破类策略会"买在一字涨停板"，回测收益凭空翻倍。
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import DATA_DIR

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
# 腾讯已废弃 web.ifzq 前缀（返回 501 WAF 页），去掉 web. 后正常。
# 这是主数据源；它失效会导致 2800 只股票全部穿透到东财兜底，进而触发反爬限流。
_TX_URL = "https://ifzq.gtimg.cn/appstock/app/fqkline/get"
_EM_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_LIST = "https://push2.eastmoney.com/api/qt/clist/get"
_EM_LIST_HOSTS = ["push2", "82.push2", "push2delay"]
_UT = "fa5fd1943c7b386f172d6893dbfba10b"

_TX_MAX = 640          # 腾讯单次最多返回 640 根日线
_COLS = ["open", "high", "low", "close", "volume", "amount",
         "raw_open", "raw_high", "raw_low", "raw_close", "tradable", "name"]


# ---------------------------------------------------------------- 会话管理
def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Accept": "*/*", "Connection": "keep-alive"})
    return s


# requests.Session 不是线程安全的 —— 多线程下载时每个线程必须有自己的会话，
# 否则会出现连接复用冲突（表现为随机的 SSL / 连接重置错误）。
_local = threading.local()


def _get_session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = _new_session()
        _local.session = s
    return s


def _reset_session() -> None:
    """连续请求被掐 TLS 连接时重建会话，比死等重试有效。"""
    s = getattr(_local, "session", None)
    if s is not None:
        try:
            s.close()
        except Exception:                                      # noqa: BLE001
            pass
    _local.session = _new_session()


def _get_json(url: str, params: dict, retries: int = 3, timeout: float = 15.0) -> dict:
    last = None
    for i in range(retries):
        try:
            r = _get_session().get(url, params=params, timeout=timeout)
            r.raise_for_status()
            txt = r.text.strip()
            if txt.startswith("("):
                txt = txt[txt.index("(") + 1: txt.rindex(")")]
            return json.loads(txt)
        except Exception as e:                                 # noqa: BLE001
            last = e
            if isinstance(e, (requests.exceptions.SSLError,
                              requests.exceptions.ConnectionError)):
                _reset_session()
            time.sleep(0.5 * (i + 1) ** 1.5 + random.random() * 0.4)
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:110]}")


def _cache_dir() -> Path:
    p = Path(DATA_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- 代码转换
def tx_code(symbol: str) -> str:
    """6位**个股**代码 -> 腾讯代码（sh600519 / sz000001）。

    指数用 _tx_index_code —— 000001 既是上证指数(sh)又是平安银行(sz)，
    两者必须分开处理，不能共用一个映射。
    """
    s = str(symbol).zfill(6)
    return f"sh{s}" if s.startswith(("5", "6", "9")) else f"sz{s}"


def _tx_index_code(symbol: str) -> str:
    """指数专用：沪深300/上证系列在 sh，深证成指等在 sz。"""
    s = str(symbol).zfill(6)
    return f"sz{s}" if s.startswith("399") else f"sh{s}"


def em_secid(symbol: str) -> str:
    s = str(symbol).zfill(6)
    return f"1.{s}" if s.startswith(("5", "6", "9")) else f"0.{s}"


# ---------------------------------------------------------------- 腾讯源
def _tx_once(code: str, fq: str, beg: str, end: str) -> list:
    js = _get_json(_TX_URL, {"param": f"{code},day,{beg},{end},{_TX_MAX},{fq}"})
    data = js.get("data")
    if not isinstance(data, dict) or code not in data:
        return []
    node = data[code]
    rows = node.get(f"{fq}day") if fq else None
    return rows or node.get("day") or []


def _tx_kline(code: str, fq: str, start: str = "1990-01-01",
              end: str = "2050-01-01", pause: float = 0.35) -> pd.DataFrame:
    """向前翻页拿完整历史。腾讯每次返回 [beg,end] 区间内**最后** 640 根。

    单页失败不放弃整只股票：退避重试两次，仍失败就带着已有的部分历史返回。
    否则 10 次请求里只要有 1 次抖动，整只股票就白抓了。
    """
    chunks: list[list] = []
    cur_end = end
    for _ in range(30):
        rows = []
        for attempt in range(3):
            try:
                rows = _tx_once(code, fq, start, cur_end)
                break
            except Exception:                                  # noqa: BLE001
                time.sleep(1.0 + attempt * 1.5)
        if not rows:
            break
        chunks.append(rows)
        first = rows[0][0]
        if len(rows) < _TX_MAX or first <= start:
            break
        cur_end = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(pause)

    if not chunks:
        return pd.DataFrame()
    flat = [r[:6] for c in chunks for r in c]
    df = pd.DataFrame(flat, columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "close", "high", "low", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("date").set_index("date").sort_index()


# ---------------------------------------------------------------- 东财源
_EM_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57"


def _em_kline(symbol: str, fqt: int) -> pd.DataFrame:
    js = _get_json(_EM_KLINE, {
        "secid": em_secid(symbol), "ut": _UT, "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": _EM_FIELDS2, "klt": 101, "fqt": fqt,
        "beg": "19900101", "end": "20500101", "lmt": 100000,
    })
    data = js.get("data")
    if not data or not data.get("klines"):
        return pd.DataFrame()
    rows = [k.split(",")[:7] for k in data["klines"]]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low",
                                     "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.attrs["name"] = data.get("name", "")
    return df.drop_duplicates("date").set_index("date").sort_index()


# ---------------------------------------------------------------- 统一接口
def volume_unit_divisor(symbol: str) -> float:
    """腾讯行情的成交量单位在不同板块**不一致**，这里统一成「手」。

    实测 2805 只股票的中位换手率（按「手」口径计算）：
        主板60x 1.6%  创业板30x 4.2%  深主板00x 2.5%  —— 都是手，正常
        科创板688     346.0%                        —— 是「股」，差 100 倍

    不修这个，科创板的成交额会被放大 100 倍，流动性过滤形同虚设，
    回测和选股都会系统性高配科创板。
    """
    return 100.0 if str(symbol).zfill(6).startswith("688") else 1.0


def _assemble(adj: pd.DataFrame, raw: pd.DataFrame, name: str,
              amount: pd.Series | None = None, symbol: str = "") -> pd.DataFrame:
    """把后复权 + 不复权拼成标准表。volume 统一归一化为「手」。"""
    out = adj[["open", "high", "low", "close", "volume"]].copy()
    out["volume"] = out["volume"] / volume_unit_divisor(symbol)
    out = out.join(raw[["open", "high", "low", "close"]].add_prefix("raw_"), how="left")
    if amount is not None:
        out["amount"] = amount.reindex(out.index)
    else:
        # 腾讯不给成交额，用 成交量(手) × 100 × 不复权收盘 近似（误差 <3%）
        out["amount"] = out["volume"] * 100.0 * out["raw_close"]
    out["tradable"] = (out["volume"] > 0) & out["close"].notna() & out["raw_close"].notna()
    out["name"] = name
    return out[_COLS]


def load_stock(symbol: str, refresh: bool = False, start: str = "1990-01-01",
               source: str = "auto") -> pd.DataFrame:
    """单只股票日线。列见 _COLS。失败返回空 DataFrame。"""
    cache = _cache_dir() / f"{symbol}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, parse_dates=["date"], index_col="date")
        if "tradable" in df:
            df["tradable"] = df["tradable"].astype(bool)
        return df

    out = pd.DataFrame()
    if source in ("auto", "tencent"):
        code = tx_code(symbol)
        adj = _tx_kline(code, "hfq", start=start)
        time.sleep(0.3)
        raw = _tx_kline(code, "", start=start)
        if not adj.empty and not raw.empty:
            out = _assemble(adj, raw, name=_name_of(symbol), symbol=symbol)

    if out.empty and source in ("auto", "eastmoney"):
        adj = _em_kline(symbol, fqt=2)
        time.sleep(0.3)
        raw = _em_kline(symbol, fqt=0)
        if not adj.empty and not raw.empty:
            # 东财的 volume 单位是「手」且自带成交额，不需要归一化
            out = _assemble(adj, raw, name=adj.attrs.get("name", ""),
                            amount=adj.get("amount"), symbol="")

    if out.empty:
        return out
    out.to_csv(cache, index_label="date")
    return out


_NAME_CACHE: dict[str, str] = {}


def _name_of(symbol: str) -> str:
    """从已缓存的股票列表里查简称（用于识别 ST，决定涨跌停幅度）。"""
    global _NAME_CACHE
    if not _NAME_CACHE:
        f = _cache_dir() / "_symbols.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"symbol": str})
            _NAME_CACHE = dict(zip(d["symbol"], d["name"]))
    return _NAME_CACHE.get(str(symbol).zfill(6), "")


def load_index(symbol: str = "000300", refresh: bool = False,
               start: str = "1990-01-01") -> pd.DataFrame:
    """指数日线（基准 + 市场环境标签）。指数无复权概念。"""
    cache = _cache_dir() / f"idx_{symbol}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["date"], index_col="date")
    df = _tx_kline(_tx_index_code(symbol), "", start=start)
    if df.empty:
        df = _em_kline(symbol, fqt=0)
    if df.empty:
        return df
    df = df[["open", "high", "low", "close", "volume"]]
    df.to_csv(cache, index_label="date")
    return df


def load_many(symbols: list[str], refresh: bool = False, verbose: bool = True,
              pause: float = 0.25, start: str = "1990-01-01") -> dict[str, pd.DataFrame]:
    """批量加载，自动缓存。已缓存的直接读盘，不联网。

    失败的代码重跑一次通常能补齐（成功的会跳过）。
    """
    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for i, s in enumerate(symbols, 1):
        cached = (_cache_dir() / f"{s}.csv").exists() and not refresh
        try:
            df = load_stock(s, refresh=refresh, start=start)
            if not df.empty:
                out[s] = df
            else:
                failed.append(s)
        except Exception as e:                                 # noqa: BLE001
            failed.append(s)
            if verbose:
                print(f"  ! {s} {e}", flush=True)
        if verbose and (i % 10 == 0 or i == len(symbols)):
            print(f"  进度 {i}/{len(symbols)}  成功 {len(out)}  失败 {len(failed)}",
                  flush=True)
        if not cached:
            time.sleep(pause)
    if failed and verbose:
        print(f"  未取到 {len(failed)} 只: {failed[:10]}{' ...' if len(failed) > 10 else ''}")
    return out


# ---------------------------------------------------------------- 股票列表
def fetch_all_symbols(refresh: bool = False, max_pages: int = 15) -> pd.DataFrame:
    """A股快照（按总市值降序）：代码/名称/总市值/流通市值/成交额。

    max_pages=15 → 市值前 3000 只。剩下的微盘股过不了流动性过滤，抓了也白抓。

    注意：这是**今天**的列表，不含已退市股票 —— 幸存者偏差的来源，
    见 universe.py 的说明。
    """
    cache = _cache_dir() / "_symbols.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, dtype={"symbol": str})

    frames = []
    fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"     # 深主板/创业板/沪主板/科创板
    for pn in range(1, max_pages + 1):
        params = {"pn": pn, "pz": 200, "po": 1, "np": 1, "ut": _UT,
                  "fltt": 2, "invt": 2, "fid": "f20", "fs": fs,
                  "fields": "f12,f14,f20,f21,f6,f100"}
        js = None
        for host in _EM_LIST_HOSTS:
            try:
                js = _get_json(f"https://{host}.eastmoney.com/api/qt/clist/get",
                               params, retries=2, timeout=8.0)
                break
            except Exception:                                  # noqa: BLE001
                time.sleep(0.5)
        if js is None:
            print(f"  ! 第 {pn} 页失败，用已取到的 {sum(len(f) for f in frames)} 只继续")
            break
        diff = (js.get("data") or {}).get("diff") or []
        if not diff:
            break
        frames.append(pd.DataFrame(diff))
        time.sleep(0.35)

    if not frames:
        raise RuntimeError("拉取股票列表失败")
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "f12": "symbol", "f14": "name", "f20": "mcap", "f21": "float_mcap",
        "f6": "amount", "f100": "industry"})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    for c in ("mcap", "float_mcap", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "industry" in df:
        # f100 是东财所属行业。没有它的话 rules.yaml 的 exclude_industries
        # 会静默失效 —— 代码去查 meta["industry"]，列不存在就整条规则跳过。
        df["industry"] = df["industry"].astype(str).replace(
            {"-": "", "nan": "", "None": ""})
    df = df.dropna(subset=["mcap"]).drop_duplicates("symbol").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------- 当日补齐
_TX_RT = "https://qt.gtimg.cn/q="


def fetch_today_bars(symbols: list[str], batch: int = 60,
                     pause: float = 0.25) -> pd.DataFrame:
    """收盘后用**实时行情**补出今天的日线。

    为什么需要：腾讯的历史日线接口跑批比实时行情晚很多（实测收盘一小时后
    仍只给到昨天），但实时接口在 15:00 后返回的就是完整的今日 OHLCV ——
    现价即收盘价。不补这一根，周五收盘后就没法当晚扫描，得等到第二天。

    ⚠️ 只在 15:00 之后调用才有意义，盘中拿到的是未完成的 bar。
    ⚠️ 返回的是**不复权**价。后复权价由调用方用昨日复权系数外推
       （除权日会有偏差，等厂商批处理出来后会被覆盖修正）。
    """
    out = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        q = ",".join(tx_code(s) for s in chunk)
        try:
            r = _get_session().get(_TX_RT + q, timeout=20)
            r.encoding = "gbk"
            text = r.text
        except Exception:                                      # noqa: BLE001
            continue
        for line in text.splitlines():
            if "~" not in line:
                continue
            p = line.split("~")
            if len(p) < 40:
                continue
            code = p[2].strip()
            try:
                last, prev_close, op = float(p[3]), float(p[4]), float(p[5])
                hi, lo = float(p[33]), float(p[34])
                vol, amt_wan = float(p[36]), float(p[37])
                ts = p[30]
            except (ValueError, IndexError):
                continue
            # 停牌或未开盘：价格为 0
            if last <= 0 or op <= 0 or vol <= 0:
                continue
            sym6 = code.zfill(6)
            # 实时接口和历史接口一样：**科创板返回的是股不是手**。
            # 用成交额反推可验证：688390 量 7,449,089 × 64元 = 4.77亿，
            # 与接口给的成交额 4.82亿 吻合；若当成手则是 480亿，差 100 倍。
            vol = vol / volume_unit_divisor(sym6)
            out.append({"symbol": sym6,
                        "date": pd.Timestamp(ts[:8]) if len(ts) >= 8 else pd.NaT,
                        "raw_open": op, "raw_high": hi, "raw_low": lo,
                        "raw_close": last, "prev_close": prev_close,
                        "volume": vol, "amount": amt_wan * 1e4})
        time.sleep(pause)
    df = pd.DataFrame(out)
    return df.dropna(subset=["date"]) if not df.empty else df


def append_today(symbol: str, row: pd.Series) -> bool:
    """把补来的今日 bar 追加进缓存。已存在则覆盖。返回是否写入。"""
    cache = _cache_dir() / f"{symbol}.csv"
    if not cache.exists():
        return False
    try:
        d = pd.read_csv(cache, parse_dates=["date"], index_col="date")
    except Exception:                                          # noqa: BLE001
        return False
    if d.empty:
        return False
    day = pd.Timestamp(row["date"]).normalize()
    prev = d[d.index < day]
    if prev.empty:
        return False
    last = prev.iloc[-1]

    # 后复权价用**昨日复权系数**外推。除权日会有偏差，
    # 等厂商日线批处理出来后 update_data 会覆盖修正。
    try:
        factor = float(last["close"]) / float(last["raw_close"])
    except Exception:                                          # noqa: BLE001
        return False
    if not np.isfinite(factor) or factor <= 0:
        return False

    new = pd.DataFrame([{
        "open": row["raw_open"] * factor, "high": row["raw_high"] * factor,
        "low": row["raw_low"] * factor, "close": row["raw_close"] * factor,
        "volume": row["volume"], "amount": row["amount"],
        "raw_open": row["raw_open"], "raw_high": row["raw_high"],
        "raw_low": row["raw_low"], "raw_close": row["raw_close"],
        "tradable": True, "name": last.get("name", ""),
        # 标记为临时：后复权价是用昨日系数外推的，除权日会偏。
        # update_stock 看到这个标记会强制重取，用厂商正式数据覆盖。
        "provisional": True,
    }], index=[day])
    new.index.name = "date"
    if "provisional" not in d.columns:
        d = d.assign(provisional=False)
    merged = pd.concat([d[d.index != day], new]).sort_index()
    merged["provisional"] = merged["provisional"].fillna(False).astype(bool)
    merged.to_csv(cache, index_label="date")
    return True


def append_today_index(symbol: str = "000300", code: str | None = None) -> bool:
    """收盘后补当日指数 bar。

    和个股同理：腾讯历史日线跑批晚，但实时接口收盘后就是完整的今日 OHLC。
    指数没有复权问题，直接写即可。
    """
    cache = _cache_dir() / f"idx_{symbol}.csv"
    if not cache.exists():
        return False
    try:
        d = pd.read_csv(cache, parse_dates=["date"], index_col="date")
    except Exception:                                          # noqa: BLE001
        return False
    if d.empty:
        return False

    c = code or _tx_index_code(symbol)
    try:
        r = _get_session().get(_TX_RT + c, timeout=20)
        r.encoding = "gbk"
        p = r.text.split("~")
        if len(p) < 40:
            return False
        last, op = float(p[3]), float(p[5])
        hi, lo, vol = float(p[33]), float(p[34]), float(p[36] or 0)
        ts = p[30]
    except Exception:                                          # noqa: BLE001
        return False
    if last <= 0 or op <= 0:
        return False

    day = pd.Timestamp(ts[:8]) if ts[:8].isdigit() else pd.Timestamp.today().normalize()
    day = day.normalize()
    new = pd.DataFrame([{"open": op, "high": hi, "low": lo,
                         "close": last, "volume": vol}], index=[day])
    new.index.name = "date"
    merged = pd.concat([d[d.index != day], new]).sort_index()
    merged.to_csv(cache, index_label="date")
    return True
