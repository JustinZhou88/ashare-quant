"""A股另类数据：融资融券、龙虎榜、财务、股东户数。

为什么不是北向资金：**2024-08-19 起沪深港通停止披露个股每日持股明细和实时数据，
改为季度披露**。历史数据（2017 ~ 2024.8）还能做研究，但作为实盘信号已经死了。
融资融券是它之后最好的日频、个股级资金面数据。

point-in-time 纪律（这是另类数据回测最容易出错的地方）：
  - 融资融券：交易所在**次一交易日**公布 T 日数据 → 默认 lag=1 个交易日
  - 龙虎榜：T 日盘后公布 → T 日收盘信号可用，lag=0
  - 财务/股东户数：用**公告日**对齐，不是报告期。用报告期 = 提前几个月
    知道财报，这是 A股回测最经典的未来函数。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from .loader import _get_json, _get_session, _reset_session

_EM_DC = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_MARGIN_COLS = {
    "DATE": "date", "SCODE": "symbol", "RZYE": "rz_balance",
    "RZMRE": "rz_buy", "RQYL": "rq_volume", "RQYE": "rq_balance",
    "RZRQYE": "rzrq_balance",
}


def _alt_dir() -> Path:
    p = Path(DATA_DIR) / "alt"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- 融资融券
def load_margin(symbol: str, start: str = "2015-01-01",
                refresh: bool = False, page_size: int = 500) -> pd.DataFrame:
    """单只股票的融资融券历史。

    列: rz_balance(融资余额) rz_buy(融资买入额) rq_volume(融券余量)
        rq_balance(融券余额) rzrq_balance(融资融券余额)
    非两融标的返回空表 —— 这本身就是信息（约 1/3 的 A股不是两融标的）。
    """
    cache = _alt_dir() / f"margin_{symbol}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["date"], index_col="date")

    rows: list[dict] = []
    for page in range(1, 30):
        params = {
            "reportName": "RPTA_WEB_RZRQ_GGMX", "columns": "ALL",
            "filter": f'(SCODE="{symbol}")', "pageNumber": page,
            "pageSize": page_size, "sortColumns": "DATE", "sortTypes": -1,
            "source": "WEB", "client": "WEB",
        }
        try:
            js = _get_json(_EM_DC, params, retries=3, timeout=20)
        except Exception:                                      # noqa: BLE001
            _reset_session()
            break
        data = (js.get("result") or {}).get("data") or []
        if not data:
            break
        rows += data
        if len(data) < page_size or str(data[-1].get("DATE", ""))[:10] < start:
            break
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    keep = {k: v for k, v in _MARGIN_COLS.items() if k in df.columns}
    df = df[list(keep)].rename(columns=keep)
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
    for c in df.columns:
        if c not in ("date", "symbol"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.drop(columns=["symbol"], errors="ignore")
            .drop_duplicates("date").set_index("date").sort_index())
    df = df[df.index >= start]
    df.to_csv(cache, index_label="date")
    return df


def load_margin_many(symbols: list[str], start: str = "2015-01-01",
                     refresh: bool = False, verbose: bool = True,
                     pause: float = 0.25,
                     cache_only: bool = False) -> dict[str, pd.DataFrame]:
    """批量加载两融数据。

    cache_only=True：**只读本地缓存，绝不联网**。
    每周扫描必须用这个 —— 全市场 2800 只逐个下载要 11 小时，会把扫描卡死。
    补数据请单独跑 `python download_margin.py`（多线程，约 2 小时）。
    """
    if cache_only:
        out = {}
        for s in symbols:
            f = _alt_dir() / f"margin_{s}.csv"
            if f.exists():
                try:
                    out[s] = pd.read_csv(f, parse_dates=["date"], index_col="date")
                except Exception:                              # noqa: BLE001
                    pass
        if verbose:
            print(f"  两融缓存命中 {len(out)}/{len(symbols)} 只")
        return out

    out: dict[str, pd.DataFrame] = {}
    n_empty = 0
    for i, s in enumerate(symbols, 1):
        cached = (_alt_dir() / f"margin_{s}.csv").exists() and not refresh
        try:
            d = load_margin(s, start=start, refresh=refresh)
            if not d.empty:
                out[s] = d
            else:
                n_empty += 1
        except Exception as e:                                 # noqa: BLE001
            n_empty += 1
            if verbose:
                print(f"  ! {s} {type(e).__name__}: {str(e)[:60]}", flush=True)
        if verbose and (i % 20 == 0 or i == len(symbols)):
            print(f"  两融 {i}/{len(symbols)}  有数据 {len(out)}  无数据 {n_empty}",
                  flush=True)
        if not cached:
            time.sleep(pause)
    return out


def build_margin_panel(margin: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                       symbols: list[str], lag: int = 1) -> dict[str, pd.DataFrame]:
    """把两融数据对齐到行情面板，并**按交易日滞后 lag 天**。

    lag=1 的理由：交易所次一交易日才公布 T 日两融明细。不滞后 = 未来函数。
    非两融标的整列为 NaN，因子层会把它们排除在候选之外。
    """
    fields = ["rz_balance", "rz_buy", "rq_volume", "rq_balance"]
    out: dict[str, pd.DataFrame] = {}
    for f in fields:
        cols = {}
        for s in symbols:
            d = margin.get(s)
            cols[s] = (d[f].reindex(dates).ffill() if d is not None and f in d
                       else pd.Series(np.nan, index=dates))
        out[f] = pd.DataFrame(cols, index=dates).shift(lag)
    return out


# ---------------------------------------------------------------- 龙虎榜
def load_lhb(start: str = "20150101", end: str = "20260801",
             refresh: bool = False) -> pd.DataFrame:
    """龙虎榜明细。T 日盘后公布，所以 T 日收盘信号可用（lag=0）。"""
    cache = _alt_dir() / f"lhb_{start}_{end}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["date"], dtype={"symbol": str})

    import akshare as ak
    frames = []
    # 按年切分，一次拉太长会超时
    for y in range(int(start[:4]), int(end[:4]) + 1):
        b = max(f"{y}0101", start)
        e = min(f"{y}1231", end)
        try:
            d = ak.stock_lhb_detail_em(start_date=b, end_date=e)
        except Exception as e_:                                # noqa: BLE001
            print(f"  ! 龙虎榜 {y} 失败: {str(e_)[:60]}")
            continue
        if d is not None and len(d):
            frames.append(d)
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ren = {"代码": "symbol", "名称": "name", "上榜日": "date",
           "龙虎榜净买额": "lhb_net", "龙虎榜买入额": "lhb_buy",
           "龙虎榜卖出额": "lhb_sell", "解读": "reason", "换手率": "turnover"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------- 财务
def load_earnings(periods: list[str], refresh: bool = False) -> pd.DataFrame:
    """业绩报表。**用「最新公告日期」作为可得日**，不是报告期。

    periods: 报告期列表，如 ['20240331','20240630',...]
    """
    cache = _alt_dir() / "earnings.csv"
    if cache.exists() and not refresh:
        d = pd.read_csv(cache, parse_dates=["ann_date"], dtype={"symbol": str})
        if set(periods) <= set(d["period"].astype(str)):
            return d

    import akshare as ak
    frames = []
    for p in periods:
        try:
            d = ak.stock_yjbb_em(date=p)
        except Exception as e:                                 # noqa: BLE001
            print(f"  ! 业绩报表 {p} 失败: {str(e)[:60]}")
            continue
        if d is None or not len(d):
            continue
        d = d.rename(columns={
            "股票代码": "symbol", "最新公告日期": "ann_date",
            "每股收益": "eps", "净利润-同比增长": "np_yoy",
            "营业总收入-同比增长": "rev_yoy", "净资产收益率": "roe",
            "销售毛利率": "gross_margin", "所处行业": "industry"})
        d["period"] = p
        frames.append(d[[c for c in ["symbol", "ann_date", "period", "eps", "np_yoy",
                                     "rev_yoy", "roe", "gross_margin", "industry"]
                         if c in d.columns]])
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df = df.dropna(subset=["ann_date"])
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------- 股东户数
def load_holders(periods: list[str], refresh: bool = False) -> pd.DataFrame:
    """股东户数 —— 散户情绪/筹码集中度的**可回测代理**。

    这是判断「要不要花两周搭爬虫」的关键实验：股东户数和微博情绪测的是
    同一件事（散户在进还是在出），但它有十年历史、有公告日、免费且稳定。
    如果它都测不出增量 alpha，爬来的社交媒体情绪大概率也测不出来。

    对齐用 `公告日期`，不是 `股东户数统计截止日` —— 两者常差 3~4 个月，
    用截止日 = 提前几个月知道筹码变化，是典型的未来函数。
    """
    cache = _alt_dir() / "holders.csv"
    if cache.exists() and not refresh:
        d = pd.read_csv(cache, parse_dates=["ann_date"], dtype={"symbol": str})
        if set(periods) <= set(d["period"].astype(str)):
            return d

    import akshare as ak
    frames = []
    for p in periods:
        try:
            d = ak.stock_zh_a_gdhs(symbol=p)
        except Exception as e:                                 # noqa: BLE001
            print(f"  ! 股东户数 {p} 失败: {str(e)[:60]}", flush=True)
            continue
        if d is None or not len(d):
            continue
        d = d.rename(columns={
            "代码": "symbol", "名称": "name", "公告日期": "ann_date",
            "股东户数-本次": "holders", "股东户数-上次": "holders_prev",
            "股东户数-增减比例": "holders_chg_pct",
            "户均持股市值": "avg_hold_value", "股东户数统计截止日-本次": "period_end"})
        d["period"] = p
        cols = [c for c in ["symbol", "name", "ann_date", "period", "period_end",
                            "holders", "holders_prev", "holders_chg_pct",
                            "avg_hold_value"] if c in d.columns]
        frames.append(d[cols])
        print(f"    {p}: {len(d)} 条", flush=True)
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df = df.dropna(subset=["ann_date"])
    for c in ("holders", "holders_prev", "holders_chg_pct", "avg_hold_value"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------- 分析师研报
_REPORT_URL = "https://reportapi.eastmoney.com/report/list"


def load_reports(start: str = "2017-01-01", end: str | None = None,
                 refresh: bool = False, page_size: int = 100) -> pd.DataFrame:
    """券商研报（含评级变动与 EPS 预测）—— 分析师预期修正因子的原料。

    为什么值得单独接：**分析师预期修正（revision momentum）是学术界最稳健的
    异象之一**，而且它是价格和资金流之外的独立信息源。东财这个接口可以
    **按日期范围拉全市场**（不必逐只股票），十年约 1000 次请求。

    point-in-time：用 `publishDate`，天然干净 —— 研报发布日就是信息可得日。

    字段：
      rating_value / last_rating_value  东财评级分值（越大越看好）
      eps_this / eps_next               当年 / 次年 EPS 预测
    """
    cache = _alt_dir() / "reports.csv"
    if cache.exists() and not refresh:
        d = pd.read_csv(cache, parse_dates=["publish_date"], dtype={"symbol": str})
        # 留 45 天容差：请求 start=2017-01-01 时，缓存里最早的研报可能是 2017-01-02
        # （那几天没人发研报）。严格比较会误判缓存不完整，导致每次都重下 14 万篇。
        if not d.empty and d["publish_date"].min() <= pd.Timestamp(start) + \
                pd.Timedelta(days=45):
            return d

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    months = pd.date_range(start, end, freq="MS")
    rows: list[dict] = []
    for i, m0 in enumerate(months, 1):
        m1 = (m0 + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        page = 1
        while page <= 40:
            params = {"industryCode": "*", "pageSize": page_size, "pageNo": page,
                      "qType": 0, "beginTime": m0.strftime("%Y-%m-%d"), "endTime": m1}
            try:
                # 用线程局部会话（_session 那个模块级变量已被 _get_session 取代）
                r = _get_session().get(
                    _REPORT_URL, params=params, timeout=20,
                    headers={"Referer": "https://data.eastmoney.com/"})
                js = json.loads(r.text.strip().lstrip("(").rstrip(")"))
            except Exception as e:                             # noqa: BLE001
                # 别静默吞异常 —— 之前少了 import json，NameError 被这里吃掉，
                # 表现成"接口返回 0 篇"，排查了好一阵。
                print(f"  ! 研报 {m0:%Y-%m} 第{page}页失败: "
                      f"{type(e).__name__}: {str(e)[:70]}", flush=True)
                break
            data = js.get("data") or []
            if not data:
                break
            rows += data
            if page >= int(js.get("TotalPage") or 1):
                break
            page += 1
            time.sleep(0.25)
        if i % 12 == 0:
            print(f"    研报 {i}/{len(months)} 月，累计 {len(rows)} 篇", flush=True)
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = {"stockCode": "symbol", "stockName": "name", "publishDate": "publish_date",
            "orgSName": "org", "emRatingValue": "rating_value",
            "lastEmRatingValue": "last_rating_value", "emRatingName": "rating_name",
            "predictThisYearEps": "eps_this", "predictNextYearEps": "eps_next",
            "indvInduName": "industry"}
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    for c in ("rating_value", "last_rating_value", "eps_this", "eps_next"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["publish_date", "symbol"]).drop_duplicates()
    df.to_csv(cache, index=False)
    return df


def reports_to_factors(reports: pd.DataFrame, dates: pd.DatetimeIndex,
                       symbols: list[str], window: int = 60) -> dict[str, pd.DataFrame]:
    """把研报事件流转成日频因子面板。

    - upgrade_net : 过去 window 日「评级上调数 - 下调数」
    - eps_revision: 过去 window 日当年 EPS 预测均值 / 更早一段的均值 - 1
    - coverage    : 过去 window 日覆盖研报数（关注度代理）

    全部只用 publish_date <= 当日的数据，无未来函数。
    """
    empty = pd.DataFrame(np.nan, index=dates, columns=symbols)
    if reports.empty:
        return {"upgrade_net": empty, "eps_revision": empty.copy(),
                "coverage": empty.copy()}

    r = reports[reports["symbol"].isin(symbols)].copy()
    r["chg"] = np.sign(r["rating_value"].fillna(0) - r["last_rating_value"].fillna(0))
    # 按日期 × 股票汇总成稀疏事件表，再滚动累计
    d0 = r.groupby([r["publish_date"].dt.normalize(), "symbol"]).agg(
        chg=("chg", "sum"), n=("chg", "size"), eps=("eps_this", "mean"))

    def widen(col: str) -> pd.DataFrame:
        w = d0[col].unstack("symbol").reindex(columns=symbols)
        idx = w.index.union(dates)
        return w.reindex(idx).reindex(dates)

    chg = widen("chg").fillna(0.0)
    cnt = widen("n").fillna(0.0)
    eps = widen("eps")

    upgrade_net = chg.rolling(window, min_periods=1).sum()
    coverage = cnt.rolling(window, min_periods=1).sum()
    eps_recent = eps.rolling(window, min_periods=1).mean()
    eps_prior = eps.shift(window).rolling(window, min_periods=1).mean()
    eps_rev = eps_recent / eps_prior.replace(0, np.nan) - 1.0

    return {"upgrade_net": upgrade_net, "eps_revision": eps_rev, "coverage": coverage}


def quarter_ends(start: str = "2016-03-31", end: str = "2026-06-30") -> list[str]:
    """生成季末报告期列表，格式 YYYYMMDD。"""
    return [d.strftime("%Y%m%d")
            for d in pd.date_range(start, end, freq="QE")]


def to_pit_panel(df: pd.DataFrame, field: str, dates: pd.DatetimeIndex,
                 symbols: list[str], date_col: str = "ann_date") -> pd.DataFrame:
    """把「公告日 + 数值」的事件表摊成 point-in-time 面板。

    每个值从**公告日当天**起生效，之后一直有效直到下一次公告。
    这是财务数据唯一正确的对齐方式。
    """
    if df.empty or field not in df:
        return pd.DataFrame(np.nan, index=dates, columns=symbols)
    d = df[[date_col, "symbol", field]].dropna(subset=[date_col])
    d = d.sort_values(date_col).drop_duplicates([date_col, "symbol"], keep="last")
    wide = d.pivot(index=date_col, columns="symbol", values=field)
    wide = wide.reindex(columns=symbols)
    # 公告日不一定是交易日 -> 先并入交易日轴再前向填充
    idx = wide.index.union(dates)
    return wide.reindex(idx).ffill().reindex(dates)
