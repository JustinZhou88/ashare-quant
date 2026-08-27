"""横截面向量化回测引擎（筛选阶段用）。

性能思路：**对股票维度向量化，只对时间维度循环**。
400 个策略 × 2500 天 = 100 万次循环，每次是 300 只股票的 numpy 向量运算，
比"每只股票单独循环"快两个数量级。

成交模型（每一条都是为了不骗自己）：
  - t 日收盘出信号 → t+1 日**开盘**成交（不是收盘价成交，那是未来函数）
  - T+1：开盘买入后最早次日开盘卖出，本模型天然满足
  - 开盘涨停买不进 / 开盘跌停卖不出（用**不复权**价判定）
  - 停牌日（成交量=0）不可交易，仓位原地不动
  - 双边滑点 + 佣金 + 过户费，卖出加印花税
  - 止损在**收盘**判定、次日开盘执行（日线数据做盘中止损填单是自欺欺人）

资金模型：最多同时持有 K 只（cfg.max_positions），每只固定占 1/K 资金，
剩余为现金（收益记 0，不吃利息）。名额不够时已持仓优先保留，
新信号按过去 20 日成交额从高到低补位。
这样"一次只买 1 只"和"一次买 30 只"的策略才能在同一把尺子上比较。

简化之处：不做整手取整、不算最低 5 元佣金、不做现金账户明细。
对 10 万以上资金、成交额 3000 万以上的标的，误差 < 0.1%。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BacktestConfig, limit_pct


@dataclass
class Panel:
    """对齐后的行情面板。所有 DataFrame 同 index(日期) 同 columns(代码)。"""
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    raw_open: pd.DataFrame
    raw_close: pd.DataFrame
    tradable: pd.DataFrame
    limits: np.ndarray          # 每只股票的涨跌停幅度，shape (n_symbols,)
    names: dict[str, str]

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def slice(self, start=None, end=None) -> "Panel":
        sl = slice(start, end)
        return Panel(
            open=self.open.loc[sl], high=self.high.loc[sl], low=self.low.loc[sl],
            close=self.close.loc[sl], volume=self.volume.loc[sl],
            amount=self.amount.loc[sl], raw_open=self.raw_open.loc[sl],
            raw_close=self.raw_close.loc[sl], tradable=self.tradable.loc[sl],
            limits=self.limits, names=self.names,
        )


def build_panel(data: dict[str, pd.DataFrame], min_days: int = 300) -> Panel:
    """把 {symbol: df} 对齐成面板。"""
    data = {s: d for s, d in data.items() if len(d) >= min_days}
    if not data:
        raise ValueError("没有足够长历史的股票")
    syms = sorted(data)

    def grab(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: data[s][col] for s in syms}).sort_index()

    names = {s: (str(data[s]["name"].iloc[0]) if "name" in data[s] else "") for s in syms}
    tradable = pd.DataFrame({s: data[s]["tradable"] for s in syms}).sort_index()
    tradable = tradable.fillna(False).astype(bool)
    limits = np.array([limit_pct(s, names.get(s, "")) for s in syms])

    return Panel(
        open=grab("open"), high=grab("high"), low=grab("low"), close=grab("close"),
        volume=grab("volume"), amount=grab("amount"),
        raw_open=grab("raw_open"), raw_close=grab("raw_close"),
        tradable=tradable, limits=limits, names=names,
    )


def liquidity_mask(panel: Panel, min_amount: float = 3e7, lookback: int = 60,
                   min_history: int = 250) -> pd.DataFrame:
    """可交易域：**在当时**流动性够、上市够久、没停牌的股票才允许持仓。

    全部用 rolling / cumsum 算，只看历史，没有未来函数。
    不加这层过滤，回测会大量成交在早年根本买不到的微盘股上。
    """
    amt_ma = panel.amount.rolling(lookback, min_periods=lookback).mean()
    age = panel.close.notna().cumsum()          # 上市至今的交易日数
    return (amt_ma >= min_amount) & (age >= min_history) & panel.tradable


@dataclass
class Result:
    ret: pd.Series               # 组合日收益（最多 K 只、每只 1/K 仓位）
    entries: pd.Series           # 每日新开仓笔数 —— 任意窗口的交易数由它算出
    exposure: float              # 平均资金利用率（持仓数/K）
    blocked_entries: int         # 因涨停/停牌买不进的次数（诊断用）
    pos: pd.DataFrame | None = None      # 仅在 with_trades=True 时保留
    trades: pd.DataFrame | None = None

    @property
    def equity(self) -> pd.Series:
        return (1 + self.ret).cumprod()

    def n_trades(self, start=None, end=None) -> int:
        return int(self.entries.loc[slice(start, end)].sum())


def _cost_rates(cfg: BacktestConfig) -> tuple[float, float]:
    c = cfg.cost
    entry = c.slippage_rate + c.commission_rate + c.transfer_fee_rate
    exit_ = c.slippage_rate + c.commission_rate + c.transfer_fee_rate + c.stamp_tax_rate
    return entry, exit_


def run(panel: Panel, signal: pd.DataFrame, cfg: BacktestConfig,
        mask: pd.DataFrame | None = None, with_trades: bool = False,
        score: pd.DataFrame | None = None) -> Result:
    """跑一个策略。

    signal:      与 panel 同形状的 0/1 矩阵，表示 **t 日收盘时希望持有的仓位**。
                 必须只使用 t 日及以前的信息（策略层保证，见 strategies/library.py）。
    mask:        可选的可交易域（如流动性过滤），False 处强制不持仓。
    with_trades: 筛选阶段设 False（省掉逐笔还原，快一倍）；决赛圈再设 True。
    score:       可选的信号强度矩阵。候选数超过 K 个名额时，按 score 从高到低挑。
                 截面因子策略必须传这个 —— 否则按流动性挑，等于把因子排序丢了。
                 不传则沿用流动性优先（对 0/1 技术指标策略是合理的默认）。
    """
    sig = signal.reindex(index=panel.dates, columns=panel.symbols).fillna(0.0)
    sig = (sig.to_numpy() > 0.5)
    if mask is not None:
        m = mask.reindex(index=panel.dates, columns=panel.symbols).fillna(False).to_numpy()
        sig = sig & m

    op = panel.open.to_numpy(float)
    cl = panel.close.to_numpy(float)
    lo = panel.low.to_numpy(float)
    r_op = panel.raw_open.to_numpy(float)
    r_cl = panel.raw_close.to_numpy(float)
    trad = panel.tradable.to_numpy(bool)
    lim = panel.limits

    n_t, n_s = cl.shape
    entry_rate, exit_rate = _cost_rates(cfg)
    K = max(int(cfg.max_positions), 1)
    # 名额不够时的排序依据：优先用策略给的 score，没有就用流动性
    # （流动性优先在现实中也更容易成交，对 0/1 信号是合理默认）。只用过去 20 日数据。
    if score is not None:
        rank_by = score.reindex(index=panel.dates, columns=panel.symbols).to_numpy(float)
        rank_by = np.nan_to_num(rank_by, nan=-np.inf)
    else:
        rank_by = panel.amount.rolling(20, min_periods=1).mean().to_numpy(float)
        rank_by = np.nan_to_num(rank_by, nan=0.0)

    # ---- 涨跌停：用前一交易日的**不复权**收盘价推涨跌停价 -----------------
    prev_rc = np.vstack([np.full((1, n_s), np.nan), r_cl[:-1]])
    up_px = np.round(prev_rc * (1.0 + lim), 2)
    dn_px = np.round(prev_rc * (1.0 - lim), 2)
    has_raw = np.isfinite(prev_rc) & np.isfinite(r_op)
    open_limit_up = has_raw & (r_op >= up_px - 0.005)
    open_limit_dn = has_raw & (r_op <= dn_px + 0.005)

    # 没有不复权价时降级：后复权的开盘涨幅 ≈ 官方涨跌幅（复权因子已抵消除权缺口）。
    # 留 0.2% 容差，宁可漏做几笔也不要把"一字板买入"算成收益。
    prev_c = np.vstack([np.full((1, n_s), np.nan), cl[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        open_ret = op / prev_c - 1.0
    fb_up = np.isfinite(open_ret) & (open_ret >= lim - 0.002)
    fb_dn = np.isfinite(open_ret) & (open_ret <= -lim + 0.002)
    open_limit_up = np.where(has_raw, open_limit_up, fb_up)
    open_limit_dn = np.where(has_raw, open_limit_dn, fb_dn)
    valid_px = np.isfinite(op) & np.isfinite(cl) & (op > 0) & (cl > 0)
    can_buy = trad & valid_px & ~open_limit_up
    can_sell = trad & valid_px & ~open_limit_dn

    # ---- 时间循环，股票维度向量化 ---------------------------------------
    pos = np.zeros((n_t, n_s), dtype=bool)
    port_ret = np.zeros(n_t)
    n_entries = np.zeros(n_t, dtype=np.int32)
    entry_px = np.zeros(n_s)
    held = np.zeros(n_s, dtype=int)
    blocked = 0

    prev_pos = np.zeros(n_s, dtype=bool)
    for i in range(1, n_t):
        want = sig[i - 1].copy()          # 昨收信号，今开执行

        # 收盘止损 / 超期强平：昨收判定，今开执行
        if cfg.stop_loss is not None:
            hit = prev_pos & (cl[i - 1] <= entry_px * (1 - cfg.stop_loss)) & (held >= 1)
            want &= ~hit
        if cfg.take_profit is not None:
            tp = prev_pos & (cl[i - 1] >= entry_px * (1 + cfg.take_profit)) & (held >= 1)
            want &= ~tp
        if cfg.max_hold_days is not None:
            want &= ~(prev_pos & (held >= cfg.max_hold_days))

        cand = ~prev_pos & want & can_buy[i]
        closing = prev_pos & ~want & can_sell[i]
        blocked += int(np.sum(~prev_pos & want & ~can_buy[i]))
        holding = prev_pos & ~closing

        # 组合约束：最多同时持有 K 只。已持仓的优先保留，剩余名额给流动性最好的新信号。
        slots = K - int(holding.sum())
        n_cand = int(cand.sum())
        if slots <= 0:
            opening = np.zeros(n_s, dtype=bool)
        elif n_cand > slots:
            idx = np.flatnonzero(cand)
            pick = idx[np.argsort(-rank_by[i - 1][idx], kind="stable")[:slots]]
            opening = np.zeros(n_s, dtype=bool)
            opening[pick] = True
        else:
            opening = cand

        cur = (holding | opening)
        pos[i] = cur

        # 收益归因
        r = np.zeros(n_s)
        with np.errstate(divide="ignore", invalid="ignore"):
            # 新开仓：今开买入 -> 今收，扣买入成本
            r_open_ = np.where(opening, cl[i] / op[i] - 1.0 - entry_rate, 0.0)
            # 持有中：昨收 -> 今收
            r_hold_ = np.where(holding, cl[i] / cl[i - 1] - 1.0, 0.0)
            # 平仓：昨收 -> 今开卖出，扣卖出成本
            r_exit_ = np.where(closing, op[i] / cl[i - 1] - 1.0 - exit_rate, 0.0)
        r = np.nan_to_num(r_open_) + np.nan_to_num(r_hold_) + np.nan_to_num(r_exit_)
        # 每个持仓固定占用 1/K 资金，未用满的部分是现金（收益记 0，不吃利息，保守）
        port_ret[i] = r.sum() / K
        n_entries[i] = int(opening.sum())

        entry_px = np.where(opening, op[i], entry_px)
        held = np.where(opening, 0, np.where(cur, held + 1, 0))
        prev_pos = cur

    ret_s = pd.Series(port_ret, index=panel.dates)
    entries_s = pd.Series(n_entries, index=panel.dates)
    # 仓位 = 平均动用了多少比例的资金（持仓数 / K），不是"多少比例的股票被持有"
    res = Result(ret=ret_s, entries=entries_s,
                 exposure=float(pos.sum(axis=1).mean() / K), blocked_entries=blocked)
    if with_trades:
        res.pos = pd.DataFrame(pos, index=panel.dates, columns=panel.symbols)
        res.trades = extract_trades(panel, res.pos, cfg)
    return res


def extract_trades(panel: Panel, pos: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """从持仓矩阵还原逐笔交易（含 MAE/MFE）。"""
    entry_rate, exit_rate = _cost_rates(cfg)
    p = pos.to_numpy(bool)
    op = panel.open.to_numpy(float)
    cl = panel.close.to_numpy(float)
    hi = panel.high.to_numpy(float)
    lo = panel.low.to_numpy(float)
    dates = panel.dates
    syms = panel.symbols

    rows = []
    for j, sym in enumerate(syms):
        col = p[:, j]
        if not col.any():
            continue
        d = np.diff(col.astype(np.int8), prepend=0)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        if len(ends) < len(starts):          # 最后一笔还没平，按最后一天收盘估值
            ends = np.append(ends, len(col) - 1)
        for s_i, e_i in zip(starts, ends):
            in_px = op[s_i, j]
            out_px = op[e_i, j] if e_i > s_i else cl[e_i, j]
            if not (np.isfinite(in_px) and np.isfinite(out_px)) or in_px <= 0:
                continue
            seg_hi = np.nanmax(hi[s_i:e_i + 1, j])
            seg_lo = np.nanmin(lo[s_i:e_i + 1, j])
            gross = out_px / in_px - 1.0
            rows.append({
                "symbol": sym, "name": panel.names.get(sym, ""),
                "entry_date": dates[s_i], "exit_date": dates[e_i],
                "entry_px": in_px, "exit_px": out_px,
                "hold_days": int(e_i - s_i),
                "ret_gross": gross,
                "ret_net": gross - entry_rate - exit_rate,
                "mfe": seg_hi / in_px - 1.0,       # 持仓期最大浮盈
                "mae": seg_lo / in_px - 1.0,       # 持仓期最大浮亏
                "still_open": bool(e_i == len(col) - 1 and col[-1]),
            })
    if not rows:
        return pd.DataFrame(columns=["symbol", "name", "entry_date", "exit_date",
                                     "entry_px", "exit_px", "hold_days", "ret_gross",
                                     "ret_net", "mfe", "mae", "still_open"])
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
