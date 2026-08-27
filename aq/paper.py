"""模拟盘 —— 按策略规则把历史推荐回放成一个虚拟组合。

设计上是**纯函数**：`(推荐记录, 行情, 规则) -> (成交, 持仓, 净值曲线)`。
不维护可变状态文件。好处很实际：
  - 每次重算结果完全一致，不会因为中途崩了留下半截状态
  - 数据更新、规则调整后直接重跑，历史自动按新规则重放
  - 不存在"状态文件和实际数据对不上"这种最难查的 bug

成交假设与回测引擎**完全一致**，否则模拟盘和回测的数字没法互相印证：
  - 推荐日（T）收盘出信号 → **T+1 开盘**买入
  - T+1 制度：买入当日不可卖，最早 T+2 开盘卖出
  - 开盘一字涨停买不进、跌停卖不出
  - 止损/止盈在**收盘**判定、次日开盘执行（日线上模拟盘中触价成交是自欺欺人）
  - 佣金 + 印花税 + 过户费 + 滑点，口径见 config.CostModel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BacktestConfig, limit_pct

JOURNAL = Path("journal/recommendations.csv")


@dataclass
class Trade:
    """一笔模拟交易。

    **价格分两套，别混**：
      - `open_px` / `close_px` 是**不复权**价，就是你在券商里看到、真金白银
        成交的那个数。股数按它算，界面也显示它。
      - `open_adj` / `close_adj` 是**后复权**价，只用来算收益率 —— 它包含分红
        再投资，是唯一正确的收益口径。

    踩过的坑：一开始全用后复权价，结果 12000 元预算除以虚高 2.8 倍的价格
    得到 68 股，取整到手就是 0 股，整笔交易被 `shares <= 0` 静默跳过。
    """
    symbol: str
    name: str
    open_date: str
    open_px: float            # 不复权，实际成交价
    open_adj: float           # 后复权，算收益用
    shares: int
    cost_in: float
    close_date: str = ""
    close_px: float = 0.0
    close_adj: float = 0.0
    cost_out: float = 0.0
    reason: str = ""          # 平仓原因：止损/止盈/到期/仍持有
    scan_date: str = ""

    @property
    def is_open(self) -> bool:
        return not self.close_date

    @property
    def notional(self) -> float:
        return self.shares * self.open_px

    def gross_ret(self, mark_adj: float | None = None) -> float:
        """含分红的毛收益率（后复权口径）。"""
        end = self.close_adj if self.close_date else (mark_adj or self.open_adj)
        return end / self.open_adj - 1.0 if self.open_adj > 0 else 0.0

    def pnl(self, mark_adj: float | None = None) -> float:
        return self.notional * self.gross_ret(mark_adj) - self.cost_in - self.cost_out

    def ret(self, mark_adj: float | None = None) -> float:
        base = self.notional + self.cost_in
        return self.pnl(mark_adj) / base if base > 0 else 0.0


@dataclass
class Result:
    equity: pd.Series                  # 每日总权益（现金 + 持仓市值）
    cash: pd.Series
    trades: list[Trade] = field(default_factory=list)
    init_cash: float = 100_000.0
    blocked: list[str] = field(default_factory=list)   # 涨停买不进等
    pending: list[dict] = field(default_factory=list)  # 已推荐但还没到执行日

    @property
    def open_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.is_open]

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_open]


def _load_journal() -> pd.DataFrame:
    if not JOURNAL.exists():
        return pd.DataFrame()
    j = pd.read_csv(JOURNAL, dtype={"symbol": str})
    if "dry_run" in j:
        j = j[~j["dry_run"].astype(bool)]
    if j.empty:
        return j
    j["scan_date"] = pd.to_datetime(j["scan_date"])
    return j.sort_values("scan_date")


def simulate(panel, rules: dict | None = None, init_cash: float = 100_000.0,
             journal: pd.DataFrame | None = None,
             cfg: BacktestConfig | None = None) -> Result:
    """把推荐记录回放成组合。

    rules: rules.yaml 的 risk 段（position_size / stop_loss / take_profit /
           max_hold_days / max_positions）。缺省用 BacktestConfig 的默认值。
    """
    R = rules or {}
    cfg = cfg or BacktestConfig()
    pos_pct = float(R.get("position_size") or 0.2)
    stop = R.get("stop_loss")
    take = R.get("take_profit")
    max_hold = R.get("max_hold_days")
    max_pos = int(R.get("max_positions") or cfg.max_positions)
    stop = float(stop) if stop else None
    take = float(take) if take else None
    max_hold = int(max_hold) if max_hold else None

    j = journal if journal is not None else _load_journal()
    dates = panel.dates
    empty = Result(equity=pd.Series(dtype=float), cash=pd.Series(dtype=float),
                   init_cash=init_cash)
    if j.empty:
        empty.blocked = ["还没有推荐记录 —— 先跑一次扫描"]
        return empty
    if len(dates) == 0:
        empty.blocked = ["没有行情数据"]
        return empty

    # 第一笔推荐的次一交易日开始计算净值
    first = dates[dates > j["scan_date"].min()]
    if len(first) == 0:
        # 常见且正常：周五收盘后出的推荐，执行价是下周一开盘，
        # 而下周一的行情还没入库。别静默返回空，说清楚在等什么。
        empty.blocked = [
            f"最新推荐日 {j['scan_date'].max().date()}，"
            f"行情最后一个交易日 {dates[-1].date()}",
            "执行价是推荐日的**次一交易日开盘**，该日行情尚未入库，模拟盘还无法开始。",
            "跑一次「更新数据」后即可开始计算。",
        ]
        return empty
    sim_dates = dates[dates >= first[0]]

    # 推荐 -> {执行日: [代码]}，执行日 = 推荐日的次一交易日
    todo: dict[pd.Timestamp, list[tuple[str, str]]] = {}
    pending: list[dict] = []
    for _, r in j.iterrows():
        nxt = dates[dates > r["scan_date"]]
        if len(nxt) == 0:
            # 推荐日之后还没有交易日（常见于周五收盘后扫描）。
            # 不能静默丢弃 —— 否则你会看到推荐页有票、模拟盘没有，
            # 却不知道是"还没执行"还是"被规则挡了"。
            pending.append({"symbol": str(r["symbol"]).zfill(6),
                            "name": str(r.get("name") or ""),
                            "scan_date": str(pd.Timestamp(r["scan_date"]).date()),
                            "why": "等下一个交易日开盘执行"})
            continue
        todo.setdefault(nxt[0], []).append(
            (str(r["symbol"]).zfill(6), str(r.get("name") or "")))

    op = panel.open
    cl = panel.close
    r_op = panel.raw_open
    r_cl = panel.raw_close
    trad = panel.tradable

    cash = init_cash
    live: dict[str, Trade] = {}
    trades: list[Trade] = []
    blocked: list[str] = []
    eq_rows, cash_rows = [], []

    def px(frame, d, s) -> float:
        try:
            v = float(frame.loc[d, s])
            return v if np.isfinite(v) and v > 0 else np.nan
        except Exception:                                      # noqa: BLE001
            return np.nan

    for i, d in enumerate(sim_dates):
        prev = sim_dates[i - 1] if i > 0 else None

        # ---------- 1) 先处理平仓（昨收判定，今开执行） ----------
        for sym in list(live):
            t = live[sym]
            if prev is None:
                continue
            pc = px(cl, prev, sym)
            if not np.isfinite(pc):
                continue
            held = len(sim_dates[(sim_dates > pd.Timestamp(t.open_date)) &
                                 (sim_dates <= prev)])
            # 止损/止盈必须在**同一口径**下比较：pc 是后复权收盘，
            # 所以门槛也要用后复权成本 open_adj。拿它跟不复权的 open_px 比
            # 会因为复权系数（实测有 2.8 倍的）产生完全错误的触发。
            why = ""
            if stop and pc <= t.open_adj * (1 - stop) and held >= 1:
                why = f"止损 −{stop:.0%}"
            elif take and pc >= t.open_adj * (1 + take) and held >= 1:
                why = f"止盈 +{take:.0%}"
            elif max_hold and held >= max_hold:
                why = f"到期 {max_hold} 日"
            if not why:
                continue
            o = px(op, d, sym)
            if not np.isfinite(o) or not bool(trad.loc[d, sym]):
                continue
            # 跌停卖不出
            prc, pro = px(r_cl, prev, sym), px(r_op, d, sym)
            lim = limit_pct(sym, panel.names.get(sym, ""))
            if np.isfinite(prc) and np.isfinite(pro) and \
                    pro <= round(prc * (1 - lim), 2) + 0.005:
                blocked.append(f"{d.date()} {sym} 跌停卖不出")
                continue
            o_raw = px(r_op, d, sym)
            if not np.isfinite(o_raw):
                continue
            sell_raw = o_raw * (1 - cfg.cost.slippage_rate)
            sell_adj = o * (1 - cfg.cost.slippage_rate)
            proceeds = t.notional * (sell_adj / t.open_adj)   # 收益按后复权
            t.close_date, t.close_px, t.close_adj = str(d.date()), sell_raw, sell_adj
            t.cost_out = cfg.cost.sell_cost(proceeds)
            t.reason = why
            cash += proceeds - t.cost_out
            del live[sym]

        # ---------- 2) 再处理开仓 ----------
        for sym, nm in todo.get(d, []):
            if sym in live:
                blocked.append(f"{d.date()} {sym} 已持有，跳过重复建仓")
                continue
            if len(live) >= max_pos:
                # 别静默丢弃 —— 按每周 2~3 只、持有 15 个交易日的节奏，
                # 大约第 3 周就会撞上这个上限。不记录的话，你会看到推荐页有票、
                # 模拟盘却没买，而且完全不知道为什么。
                blocked.append(
                    f"{d.date()} {sym} 组合已满 {max_pos} 只"
                    f"（{'、'.join(sorted(live))}），未建仓")
                continue
            if sym not in panel.symbols:
                continue
            o = px(op, d, sym)
            if not np.isfinite(o) or not bool(trad.loc[d, sym]):
                blocked.append(f"{d.date()} {sym} 停牌/无价，未建仓")
                continue
            # 涨停买不进
            prc, pro = (px(r_cl, prev, sym) if prev is not None else np.nan,
                        px(r_op, d, sym))
            lim = limit_pct(sym, panel.names.get(sym, ""))
            if np.isfinite(prc) and np.isfinite(pro) and \
                    pro >= round(prc * (1 + lim), 2) - 0.005:
                blocked.append(f"{d.date()} {sym} 一字涨停买不进")
                continue
            o_raw = px(r_op, d, sym)
            if not np.isfinite(o_raw):
                blocked.append(f"{d.date()} {sym} 无不复权价，未建仓")
                continue
            # 股数按**不复权**价算 —— 那才是实际掏钱的价格。
            # 用后复权价会算出虚高的价格、虚低的股数，甚至归零。
            buy_raw = o_raw * (1 + cfg.cost.slippage_rate)
            buy_adj = o * (1 + cfg.cost.slippage_rate)
            budget = init_cash * pos_pct
            shares = int(budget / buy_raw // cfg.lot_size) * cfg.lot_size
            if shares <= 0:
                blocked.append(
                    f"{d.date()} {sym} 单只预算 {budget:,.0f} 元不够买 1 手"
                    f"（{buy_raw:.2f} 元/股 × {cfg.lot_size} 股），未建仓")
                continue
            notional = shares * buy_raw
            fee = cfg.cost.buy_cost(notional)
            if notional + fee > cash:
                blocked.append(f"{d.date()} {sym} 现金不足，未建仓")
                continue
            cash -= notional + fee
            t = Trade(symbol=sym, name=nm or panel.names.get(sym, ""),
                      open_date=str(d.date()), open_px=buy_raw, open_adj=buy_adj,
                      shares=shares, cost_in=fee,
                      scan_date=str(prev.date()) if prev is not None else "")
            live[sym] = t
            trades.append(t)

        # ---------- 3) 估值 ----------
        mv = 0.0
        for sym, t in live.items():
            c = px(cl, d, sym)                      # 后复权收盘
            mv += t.notional * (c / t.open_adj if np.isfinite(c) and t.open_adj > 0 else 1.0)
        eq_rows.append(cash + mv)
        cash_rows.append(cash)

    for t in live.values():
        t.reason = "仍持有"

    return Result(equity=pd.Series(eq_rows, index=sim_dates),
                  cash=pd.Series(cash_rows, index=sim_dates),
                  trades=trades, init_cash=init_cash, blocked=blocked,
                  pending=pending)


def summarize(res: Result, panel, bench: pd.Series | None = None) -> dict:
    """模拟盘绩效。样本少时**不做年化**——把 3 个月的收益年化会得到荒谬的数字。"""
    eq = res.equity
    if eq.empty:
        return {"empty": True, "n_days": 0, "n_trades": 0}

    total = float(eq.iloc[-1] / res.init_cash - 1)
    ret = eq.pct_change().dropna()
    mdd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    closed = res.closed_trades
    wins = [t for t in closed if t.pnl() > 0]

    out = {
        "empty": False,
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "n_days": len(eq), "init_cash": res.init_cash,
        "equity": float(eq.iloc[-1]), "cash": float(res.cash.iloc[-1]),
        "total_return": total,
        "max_drawdown": mdd,
        "n_trades": len(res.trades), "n_closed": len(closed),
        "n_open": len(res.open_trades),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "realized_pnl": float(sum(t.pnl() for t in closed)),
        "blocked": len(res.blocked),
    }

    # 未实现盈亏按最新收盘价
    last = panel.dates[-1]
    unreal = 0.0
    for t in res.open_trades:
        try:
            c = float(panel.close.loc[last, t.symbol])          # 后复权
            if np.isfinite(c):
                unreal += t.pnl(c)
        except Exception:                                      # noqa: BLE001
            pass
    out["unrealized_pnl"] = unreal

    if bench is not None and len(bench):
        b = bench.reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            out["bench_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
            out["excess"] = total - out["bench_return"]

    # 只有样本够长才给年化和夏普 —— 否则是误导
    if len(ret) >= 120 and ret.std(ddof=1) > 0:
        out["sharpe"] = float(ret.mean() / ret.std(ddof=1) * np.sqrt(244))
        out["cagr"] = float((1 + total) ** (244 / len(eq)) - 1)
    else:
        out["sharpe"] = None
        out["cagr"] = None
        out["too_short"] = True
    return out


def to_dict(res: Result, panel) -> dict:
    """给前端用的结构。"""
    last = panel.dates[-1] if len(panel.dates) else None

    def mark(t: Trade, frame) -> float | None:
        if not t.is_open or last is None:
            return None
        try:
            c = float(frame.loc[last, t.symbol])
            return c if np.isfinite(c) else None
        except Exception:                                      # noqa: BLE001
            return None

    def row(t: Trade) -> dict:
        m_adj = mark(t, panel.close)          # 后复权，算收益
        m_raw = mark(t, panel.raw_close)      # 不复权，界面显示
        return {"symbol": t.symbol, "name": t.name, "open_date": t.open_date,
                "open_px": round(t.open_px, 2), "shares": t.shares,
                "close_date": t.close_date, "close_px": round(t.close_px, 2) or None,
                "mark_px": round(m_raw, 2) if m_raw else None,
                "pnl": round(t.pnl(m_adj), 2), "ret": t.ret(m_adj),
                "reason": t.reason, "cost": round(t.cost_in + t.cost_out, 2)}

    eq = res.equity
    return {
        "equity": [{"d": str(d.date()), "v": round(float(v), 2)}
                   for d, v in eq.items()] if len(eq) else [],
        "open": [row(t) for t in res.open_trades],
        "closed": [row(t) for t in res.closed_trades],
        "pending": res.pending,
        "blocked": res.blocked[-20:],
    }


def pending_actions(res: Result, panel, rules: dict | None = None,
                    cfg: BacktestConfig | None = None) -> list[dict]:
    """**下一交易日开盘该做什么** —— 卖出提醒。

    系统每周只产出「买什么」，从不说「卖什么」。但规则里有止损、止盈、
    最长持有，这些触发条件每天都在变，靠人肉盯着算持有天数迟早会漏。

    判定口径与 simulate 完全一致：用**最新收盘价**判，次一开盘执行。
    """
    R = rules or {}
    cfg = cfg or BacktestConfig()
    stop = float(R["stop_loss"]) if R.get("stop_loss") else None
    take = float(R["take_profit"]) if R.get("take_profit") else None
    max_hold = int(R["max_hold_days"]) if R.get("max_hold_days") else None

    if not res.open_trades or len(panel.dates) == 0:
        return []
    last = panel.dates[-1]
    out = []
    for t in res.open_trades:
        try:
            adj = float(panel.close.loc[last, t.symbol])
            raw = float(panel.raw_close.loc[last, t.symbol])
        except Exception:                                      # noqa: BLE001
            continue
        if not (np.isfinite(adj) and np.isfinite(raw)):
            continue
        held = int((panel.dates > pd.Timestamp(t.open_date)).sum())
        ret = t.gross_ret(adj)

        act, why, urgency = "持有", "", "hold"
        if stop and adj <= t.open_adj * (1 - stop) and held >= 1:
            act, why, urgency = "卖出", f"触发止损 −{stop:.0%}（当前 {ret:+.1%}）", "sell"
        elif take and adj >= t.open_adj * (1 + take) and held >= 1:
            act, why, urgency = "卖出", f"触发止盈 +{take:.0%}（当前 {ret:+.1%}）", "sell"
        elif max_hold and held >= max_hold:
            act, why, urgency = "卖出", f"持有已满 {held} 日（上限 {max_hold}）", "sell"
        elif max_hold and held >= max_hold - 3:
            act, why, urgency = "留意", f"已持有 {held} 日，还剩 {max_hold - held} 日到期", "warn"
        elif stop:
            # 距止损还有多远 —— 这个数比「当前盈亏」更值得盯
            gap = (adj - t.open_adj * (1 - stop)) / adj
            if gap < 0.03:
                act, why, urgency = "留意", f"距止损线仅 {gap:.1%}", "warn"
            else:
                why = f"距止损线 {gap:.1%}，已持有 {held} 日"

        out.append({
            "symbol": t.symbol, "name": t.name, "action": act, "why": why,
            "urgency": urgency, "held_days": held, "ret": ret,
            "open_px": round(t.open_px, 2), "last_px": round(raw, 2),
            "shares": t.shares,
            "stop_px": round(t.open_px * (1 - stop), 2) if stop else None,
            "take_px": round(t.open_px * (1 + take), 2) if take else None,
            "deadline_days": (max_hold - held) if max_hold else None,
        })
    order = {"sell": 0, "warn": 1, "hold": 2}
    return sorted(out, key=lambda x: (order.get(x["urgency"], 3), -abs(x["ret"])))
