"""回测引擎正确性测试 —— 用手工构造的行情验证每一条 A股规则。

跑法： .venv/bin/python tests/test_engine.py

不测的回测引擎就是玩具。这里逐条验证：
成交时点、T+1、涨跌停封锁、停牌、交易成本、无未来函数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aq.config import BacktestConfig, CostModel          # noqa: E402
from aq.engine.panel import Panel, run                   # noqa: E402

SYM = "600001"          # 主板，涨跌停 ±10%


def mkpanel(open_, close, high=None, low=None, volume=None,
            raw_open=None, raw_close=None) -> Panel:
    """构造单只股票的合成面板。默认不复权 = 后复权（无分红）。"""
    n = len(open_)
    idx = pd.bdate_range("2020-01-01", periods=n)
    f = lambda v: pd.DataFrame({SYM: np.asarray(v, float)}, index=idx)   # noqa: E731
    high = high if high is not None else np.maximum(open_, close)
    low = low if low is not None else np.minimum(open_, close)
    volume = volume if volume is not None else [1e6] * n
    raw_open = raw_open if raw_open is not None else open_
    raw_close = raw_close if raw_close is not None else close
    vol = f(volume)
    return Panel(
        open=f(open_), high=f(high), low=f(low), close=f(close),
        volume=vol, amount=vol * 10.0,
        raw_open=f(raw_open), raw_close=f(raw_close),
        tradable=(vol > 0), limits=np.array([0.10]), names={SYM: "测试股"},
    )


def sig(vals, panel) -> pd.DataFrame:
    return pd.DataFrame({SYM: np.asarray(vals, float)}, index=panel.dates)


# 单只股票的用例都用 K=1，让这只股票占满仓，收益不被 1/K 稀释
NOCOST = BacktestConfig(cost=CostModel(commission_rate=0, commission_min=0,
                                       stamp_tax_rate=0, transfer_fee_rate=0,
                                       slippage_rate=0), max_positions=1)
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


# ------------------------------------------------------------------ 1. 成交时点
def test_execution_timing() -> None:
    """信号在 t 日收盘产生，必须在 t+1 **开盘**成交，吃不到 t 日的行情。"""
    # 第 3 天(idx=3) 大涨：开 10 收 20。信号在第 3 天收盘才出现。
    op = [10, 10, 10, 10, 10, 10]
    cl = [10, 10, 10, 20, 20, 20]
    P = mkpanel(op, cl)
    s = sig([0, 0, 0, 1, 1, 1], P)          # t=3 收盘才知道
    r = run(P, s, NOCOST)

    check("信号日当天不产生收益（无未来函数）", abs(r.ret.iloc[3]) < 1e-12,
          f"第3天收益={r.ret.iloc[3]:.6f}，应为 0")
    # t=4 开盘买入，价格 10，当天收盘 20 -> +100%
    check("次日开盘成交，吃到开盘后的行情", abs(r.ret.iloc[4] - 1.0) < 1e-9,
          f"第4天收益={r.ret.iloc[4]:.6f}，应为 1.0")


# ------------------------------------------------------------------ 2. 交易成本
def test_costs() -> None:
    """一轮买卖的成本应等于 佣金×2 + 过户费×2 + 印花税 + 滑点×2。"""
    op = [10.0] * 6
    cl = [10.0] * 6
    P = mkpanel(op, cl)
    s = sig([0, 1, 1, 0, 0, 0], P)
    cfg = BacktestConfig(max_positions=1)
    r = run(P, s, cfg, with_trades=True)

    c = cfg.cost
    entry = c.slippage_rate + c.commission_rate + c.transfer_fee_rate
    exit_ = entry + c.stamp_tax_rate
    expect = (1 - entry) * (1 - exit_) - 1.0        # 两天的收益是相乘不是相加
    total = float((1 + r.ret).prod() - 1)
    check("价格不动时，总收益 = -往返成本", abs(total - expect) < 1e-9,
          f"实际 {total:.6f}，预期 {expect:.6f}")
    check("往返成本约 20 个基点", abs((entry + exit_) - 20.2e-4) < 1e-6,
          f"{(entry + exit_)*1e4:.1f}bp")


# ------------------------------------------------------------------ 3. T+1
def test_t_plus_1() -> None:
    """当日买入不能当日卖出：最短持仓 1 个交易日。"""
    op = [10.0] * 6
    cl = [10.0] * 6
    P = mkpanel(op, cl)
    s = sig([0, 1, 0, 0, 0, 0], P)          # 只在 t=1 收盘想持有
    r = run(P, s, NOCOST, with_trades=True)
    # t=2 开盘买入，t=2 收盘信号为 0 -> t=3 开盘卖出，持仓 1 天
    check("最短持仓 ≥ 1 个交易日（T+1）", int(r.trades["hold_days"].iloc[0]) >= 1,
          f"hold_days={r.trades['hold_days'].iloc[0]}")
    check("持仓期只有 t=2 和 t=3 两天在场", int(r.pos[SYM].sum()) == 1,
          f"持仓天数={int(r.pos[SYM].sum())}")


# ------------------------------------------------------------------ 4. 涨停封锁
def test_limit_up_blocks_entry() -> None:
    """开盘一字涨停买不进 —— 不封锁的话动量策略回测收益会凭空翻倍。"""
    # t=2 开盘价 = 前收 × 1.10，涨停
    op = [10.0, 10.0, 11.0, 12.0, 12.0, 12.0]
    cl = [10.0, 10.0, 11.5, 12.0, 12.0, 12.0]
    P = mkpanel(op, cl)
    s = sig([0, 1, 1, 1, 1, 1], P)
    r = run(P, s, NOCOST, with_trades=True)
    check("开盘涨停时买不进", not bool(r.pos[SYM].iloc[2]),
          f"第2天持仓={r.pos[SYM].iloc[2]}")
    check("被封锁的入场被计数", r.blocked_entries >= 1, f"blocked={r.blocked_entries}")
    check("次日不涨停则正常买入", bool(r.pos[SYM].iloc[3]),
          f"第3天持仓={r.pos[SYM].iloc[3]}")


def test_limit_down_blocks_exit() -> None:
    """开盘跌停卖不出 —— 不封锁的话回测会低估尾部风险。"""
    op = [10.0, 10.0, 10.0, 9.0, 8.5, 8.5]
    cl = [10.0, 10.0, 10.0, 9.2, 8.6, 8.6]
    P = mkpanel(op, cl)
    #  t=1 收盘想持有 -> t=2 开盘买入; t=2 收盘想清仓 -> t=3 开盘卖出，但 t=3 开盘跌停
    s = sig([0, 1, 0, 0, 0, 0], P)
    r = run(P, s, NOCOST, with_trades=True)
    check("开盘跌停时卖不出，被迫持有", bool(r.pos[SYM].iloc[3]),
          f"第3天持仓={r.pos[SYM].iloc[3]}（跌停日应仍持有）")
    check("跌停解除后成功卖出", not bool(r.pos[SYM].iloc[4]),
          f"第4天持仓={r.pos[SYM].iloc[4]}")


# ------------------------------------------------------------------ 5. 停牌
def test_suspension() -> None:
    """停牌日（成交量 0）不可交易。"""
    op = [10.0] * 6
    cl = [10.0] * 6
    vol = [1e6, 1e6, 0.0, 1e6, 1e6, 1e6]     # t=2 停牌
    P = mkpanel(op, cl, volume=vol)
    s = sig([0, 1, 1, 1, 1, 1], P)
    r = run(P, s, NOCOST, with_trades=True)
    check("停牌日不能买入", not bool(r.pos[SYM].iloc[2]),
          f"第2天持仓={r.pos[SYM].iloc[2]}")
    check("复牌后正常买入", bool(r.pos[SYM].iloc[3]))


# ------------------------------------------------------------------ 6. 止损
def test_stop_loss() -> None:
    """止损按收盘价判定、次日开盘执行（日线数据做盘中止损填单不现实）。"""
    op = [10.0, 10.0, 10.0, 10.0, 8.0, 8.0, 8.0]
    cl = [10.0, 10.0, 10.0, 8.5, 8.0, 8.0, 8.0]    # t=3 收盘跌破 -10%
    P = mkpanel(op, cl)
    s = sig([0, 1, 1, 1, 1, 1, 1], P)              # 信号一直想持有
    cfg = BacktestConfig(stop_loss=0.10, cost=NOCOST.cost, max_positions=1)
    r = run(P, s, cfg, with_trades=True)
    check("触发止损后次日开盘离场", not bool(r.pos[SYM].iloc[4]),
          f"第4天持仓={r.pos[SYM].iloc[4]}")
    check("止损当日仍持有（收盘才判定）", bool(r.pos[SYM].iloc[3]))


# ------------------------------------------------------------------ 7. 复权一致性
def test_adjusted_vs_raw() -> None:
    """涨跌停用不复权价判定：除权日不应误判为涨停。"""
    # 后复权连续上涨；不复权在 t=2 因除权从 20 掉到 10（-50%），但这不是跌停
    adj_o = [10.0, 10.2, 10.3, 10.4, 10.5, 10.6]
    adj_c = [10.1, 10.25, 10.35, 10.45, 10.55, 10.65]
    raw_o = [20.0, 20.4, 10.15, 10.4, 10.5, 10.6]
    raw_c = [20.2, 20.5, 10.175, 10.45, 10.55, 10.65]
    P = mkpanel(adj_o, adj_c, raw_open=raw_o, raw_close=raw_c)
    s = sig([0, 1, 1, 1, 1, 1], P)
    r = run(P, s, NOCOST, with_trades=True)
    # 用不复权判：前收 20.5，跌停价 18.45，开盘 10.15 <= 18.45 -> 会被判成"跌停"
    # 这正是为什么**买入**只看涨停：跌停不影响买入
    check("除权日不影响买入（只有涨停封锁买入）", bool(r.pos[SYM].iloc[2]),
          f"第2天持仓={r.pos[SYM].iloc[2]}")


# ------------------------------------------------------------------ 8. 交易还原
def test_trade_extraction() -> None:
    """逐笔交易还原：入场价、出场价、持仓天数、MAE/MFE。"""
    op = [10.0, 10.0, 10.0, 10.0, 12.0, 12.0]
    cl = [10.0, 10.0, 10.0, 11.0, 12.0, 12.0]
    hi = [10.0, 10.0, 10.0, 13.0, 12.0, 12.0]
    lo = [10.0, 10.0, 10.0, 9.0, 12.0, 12.0]
    P = mkpanel(op, cl, high=hi, low=lo)
    s = sig([0, 1, 1, 0, 0, 0], P)
    r = run(P, s, NOCOST, with_trades=True)
    t = r.trades.iloc[0]
    check("入场价 = 买入日开盘价", abs(t["entry_px"] - 10.0) < 1e-9, f"{t['entry_px']}")
    check("出场价 = 卖出日开盘价", abs(t["exit_px"] - 12.0) < 1e-9, f"{t['exit_px']}")
    check("MFE = 持仓期最高价/入场价-1", abs(t["mfe"] - 0.30) < 1e-9, f"{t['mfe']}")
    check("MAE = 持仓期最低价/入场价-1", abs(t["mae"] + 0.10) < 1e-9, f"{t['mae']}")


# ------------------------------------------------------------------ 9. 掩码
def test_mask() -> None:
    """流动性掩码为 False 时强制不持仓。"""
    op = [10.0] * 6
    cl = [10.0] * 6
    P = mkpanel(op, cl)
    s = sig([1, 1, 1, 1, 1, 1], P)
    m = pd.DataFrame({SYM: [False] * 6}, index=P.dates)
    r = run(P, s, NOCOST, mask=m, with_trades=True)
    check("掩码为 False 时全程空仓", int(r.pos[SYM].sum()) == 0 if r.pos is not None
          else r.exposure == 0, f"exposure={r.exposure}")


# ------------------------------------------------------------------ 10. 持仓上限
def test_max_positions() -> None:
    """最多同时持有 K 只；名额不够时优先流动性好的。"""
    n = 6
    idx = pd.bdate_range("2020-01-01", periods=n)
    syms = ["600001", "600002", "600003"]
    px = pd.DataFrame({s: [10.0] * n for s in syms}, index=idx)
    vol = pd.DataFrame({s: [1e6] * n for s in syms}, index=idx)
    # 成交额：600003 > 600002 > 600001
    amt = pd.DataFrame({"600001": [1e7] * n, "600002": [5e7] * n,
                        "600003": [9e7] * n}, index=idx)
    P = Panel(open=px, high=px, low=px, close=px, volume=vol, amount=amt,
              raw_open=px, raw_close=px, tradable=(vol > 0),
              limits=np.array([0.10] * 3), names={s: "" for s in syms})
    sig_all = pd.DataFrame({s: [1.0] * n for s in syms}, index=idx)

    r = run(P, sig_all, BacktestConfig(cost=NOCOST.cost, max_positions=2),
            with_trades=True)
    held = r.pos.iloc[-1]
    check("持仓数不超过 K", int(r.pos.sum(axis=1).max()) == 2,
          f"最大同时持仓={int(r.pos.sum(axis=1).max())}")
    check("名额不够时优先买流动性最好的",
          bool(held["600003"]) and bool(held["600002"]) and not bool(held["600001"]),
          f"持有={held.to_dict()}")

    r1 = run(P, sig_all, BacktestConfig(cost=NOCOST.cost, max_positions=5))
    # 第 1 根 K 线没有前一日信号，必然空仓：6 天里只有 5 天持有 3 只
    check("K 大于信号数时不强行凑满（3/5 仓位，首日空仓）",
          abs(r1.exposure - (5 / 6) * (3 / 5)) < 1e-9,
          f"资金利用率={r1.exposure:.3f}，应为 0.50")


def main() -> int:
    for fn in [test_execution_timing, test_costs, test_t_plus_1,
               test_limit_up_blocks_entry, test_limit_down_blocks_exit,
               test_suspension, test_stop_loss, test_adjusted_vs_raw,
               test_trade_extraction, test_mask, test_max_positions]:
        try:
            fn()
        except Exception as e:                                  # noqa: BLE001
            check(f"{fn.__name__} 抛异常", False, f"{type(e).__name__}: {e}")

    ok = sum(1 for _, p, _ in CHECKS if p)
    print(f"\n{'='*66}\n回测引擎规则验证\n{'='*66}")
    for name, passed, detail in CHECKS:
        mark = "✅" if passed else "❌"
        print(f"{mark} {name}" + (f"    [{detail}]" if detail and not passed else ""))
    print(f"{'='*66}\n{ok}/{len(CHECKS)} 通过\n")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
