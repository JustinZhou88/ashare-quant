"""A股交易规则与成本参数。

所有数字都可以改，但请改得有依据 —— 这里的默认值直接决定回测是不是在骗你。
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------- 交易成本
@dataclass(frozen=True)
class CostModel:
    """A股实际交易成本（2023-08-28 印花税减半后的口径）。"""

    commission_rate: float = 2.5e-4   # 佣金 万2.5，双边
    commission_min: float = 5.0       # 单笔最低 5 元
    stamp_tax_rate: float = 5e-4      # 印花税 万5，仅卖出
    transfer_fee_rate: float = 1e-5   # 过户费 万0.1，双边（2022-04 起沪深统一）
    slippage_rate: float = 5e-4       # 滑点，单边万5。小市值股票应调高

    def buy_cost(self, notional: float) -> float:
        """买入总费用（不含滑点，滑点在成交价里体现）。"""
        return max(notional * self.commission_rate, self.commission_min) + \
            notional * self.transfer_fee_rate

    def sell_cost(self, notional: float) -> float:
        """卖出总费用（含印花税）。"""
        return max(notional * self.commission_rate, self.commission_min) + \
            notional * self.transfer_fee_rate + notional * self.stamp_tax_rate


# ---------------------------------------------------------------- 涨跌停
def limit_pct(symbol: str, name: str = "") -> float:
    """返回该股票的涨跌停幅度（0.10 = 10%）。

    规则按板块 + 是否 ST 判定。name 传入股票简称可识别 ST/*ST。
    """
    if "ST" in name.upper():
        return 0.05
    if symbol.startswith(("300", "301")):      # 创业板注册制
        return 0.20
    if symbol.startswith("688"):               # 科创板
        return 0.20
    if symbol.startswith(("8", "4", "920")):   # 北交所
        return 0.30
    return 0.10                                # 沪深主板


# ---------------------------------------------------------------- 回测配置
@dataclass
class BacktestConfig:
    init_cash: float = 100_000.0
    lot_size: int = 100                # 一手 = 100 股
    cost: CostModel = field(default_factory=CostModel)

    # A股结构性约束
    t_plus_1: bool = True              # 当日买入次日才能卖
    block_limit_up_entry: bool = True  # 开盘涨停买不进
    block_limit_down_exit: bool = True # 开盘跌停卖不出
    exec_price: str = "open"           # 信号出现在 t 日收盘，t+1 开盘成交

    # 风控
    stop_loss: float | None = None     # 例如 0.08 = 固定 8% 止损
    take_profit: float | None = None   # 例如 0.15 = 涨 15% 止盈
    max_hold_days: int | None = None   # 最长持有天数，None = 不限
    # 止损/止盈都在**收盘**判定、次日开盘执行。
    # 日线数据上模拟盘中触价成交是自欺欺人 —— 你不知道当天是先触止损还是先触止盈，
    # 按收盘判定虽然保守，但每一笔都是真能成交的。

    # 组合模型：最多同时持有 K 只，每只固定 1/K 仓位，其余为现金。
    # 这样不同选择性的策略之间才可比 —— 否则"一次只买1只"的策略
    # 在 1/N sleeve 模型下仓位只有 1/74，年化被压到 1.5%，夏普却虚高。
    max_positions: int = 10


# ---------------------------------------------------------------- 数据
DATA_DIR = "data_cache"
BENCHMARK = "000300"       # 沪深300
TRADING_DAYS = 244         # A股年均交易日
