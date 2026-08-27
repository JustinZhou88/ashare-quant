"""规则引擎 —— 把 rules.yaml 里的交易心得变成可执行的过滤器。

设计原则：**硬约束在 LLM 之前执行**。
两个理由：
  1. 省钱 —— 被硬约束刷掉的票不用花 token
  2. 防止被说服 —— LLM 很擅长为任何标的编出买入理由。
     "不碰 ST" 这种铁律必须是代码级的，不能是 prompt 里的一句话。

软偏好才交给 LLM，因为它们本来就需要判断力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DEFAULT_RULES = "rules.yaml"


@dataclass
class Rules:
    hard: dict = field(default_factory=dict)
    soft: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    schedule: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES) -> "Rules":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"找不到规则文件 {p}，先建一个（见项目根目录模板）")
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(hard=d.get("hard_filters") or {}, soft=d.get("soft_prefs") or {},
                   risk=d.get("risk") or {}, schedule=d.get("schedule") or {}, raw=d)

    def get(self, section: str, key: str, default=None):
        return (getattr(self, section, {}) or {}).get(key, default)


# ---------------------------------------------------------------- 硬过滤
def apply_hard_filters(panel, rules: Rules, date: pd.Timestamp,
                       meta: pd.DataFrame | None = None,
                       verbose: bool = False) -> tuple[list[str], dict[str, int]]:
    """在 `date` 这天，返回通过全部硬约束的股票列表 + 各条约束刷掉了多少只。

    只使用 date 及以前的数据 —— 无未来函数。
    meta: 可选的股票静态信息（市值、行业），来自 loader.fetch_all_symbols()。
    """
    H = rules.hard
    syms = list(panel.symbols)
    reasons: dict[str, int] = {}
    alive = pd.Series(True, index=syms)

    hist = panel.close.loc[:date]
    if hist.empty:
        return [], {"无数据": len(syms)}

    def cut(name: str, ok: pd.Series) -> None:
        nonlocal alive
        ok = ok.reindex(syms).fillna(False)
        killed = int((alive & ~ok).sum())
        if killed:
            reasons[name] = killed
        alive = alive & ok

    # --- ST / 退市 ---
    if H.get("exclude_st", True):
        names = pd.Series({s: panel.names.get(s, "") for s in syms})
        cut("ST/退市", ~names.str.upper().str.contains("ST|退", na=False))

    # --- 板块 ---
    for pref in (H.get("exclude_boards") or []):
        cut(f"板块{pref}", ~pd.Series({s: s.startswith(str(pref)) for s in syms}))

    # --- 上市时长（次新股）---
    n_new = int(H.get("exclude_new_days") or 0)
    if n_new > 0:
        age = panel.close.loc[:date].notna().sum()
        cut(f"上市不足{n_new}日", age >= n_new)

    # --- 停牌 ---
    cut("当日停牌", panel.tradable.loc[:date].iloc[-1])

    # --- 流动性 ---
    min_amt = H.get("min_amount_20d")
    if min_amt:
        amt20 = panel.amount.loc[:date].iloc[-20:].mean()
        cut(f"成交额<{float(min_amt)/1e8:.1f}亿", amt20 >= float(min_amt))

    # --- 市值（用今日快照 × 价格变动近似历史市值）---
    if meta is not None and not meta.empty and ("min_mcap" in H or "max_mcap" in H):
        m = meta.set_index("symbol")["mcap"].reindex(syms)
        last_px = panel.raw_close.ffill().iloc[-1].reindex(syms)
        px_at = panel.raw_close.loc[:date].ffill().iloc[-1].reindex(syms)
        approx = m * (px_at / last_px)          # 股本不变假设，够用
        if H.get("min_mcap"):
            cut(f"市值<{float(H['min_mcap'])/1e8:.0f}亿", approx >= float(H["min_mcap"]))
        if H.get("max_mcap"):
            cut("市值超上限", approx <= float(H["max_mcap"]))

    # --- 波动率 ---
    ret = panel.close.loc[:date].pct_change()
    vol60 = ret.iloc[-60:].std() * np.sqrt(244)
    if H.get("min_vol_60d"):
        cut(f"波动率<{float(H['min_vol_60d']):.0%}", vol60 >= float(H["min_vol_60d"]))
    if H.get("max_vol_60d"):
        cut(f"波动率>{float(H['max_vol_60d']):.0%}", vol60 <= float(H["max_vol_60d"]))

    # --- 连续涨停（追高保护）---
    max_lu = H.get("max_consecutive_limit_up")
    if max_lu is not None:
        rc = panel.raw_close.loc[:date]
        pct = rc.pct_change().iloc[-10:]
        lim = pd.Series(panel.limits, index=syms)
        near_limit = pct.ge(lim - 0.005, axis=1)
        # 末尾连续涨停天数
        streak = near_limit.iloc[::-1].cummin().sum()
        cut(f"连续涨停>{max_lu}", streak <= int(max_lu))

    # --- 买得起吗 ---
    # A股 1 手 = 100 股。单只预算买不起 1 手的票，**永远不可能成交** ——
    # 让它进候选池只会白花 LLM 的 token，还可能推荐一只你买不了的股票。
    # 这个门槛不是主观参数，是由本金和仓位比例算出来的硬约束。
    risk = rules.risk or {}
    cash = float(risk.get("init_cash") or 100_000)
    pos = float(risk.get("position_size") or 0.2)
    budget = cash * pos
    max_px = budget / 100.0
    if max_px > 0:
        px_now = panel.raw_close.loc[:date].ffill().iloc[-1].reindex(syms)
        cut(f"买不起1手(>{max_px:.0f}元)", px_now <= max_px)

    # --- 行业 / 个股黑名单 ---
    bl = set(str(x).zfill(6) for x in (H.get("exclude_symbols") or []))
    if bl:
        cut("个股黑名单", pd.Series({s: s not in bl for s in syms}))
    ind_bl = set(H.get("exclude_industries") or [])
    if ind_bl and meta is not None and "industry" in (meta.columns if meta is not None else []):
        im = meta.set_index("symbol")["industry"].reindex(syms)
        cut("行业黑名单", ~im.isin(ind_bl))

    passed = [s for s in syms if bool(alive.get(s, False))]
    if verbose:
        print(f"  硬过滤: {len(syms)} -> {len(passed)}")
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    -{v:4d}  {k}")
    return passed, reasons


def soft_prefs_text(rules: Rules) -> str:
    """把软偏好渲染成给 LLM 看的中文段落。"""
    S = rules.soft or {}
    parts = []
    if S.get("likes"):
        parts.append("【用户偏好的形态】\n" + "\n".join(f"- {x}" for x in S["likes"]))
    if S.get("avoids"):
        parts.append("【用户明确回避的情况】\n" + "\n".join(f"- {x}" for x in S["avoids"]))
    if S.get("fast_money_definition"):
        parts.append("【用户对「来钱快」的定义】\n" +
                     "\n".join(f"- {x}" for x in S["fast_money_definition"]))
    if not parts:
        return "（用户尚未填写个人偏好，仅按通用 A股逻辑判断。）"
    return "\n\n".join(parts)
