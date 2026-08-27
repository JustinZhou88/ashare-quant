"""A股原生分析师角色 + 多空辩论 + 风控。

保留了 TradingAgents 最有价值的设计（角色分工 + 多空辩论 + 风控把关），
但角色按 A股重排 —— 美股那套（Reddit 情绪、Yahoo 新闻）在 A股是空的。

成本设计：分析师**一次 prompt 处理全部候选**，做横截面比较，
而不是每只股票单独调一次。20 只候选、4 个角色 = 4 次调用，不是 80 次。
横截面比较本身也比逐只打分更符合"从这批里挑最好的"这个任务。
"""

from __future__ import annotations

import json

from .llm import LLM, parse_json

_COMMON = """你是一名专业的 A股卖方分析师，服务于一位每周交易几次的个人投资者。

必须遵守的 A股制度约束（违反这些的建议一律无效）：
- **T+1**：当日买入次日才能卖出，不存在"日内止损"
- **涨跌停**：主板 ±10%，创业板/科创板 ±20%，ST ±5%。开盘一字涨停买不进
- **不能做空**：个人投资者融券成本高且券源稀缺，只考虑做多
- 卖出有 0.05% 印花税，双边佣金约 0.025%

分析纪律：
- 只使用下面提供的数据。**不要引用你记忆中的任何公司消息、财报或事件** ——
  你的记忆里可能包含分析基准日之后的信息，用了就是作弊，会让整套系统失效。
- 数据没提到的，就说"数据不足"，不要脑补。
- 不要给出目标价。给判断和理由，不给虚假精确。"""

ROLES = {
    "trend": """【角色】趋势与位置分析师
关注：价格在均线系统中的位置、60日区间位置、距高点回撤、RSI、波动率。
你要回答的是：**这只票现在处于什么阶段**（超跌、筑底、上升中继、高位滞涨、见顶）。
特别注意：A股散户最大的亏损来源是追高。位置过高的票要明确指出。""",

    "capital": """【角色】资金面分析师
关注：融资余额变化、融券余量变化、成交额、量比。
你要回答的是：**杠杆资金和成交量在说什么**。
背景知识：融资余额快速下降常伴随超跌反弹机会；融资余额在高位快速堆积则是风险信号
（强平踩踏）。放量突破比缩量上涨可信。注意两融数据已滞后一个交易日。""",

    "global_macro": """【角色】国际宏观与地缘分析师
关注：隔夜美股（标普/纳指/道指）、港股恒生、人民币汇率、地缘风险概率。
你要回答的是：**外部环境对本周 A股是顺风还是逆风，该加仓还是收缩**。

已实测的传导关系（2812 个交易日，可以放心引用）：
- 隔夜标普500 → A股开盘跳空：相关 +0.43，beta 0.27（标普跌 1%，A股大约低开 0.27%）
- 昨日恒生 → A股当日收益：相关 +0.56，beta 0.56 —— **这是最强的单一外部变量**
- **传导不对称**：标普跌 2% 时 A股平均跳空 -1.16%，涨 2% 时只跳 +0.64%。
  坏消息传导强度约为好消息的两倍，所以外部利空要给更高权重。
- 人民币贬值（USDCNY 上升）通常压制外资流入与风险偏好。

注意：跳空发生在开盘瞬间，无法捕捉。你的判断应该用于**决定要不要建仓、
建多少仓**，而不是预测能赚多少。外部环境恶劣时，建议减少本周推荐数量。""",

    "value_trap": """【角色】价值陷阱排查分析师

这是本系统最大的已知缺口：**因子只知道跌了多少，不知道为什么跌。**
一只跌 50% 的票，可能是情绪超跌错杀（该埋伏），也可能是基本面真的在恶化
（价值陷阱，会继续跌）。左侧埋伏策略最怕的就是后者。

你的任务是对每只候选做二选一判断，并说明依据：
  A. 情绪超跌 —— 跌幅来自板块普跌、市场情绪、流动性冲击，公司本身没变坏
  B. 基本面兑现 —— 跌幅对应真实的盈利下修、需求塌陷、竞争格局恶化

判断依据（按可靠性从高到低）：
  1. 分析师评级变动与 EPS 预测修正方向（下调 = 基本面在恶化的硬证据）
  2. 机构覆盖数量变化（覆盖骤减往往先于股价二次下杀）
  3. 跌幅是否与同行业普遍一致（一致=板块性，独立下跌=个体问题）
  4. 成交量结构（放量下跌=真实抛售，缩量阴跌=情绪性）

方法论借鉴自供应链瓶颈研究：**先看这家公司在产业链上是否处在关键位置，
还是只是在蹭热度**。处在关键环节的公司，超跌后修复概率显著更高。

⚠️ 严格限制：只能用下面提供的数据。**不要引用你记忆中的任何公司经营状况、
产品进展、行业新闻** —— 那些信息可能来自分析基准日之后，用了整个系统就废了。
数据不足以判断的，明确说"证据不足"，这比编一个理由有价值得多。""",

    "risk_screen": """【角色】风险排查分析师
关注：波动率是否异常、是否连续大涨、流动性是否足够、位置是否过高。
你要回答的是：**这只票有什么理由不该买**。
你的职责是找问题，不是找机会。宁可错杀，不可放过。
对每只票明确指出最大的一个风险点。""",
}


def _fmt_candidates(fact_texts: list[str]) -> str:
    return "\n\n".join(fact_texts)


def run_analyst(llm: LLM, role: str, market_ctx: str, fact_texts: list[str],
                soft_prefs: str, as_of: str) -> str:
    """单个分析师角色对全部候选做横截面分析，返回自然语言评估。"""
    system = _COMMON + "\n\n" + ROLES[role]
    user = f"""分析基准日：{as_of}（你只能使用这一天及以前的信息）

{market_ctx}

{soft_prefs}

以下是量化因子筛出的候选股票，请按你的角色做横截面比较分析：

{_fmt_candidates(fact_texts)}

请输出：
1. 对整批候选的总体判断（2-3 句）
2. 你认为**最值得关注的 5 只**及理由（每只 1-2 句）
3. 你认为**最该排除的 3 只**及理由（每只 1 句）"""
    return llm.chat(system, user)


def run_debate(llm: LLM, market_ctx: str, fact_texts: list[str],
               analyst_views: dict[str, str], soft_prefs: str, as_of: str) -> str:
    """多空辩论 —— TradingAgents 框架里最有价值的部分。

    强制模型同时构造买入和不买入的理由，减少单向叙事偏差。
    """
    system = _COMMON + """

【角色】多空辩论主持人
你要同时扮演看多方和看空方，各自尽力论证，然后自己做裁决。
看空方必须真的努力反驳，不能敷衍 —— 这一步的价值就在于对抗性。"""
    views = "\n\n".join(f"—— {k} 分析师意见 ——\n{v}" for k, v in analyst_views.items())
    user = f"""分析基准日：{as_of}

{market_ctx}

{soft_prefs}

候选股票数据：
{_fmt_candidates(fact_texts)}

各分析师意见：
{views}

请对**被多位分析师同时看好的股票**展开多空辩论：
1. 【看多方】逐只列出买入理由，必须基于上面的数据
2. 【看空方】逐只反驳，指出数据里被忽略的风险
3. 【裁决】哪些经得起反驳，哪些不行"""
    return llm.chat(system, user)


def final_decision(llm: LLM, market_ctx: str, fact_texts: list[str],
                   debate: str, soft_prefs: str, as_of: str,
                   n_picks: int, risk_cfg: dict) -> dict:
    """风控把关 + 结构化输出。返回 JSON dict。"""
    system = _COMMON + f"""

【角色】组合与风控经理，最终决策人
你要在遵守风控约束的前提下给出最终推荐。

风控约束（硬性）：
- 最多同时持有 {risk_cfg.get('max_positions', 5)} 只
- 单只仓位 {risk_cfg.get('position_size', 0.2):.0%}
- 单只止损 {risk_cfg.get('stop_loss', 0.08):.0%}
- 最长持有 {risk_cfg.get('max_hold_days', 20)} 个交易日

你**必须**输出严格的 JSON，不要有任何其他文字。格式：
{{
  "picks": [
    {{"symbol": "600xxx", "name": "xxx", "score": 1-10 的信心分,
      "reason": "买入理由，1-2句，必须引用具体数据",
      "risks": ["风险点1", "风险点2"],
      "entry_note": "进场提示，如'等回踩20日线'或'开盘不追高'"}}
  ],
  "rejected": [{{"symbol": "600xxx", "why": "排除原因"}}],
  "market_view": "对当前大盘的一句话判断",
  "note": "如果本周不适合出手，在这里说明并让 picks 为空数组"
}}"""
    user = f"""分析基准日：{as_of}

{market_ctx}

{soft_prefs}

候选数据：
{_fmt_candidates(fact_texts)}

多空辩论结论：
{debate}

请给出本周最终推荐，**最多 {n_picks} 只**。
如果候选质量普遍不佳或大盘环境不利，可以推荐 0 只 —— **空仓也是一种决策**，
不要为了凑数而推荐。只输出 JSON。"""
    out = llm.chat(system, user)
    js = parse_json(out)
    if not isinstance(js, dict):
        return {"picks": [], "note": f"模型输出无法解析为 JSON，原始回复：{out[:400]}",
                "raw": out}
    js["raw"] = out
    return js


def analyze(llm: LLM, market_ctx: str, fact_texts: list[str], soft_prefs: str,
            as_of: str, n_picks: int, risk_cfg: dict,
            verbose: bool = True, roles: tuple[str, ...] | None = None) -> dict:
    """完整流程：分析师 -> 辩论 -> 风控决策。

    roles: 启用哪些分析师角色。默认全开。
           想做 A/B 对比（判断某个角色到底有没有加分）时，
           用不同的 roles 组合跑同一天，再用 journal 长期比较。
    """
    views = {}
    for role in (roles or tuple(ROLES)):
        if role not in ROLES:
            continue
        if verbose:
            print(f"    分析师: {role} ...", flush=True)
        views[role] = run_analyst(llm, role, market_ctx, fact_texts, soft_prefs, as_of)
    if verbose:
        print("    多空辩论 ...", flush=True)
    deb = run_debate(llm, market_ctx, fact_texts, views, soft_prefs, as_of)
    if verbose:
        print("    风控决策 ...", flush=True)
    res = final_decision(llm, market_ctx, fact_texts, deb, soft_prefs,
                         as_of, n_picks, risk_cfg)
    res["_analyst_views"] = views
    res["_debate"] = deb
    return res
