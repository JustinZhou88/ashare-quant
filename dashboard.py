"""生成系统总览面板（自包含 HTML，双击即开，不需要服务器）。

    python dashboard.py            # 生成并自动用浏览器打开
    python dashboard.py --no-open  # 只生成文件

为什么不做成 Web 服务：多一个常驻进程就多一个会挂的东西，
而这个面板的内容只在每周扫描后变一次。生成静态文件最省心，
也方便你直接发给别人看。

面板内容全部来自本地文件，不联网：
    journal/recommendations.csv    记账
    journal/scan_*.json            最近一次扫描的完整结果
    data_cache/                    数据新鲜度
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

OUT = Path("dashboard.html")


# ---------------------------------------------------------------- 采集状态
def data_status() -> list[dict]:
    rows = []
    px = [f for f in glob.glob("data_cache/*.csv")
          if os.path.basename(f)[:6].isdigit()]
    latest = ""
    if px:
        try:
            d = pd.read_csv(px[0], parse_dates=["date"])
            latest = str(d["date"].max().date())
        except Exception:                                      # noqa: BLE001
            pass
    rows.append({"名称": "行情日线", "数量": f"{len(px)} 只", "最新": latest,
                 "路径": "data_cache/"})
    rows.append({"名称": "融资融券", "数量": f"{len(glob.glob('data_cache/alt/margin_*.csv'))} 只",
                 "最新": "", "路径": "data_cache/alt/"})
    for f, name in [("data_cache/alt/reports.csv", "券商研报"),
                    ("data_cache/alt/holders.csv", "股东户数")]:
        if os.path.exists(f):
            try:
                d = pd.read_csv(f)
                col = "publish_date" if "publish_date" in d else "ann_date"
                lt = str(pd.to_datetime(d[col]).max().date()) if col in d else ""
                rows.append({"名称": name, "数量": f"{len(d):,} 条", "最新": lt, "路径": f})
            except Exception:                                  # noqa: BLE001
                pass
    for src in ["x_influencer", "guba", "mediacrawler"]:
        p = Path("sentiment_feed") / src
        n = len(list(p.glob("*"))) if p.exists() else 0
        rows.append({"名称": f"情绪源 · {src}", "数量": f"{n} 个文件" if n else "无数据",
                     "最新": "", "路径": str(p)})
    return rows


def key_status() -> list[dict]:
    ks = [("GEMINI_API_KEY", "Gemini", "免费 1500 次/天，够用 250 倍"),
          ("XAI_API_KEY", "Grok", "可实时检索 X，替代爬虫"),
          ("DEEPSEEK_API_KEY", "DeepSeek", "约 ¥0.15/次扫描"),
          ("ZHIPU_API_KEY", "智谱", "glm-4-flash 免费")]
    return [{"env": e, "名称": n, "状态": "已设置" if os.environ.get(e) else "未设置",
             "说明": d} for e, n, d in ks]


def latest_scan() -> tuple[str, dict]:
    fs = sorted(glob.glob("journal/scan_*.json"))
    if not fs:
        return "", {}
    f = fs[-1]
    try:
        return os.path.basename(f)[5:-5], json.load(open(f, encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return "", {}


def journal_stats() -> tuple[pd.DataFrame, dict]:
    f = "journal/recommendations.csv"
    if not os.path.exists(f):
        return pd.DataFrame(), {}
    j = pd.read_csv(f, dtype={"symbol": str})
    real = j[~j["dry_run"].astype(bool)] if "dry_run" in j else j
    st = {}
    for h in (5, 10, 20):
        c = f"ret_{h}d"
        if c in real:
            s = real[c].dropna()
            if len(s):
                e = real.get(f"excess_{h}d", pd.Series(dtype=float)).dropna()
                st[h] = {"n": len(s), "avg": s.mean(), "win": (s > 0).mean(),
                         "exc": e.mean() if len(e) else None}
    return real, st


# ---------------------------------------------------------------- 渲染
def esc(x) -> str:
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def pct(x, nd=2) -> str:
    try:
        return f"{float(x):+.{nd}%}"
    except Exception:                                          # noqa: BLE001
        return "—"


CSS = """
:root{--paper:#f6f7f9;--surface:#fff;--surface-2:#eef1f5;--text:#161b23;--text-2:#4d5765;
--text-3:#7a8492;--line:#dce0e7;--line-2:#c5cbd4;--brass:#9a6c1c;--brass-soft:#f2e6cd;
--up:#c23a3a;--down:#1f8a5f;--ok:#1f8a5f;--off:#a0a8b4;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
--cjk:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
@media (prefers-color-scheme:dark){:root{--paper:#0e1218;--surface:#161b23;--surface-2:#1e242e;
--text:#e6eaf0;--text-2:#a5afbd;--text-3:#78828f;--line:#242b35;--line-2:#333c48;
--brass:#d6a24e;--brass-soft:#33280f;--up:#e05a5a;--down:#3aab7d;--ok:#3aab7d;--off:#5c6672}}
:root[data-theme=dark]{--paper:#0e1218;--surface:#161b23;--surface-2:#1e242e;--text:#e6eaf0;
--text-2:#a5afbd;--text-3:#78828f;--line:#242b35;--line-2:#333c48;--brass:#d6a24e;
--brass-soft:#33280f;--up:#e05a5a;--down:#3aab7d;--ok:#3aab7d;--off:#5c6672}
:root[data-theme=light]{--paper:#f6f7f9;--surface:#fff;--surface-2:#eef1f5;--text:#161b23;
--text-2:#4d5765;--text-3:#7a8492;--line:#dce0e7;--line-2:#c5cbd4;--brass:#9a6c1c;
--brass-soft:#f2e6cd;--up:#c23a3a;--down:#1f8a5f;--ok:#1f8a5f;--off:#a0a8b4}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--text);font-family:var(--cjk);line-height:1.7}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 80px}
h1{font-size:30px;margin:0;letter-spacing:-.01em}
h2{font-size:18px;margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line-2)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
color:var(--brass);margin-bottom:10px}
.sub{color:var(--text-2);font-size:14px;margin-top:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px 18px}
.card .l{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
color:var(--text-3);margin-bottom:6px}
.card .v{font-family:var(--mono);font-size:26px;font-weight:600;font-variant-numeric:tabular-nums;
line-height:1.15}
.card .d{font-size:12.5px;color:var(--text-2);margin-top:4px}
.scroller{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;width:100%;min-width:520px;font-size:13.5px}
th,td{text-align:left;padding:8px 14px 8px 0;border-bottom:1px solid var(--line)}
th{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;
color:var(--text-3);font-weight:500;white-space:nowrap}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;
white-space:nowrap;padding-right:0}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:3px;font-weight:600}
.pill.on{background:var(--brass-soft);color:var(--brass)}
.pill.off{background:var(--surface-2);color:var(--off)}
.up{color:var(--up)}.down{color:var(--down)}
.pick{background:var(--surface);border:1px solid var(--line-2);border-radius:6px;
padding:20px;margin:12px 0}
.pick-h{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}
.pick-h .c{font-family:var(--mono);font-size:18px;font-weight:600}
.pick-h .nm{font-size:17px;font-weight:650}
.pick-h .px{margin-left:auto;font-family:var(--mono);color:var(--text-2)}
.pick p{margin:6px 0;font-size:14px;color:var(--text-2)}
.pick b{color:var(--text)}
.note{background:var(--surface-2);border-left:2px solid var(--brass);padding:11px 15px;
font-size:13.5px;color:var(--text-2);margin:12px 0}
.todo{display:grid;grid-template-columns:auto 1fr;gap:10px 14px;font-size:14px;align-items:start}
.badge{font-family:var(--mono);font-size:10px;padding:3px 8px;border-radius:2px;
white-space:nowrap;font-weight:600;margin-top:3px}
.badge.you{background:var(--brass-soft);color:var(--brass)}
.badge.time{background:var(--surface-2);color:var(--text-3);border:1px dashed var(--line-2)}
code{font-family:var(--mono);font-size:.86em;background:var(--surface-2);padding:1px 5px;
border-radius:3px}
.foot{margin-top:40px;padding-top:18px;border-top:1px solid var(--line);
font-size:12.5px;color:var(--text-3)}
@media(max-width:640px){.pick-h .px{margin-left:0;width:100%}}
"""


def render() -> str:
    scan_date, scan = latest_scan()
    jr, st = journal_stats()
    ds = data_status()
    ks = key_status()
    picks = scan.get("picks") or []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    P = [f"<title>A股量化系统 · 总览</title><style>{CSS}</style>",
         '<div class="wrap">',
         '<div class="eyebrow">系统总览</div><h1>A股量化选股</h1>',
         f'<p class="sub">生成于 {now} · 全部数据来自本地，不联网</p>']

    # 概览卡片
    n_px = sum(1 for r in ds if r["名称"] == "行情日线")
    px_row = next((r for r in ds if r["名称"] == "行情日线"), {})
    n_j = len(jr)
    mature = st.get(20, {}).get("n", 0)
    P.append('<h2>关键状态</h2><div class="grid">')
    P.append(f'<div class="card"><div class="l">股票池</div><div class="v">{px_row.get("数量","—")}</div>'
             f'<div class="d">数据截至 {px_row.get("最新","—")}</div></div>')
    P.append(f'<div class="card"><div class="l">累计推荐</div><div class="v">{n_j}</div>'
             f'<div class="d">已到期 20 日样本 {mature} 个</div></div>')
    P.append('<div class="card"><div class="l">因子层样本外</div>'
             '<div class="v">0.89</div><div class="d">夏普 · 年化 +23.2% · 回撤 −28.5%</div></div>')
    P.append('<div class="card"><div class="l">LLM 层证据</div>'
             f'<div class="v" style="color:var(--off)">{mature}/30</div>'
             '<div class="d">达到 30 个样本才有初步结论</div></div>')
    P.append('</div>')

    # 最近推荐
    P.append(f'<h2>最近一次推荐 · {esc(scan_date) or "无"}</h2>')
    if not picks:
        P.append('<p class="sub">还没有扫描记录。运行 <code>python scan_weekly.py '
                 '--universe universe_full.txt</code></p>')
    for p in picks:
        sym = str(p.get("symbol", "")).zfill(6)
        fc = p.get("forecast") or {}
        P.append('<div class="pick"><div class="pick-h">'
                 f'<span class="c">{esc(sym)}</span>'
                 f'<span class="nm">{esc(p.get("name",""))}</span>'
                 f'<span class="pill on">信心 {esc(p.get("score","—"))}/10</span>')
        if fc.get("price_at_scan"):
            P.append(f'<span class="px">{fc["price_at_scan"]:.2f} 元</span>')
        P.append('</div>')
        P.append(f'<p><b>理由</b> {esc(p.get("reason",""))}</p>')
        if p.get("risks"):
            P.append(f'<p><b>风险</b> {esc("；".join(p["risks"]))}</p>')
        if p.get("entry_note"):
            P.append(f'<p><b>进场</b> {esc(p["entry_note"])}</p>')
        if fc.get("enough"):
            q = fc["peak_price_q"]; m = fc["mfe_q"]
            P.append(f'<div class="note">历史类比 {fc["n_analogs"]:,} 个样本 · '
                     f'预期最高价中位 <b>{q["p50"]:.2f}</b>（{pct(m["p50"],1)}），'
                     f'见顶中位第 {fc["days_to_peak_median"]:.0f} 日 · '
                     f'摸到 +15% {fc["p_hit_15"]:.1%} · '
                     f'触发止损 {fc["p_stop_15"]:.1%} · '
                     f'期望收益 {pct(fc["expected_ret"])}</div>')
        P.append('</div>')
    if scan.get("market_view"):
        P.append(f'<div class="note"><b>大盘观点</b>　{esc(scan["market_view"])}</div>')

    # 战绩
    P.append('<h2>累计战绩</h2>')
    if not st:
        P.append('<p class="sub">还没有已到期的样本。每周扫描后运行 '
                 '<code>python scan_weekly.py --journal</code> 更新。</p>')
    else:
        P.append('<div class="scroller"><table><thead><tr><th>持有</th><th class="n">样本</th>'
                 '<th class="n">平均收益</th><th class="n">胜率</th>'
                 '<th class="n">超额</th></tr></thead><tbody>')
        for h, v in sorted(st.items()):
            cls = "up" if v["avg"] > 0 else "down"
            ex = pct(v["exc"]) if v["exc"] is not None else "—"
            P.append(f'<tr><td>{h} 日</td><td class="n">{v["n"]}</td>'
                     f'<td class="n {cls}">{pct(v["avg"])}</td>'
                     f'<td class="n">{v["win"]:.0%}</td><td class="n">{ex}</td></tr>')
        P.append('</tbody></table></div>')
        if mature < 30:
            P.append(f'<div class="note">⚠️ 已到期样本只有 {mature} 个，'
                     f'<b>远不足以判断系统是否有效</b>。按每周 2–3 只算，'
                     f'需累计 3 个月才有初步参考价值。现在的胜率请当噪声看。</div>')

    # 数据
    P.append('<h2>数据与接口</h2><div class="scroller"><table><thead><tr>'
             '<th>数据源</th><th class="n">规模</th><th>最新</th><th>位置</th>'
             '</tr></thead><tbody>')
    for r in ds:
        P.append(f'<tr><td>{esc(r["名称"])}</td><td class="n">{esc(r["数量"])}</td>'
                 f'<td>{esc(r["最新"]) or "—"}</td><td><code>{esc(r["路径"])}</code></td></tr>')
    P.append('</tbody></table></div>')

    P.append('<div class="scroller"><table><thead><tr><th>模型 / Key</th><th>状态</th>'
             '<th>说明</th></tr></thead><tbody>')
    for k in ks:
        on = k["状态"] == "已设置"
        P.append(f'<tr><td>{esc(k["名称"])} <code>{esc(k["env"])}</code></td>'
                 f'<td><span class="pill {"on" if on else "off"}">{k["状态"]}</span></td>'
                 f'<td>{esc(k["说明"])}</td></tr>')
    P.append('</tbody></table></div>')

    # 待办
    P.append('<h2>还缺什么</h2><div class="todo">')
    todos = []
    if not any(os.environ.get(k["env"]) for k in ks):
        todos.append(("you", "设置任一 LLM key 并写进 shell 配置",
                      "现在每次都要临时 export。Gemini 免费额度是需求的 250 倍。"))
    if not os.environ.get("XAI_API_KEY"):
        todos.append(("you", "申请 XAI_API_KEY（可选）",
                      "Grok 能官方检索 X，替代爬虫；不需要登录也不会封号。"))
    fx = Path("sentiment_feed/x_influencer")
    if not fx.exists() or not list(fx.glob("*")):
        todos.append(("you", "X 情绪源尚无数据",
                      "两条路：Grok 检索（需 key），或 tools/x_scrape.py（需登录 X）。"))
    todos.append(("time", f"记账积累到 30 个已到期样本（现 {mature} 个）",
                  "LLM 层的有效性无法回测，只能前瞻验证。约需 3 个月。"))
    for kind, t, d in todos:
        lbl = "待你处理" if kind == "you" else "需要时间"
        P.append(f'<span class="badge {kind}">{lbl}</span><div><b>{esc(t)}</b><br>'
                 f'<span style="color:var(--text-2)">{esc(d)}</span></div>')
    P.append('</div>')

    # 命令
    P.append('<h2>每周操作</h2>'
             '<div class="scroller"><table><thead><tr><th>做什么</th><th>命令</th>'
             '</tr></thead><tbody>'
             '<tr><td>更新数据（约 12 分钟）</td><td><code>python update_data.py</code></td></tr>'
             '<tr><td>扫描出推荐</td><td><code>python scan_weekly.py --universe universe_full.txt</code></td></tr>'
             '<tr><td>更新战绩</td><td><code>python scan_weekly.py --journal</code></td></tr>'
             '<tr><td>刷新本面板</td><td><code>python dashboard.py</code></td></tr>'
             '</tbody></table></div>')

    P.append('<p class="foot">仅供研究，不构成投资建议。'
             '因子层有统计支撑（2804 只全市场、6 折滚动前推、多重检验校正）；'
             'LLM 层与外部情绪源目前**没有任何有效性证据**，正在前瞻记账验证中。</p>')
    P.append('</div>')
    return "\n".join(P)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    html = render()
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"面板已生成: {os.path.abspath(a.out)}")
    if not a.no_open:
        webbrowser.open(f"file://{os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
