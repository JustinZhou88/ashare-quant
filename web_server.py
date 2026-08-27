"""本地面板服务 —— 侧边栏导航 + 可触发任务 + 实时日志。

    python web_server.py          # 打开 http://127.0.0.1:5111
    python web_server.py --port 8080

设计取舍：
  - **只监听 127.0.0.1**，不对外暴露。这台机器上有你的 API key 和交易记录，
    绑 0.0.0.0 等于把它挂到局域网上。
  - 长任务（更新数据、扫描）用子进程跑，日志经 SSE 实时推到页面。
    不用 WebSocket —— SSE 单向推送足够，少一个依赖少一层可能出错的地方。
  - 同一时刻只允许一个任务在跑。两个 scan 并发写 journal 会互相覆盖。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, render_template_string, request

from aq.envfile import ENV_FILE
from aq.envfile import load as load_env
from aq.envfile import write as write_env

load_env()

ROOT = Path(__file__).resolve().parent
# 用**当前解释器**，不写死 .venv 路径 —— 服务器上的虚拟环境位置可能不同，
# 写死会导致子进程用错 Python（表现是任务一起就 ModuleNotFoundError）。
PY = sys.executable
app = Flask(__name__)


def clean(o):
    """把 NaN / Inf / numpy 标量转成 JSON 合法的值。

    **必须对每个 API 出口都过一遍。** 踩过的坑：pandas 的 NaN 经 jsonify
    会序列化成裸 `NaN`，那不是合法 JSON —— 浏览器 JSON.parse 直接抛异常，
    表现是「页面一片空白」，而后端接口返回 200、日志里什么都看不出来。
    """
    import math

    import numpy as _np
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return clean(o.tolist())
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    if o is pd.NaT:
        return None
    return o

# ---------------------------------------------------------------- 鉴权
# 绑 127.0.0.1 时不需要鉴权（只有本机能访问）。
# 一旦绑到对外地址，**必须**有 token —— /api/run 能起子进程，
# /api/settings 能写文件，裸奔等于把服务器交出去。
PANEL_TOKEN = os.environ.get("PANEL_TOKEN", "").strip()
# 只放行登录接口本身。不要把 "/login" 也加进来 —— 没有对应路由，
# 放行后会直接 404；不放行时 _guard 会替它渲染登录页，才是想要的行为。
_PUBLIC = {"/api/login"}


@app.before_request
def _guard():
    if not PANEL_TOKEN:
        return None                       # 未设 token = 仅本机模式
    if request.path in _PUBLIC or request.path.startswith("/static"):
        return None
    if request.cookies.get("panel_token") == PANEL_TOKEN:
        return None
    if request.headers.get("X-Panel-Token") == PANEL_TOKEN:
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "未授权"}), 401
    return render_template_string(LOGIN_PAGE), 401


@app.route("/api/login", methods=["POST"])
def api_login():
    tok = (request.get_json(force=True, silent=True) or {}).get("token", "")
    if PANEL_TOKEN and tok == PANEL_TOKEN:
        r = jsonify({"ok": True})
        # 8 小时有效；HttpOnly 防 XSS 偷 cookie；SameSite 防 CSRF
        r.set_cookie("panel_token", tok, max_age=8 * 3600,
                     httponly=True, samesite="Lax")
        return r
    return jsonify({"ok": False, "msg": "口令不对"}), 401


LOGIN_PAGE = """<!doctype html><meta charset="utf-8"><title>登录</title>
<style>body{font-family:-apple-system,"PingFang SC",sans-serif;background:#0e1218;
color:#e6eaf0;display:grid;place-items:center;height:100vh;margin:0}
.box{background:#161b23;border:1px solid #242b35;border-radius:8px;padding:32px;width:320px}
h1{font-size:17px;margin:0 0 6px}p{color:#7b8593;font-size:13px;margin:0 0 18px}
input{width:100%;padding:9px 11px;border:1px solid #333c48;border-radius:5px;
background:#0e1218;color:#e6eaf0;font-size:14px;box-sizing:border-box}
button{width:100%;margin-top:12px;padding:9px;border:0;border-radius:5px;
background:#d6a24e;color:#0e1218;font-weight:600;font-size:14px;cursor:pointer}
#e{color:#e05a5a;font-size:13px;margin-top:10px;min-height:18px}</style>
<div class="box"><h1>A股量化控制台</h1><p>请输入访问口令</p>
<input id="t" type="password" autofocus placeholder="PANEL_TOKEN">
<button onclick="go()">进入</button><div id="e"></div></div>
<script>
const go=async()=>{const r=await fetch('/api/login',{method:'POST',
 headers:{'Content-Type':'application/json'},
 body:JSON.stringify({token:document.getElementById('t').value})});
 if(r.ok)location.href='/';else document.getElementById('e').textContent='口令不对';};
document.getElementById('t').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
</script>"""


# ---------------------------------------------------------------- 任务管理
_task_lock = threading.Lock()
_task = {"name": None, "proc": None, "started": None, "log": [], "rc": None}
_subs: list[queue.Queue] = []

TASKS = {
    "update": {
        "label": "更新数据",
        "cmd": [PY, "-u", "update_data.py"],
        "desc": "增量拉取行情、指数、国际环境。约 12 分钟。",
    },
    "scan": {
        "label": "运行扫描",
        "cmd": [PY, "-u", "scan_weekly.py", "--universe", "universe_full.txt"],
        "desc": "硬规则 → 因子排序 → 五分析师 → 辩论 → 推荐。需要 LLM key。",
    },
    "scan_nollm": {
        "label": "只跑因子（不调 LLM）",
        "cmd": [PY, "-u", "scan_weekly.py", "--universe", "universe_full.txt", "--no-llm"],
        "desc": "只出因子候选池，不花 token。用来快速看今天factor排名。",
    },
    "journal": {
        "label": "更新战绩",
        "cmd": [PY, "-u", "scan_weekly.py", "--journal"],
        "desc": "给历史推荐补算 5/10/20 日前瞻收益。",
    },
}


def _broadcast(line: str) -> None:
    _task["log"].append(line)
    if len(_task["log"]) > 2000:
        del _task["log"][:500]
    for q in list(_subs):
        try:
            q.put_nowait(line)
        except Exception:                                      # noqa: BLE001
            pass


def _run(key: str) -> None:
    spec = TASKS[key]
    env = dict(os.environ)
    try:
        p = subprocess.Popen(spec["cmd"], cwd=str(ROOT), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    except Exception as e:                                     # noqa: BLE001
        _broadcast(f"[启动失败] {type(e).__name__}: {e}")
        _task["rc"] = -1
        _task["name"] = None
        return
    _task["proc"] = p
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.strip():
            _broadcast(line)
    p.wait()
    _task["rc"] = p.returncode
    _broadcast(f"[完成] 退出码 {p.returncode}")
    _task["name"] = None
    _task["proc"] = None


# ---------------------------------------------------------------- 数据读取
def read_status() -> dict:
    px = [f for f in glob.glob("data_cache/*.csv")
          if os.path.basename(f)[:6].isdigit()]
    latest = ""
    if px:
        try:
            latest = str(pd.read_csv(px[0], parse_dates=["date"])["date"].max().date())
        except Exception:                                      # noqa: BLE001
            pass
    src = {}
    for s in ("x_influencer", "guba", "mediacrawler"):
        p = Path("sentiment_feed") / s
        src[s] = len(list(p.glob("*"))) if p.exists() else 0
    keys = {k: bool(os.environ.get(k)) for k in
            ("GEMINI_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY", "ZHIPU_API_KEY")}
    counts = {}
    for f, name in [("data_cache/alt/reports.csv", "reports"),
                    ("data_cache/alt/holders.csv", "holders")]:
        try:
            counts[name] = len(pd.read_csv(f)) if os.path.exists(f) else 0
        except Exception:                                      # noqa: BLE001
            counts[name] = 0
    return {
        "stocks": len(px), "px_latest": latest,
        "margin": len(glob.glob("data_cache/alt/margin_*.csv")),
        "reports": counts.get("reports", 0), "holders": counts.get("holders", 0),
        "feeds": src, "keys": keys,
        "running": _task["name"], "tasks": {k: v["label"] for k, v in TASKS.items()},
    }


def read_picks() -> dict:
    fs = sorted(glob.glob("journal/scan_*.json"))
    if not fs:
        return {"date": "", "picks": [], "market_view": "", "rejected": []}
    try:
        d = json.load(open(fs[-1], encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {"date": "", "picks": [], "market_view": "", "rejected": []}
    return {"date": os.path.basename(fs[-1])[5:-5], "picks": d.get("picks") or [],
            "market_view": d.get("market_view", ""), "rejected": d.get("rejected") or [],
            "views": {k: v for k, v in (d.get("_analyst_views") or {}).items()},
            "debate": d.get("_debate", "")}


def read_journal() -> dict:
    f = "journal/recommendations.csv"
    if not os.path.exists(f):
        return {"rows": [], "stats": {}, "mature": 0}
    j = pd.read_csv(f, dtype={"symbol": str})
    if "dry_run" in j:
        j = j[~j["dry_run"].astype(bool)]
    stats = {}
    for h in (5, 10, 20):
        c = f"ret_{h}d"
        if c in j:
            s = j[c].dropna()
            if len(s):
                e = j.get(f"excess_{h}d", pd.Series(dtype=float)).dropna()
                stats[h] = {"n": int(len(s)), "avg": float(s.mean()),
                            "win": float((s > 0).mean()),
                            "exc": float(e.mean()) if len(e) else None}
    keep = [c for c in ["scan_date", "symbol", "name", "score", "price_at_scan",
                        "entry_px", "ret_5d", "ret_10d", "ret_20d", "excess_20d"]
            if c in j]
    rows = j[keep].where(pd.notna(j[keep]), None).to_dict("records")
    return {"rows": rows, "stats": stats,
            "mature": stats.get(20, {}).get("n", 0)}


def read_candidates() -> dict:
    fs = sorted(glob.glob("journal/candidates_*.json"))
    if not fs:
        return {"date": "", "facts": []}
    try:
        d = json.load(open(fs[-1], encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {"date": "", "facts": []}
    fa = list((d.get("facts") or {}).values())
    fa.sort(key=lambda x: x.get("factor_rank") or 999)
    return {"date": d.get("as_of", ""), "facts": fa}


# ---------------------------------------------------------------- 路由
@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/status")
def api_status():
    return jsonify(clean(read_status()))


@app.route("/api/picks")
def api_picks():
    return jsonify(clean(read_picks()))


@app.route("/api/journal")
def api_journal():
    return jsonify(clean(read_journal()))


@app.route("/api/candidates")
def api_candidates():
    return jsonify(clean(read_candidates()))


@app.route("/api/factors")
def api_factors():
    """因子研究结论 —— 写死在这里，因为它们是已经跑完的验证结果。"""
    return jsonify({
        "adopted": {"name": "中周期截面反转 30/60/90 等权",
                    "cagr": 0.2321, "sharpe": 0.88, "mdd": -0.3108,
                    "calmar": 0.75, "pos_month": 0.64},
        "bench": {"name": "沪深300", "cagr": -0.0227, "sharpe": -0.04, "mdd": -0.4560},
        "windows": [[20, 0.34], [30, 0.92], [40, 0.60], [60, 0.62],
                    [90, 0.87], [120, 0.52], [180, 0.33], [250, 0.54]],
        "rejected": [
            {"n": "两融反转", "t": 1.60, "why": "增量 alpha 不显著"},
            {"n": "股东户数", "t": 0.62, "why": "小池子 t=2.67 是运气，全市场垮掉"},
            {"n": "波动倾斜", "t": 0.24, "why": "简化模拟好看，正式引擎无增量"},
            {"n": "短周期反转 10/20", "t": None, "why": "加进去夏普 0.88→0.13"},
            {"n": "动量（全部窗口）", "t": -1.47, "why": "夏普 -0.24~-0.62，回撤 -78%~-89%"},
            {"n": "缩短持有周期", "t": None, "why": "基于五月数据的假设，全样本测下来是反的"},
            {"n": "止盈", "t": None, "why": "15% 以上完全平坦，差异在噪声内"},
        ],
    })


_panel_cache = {"panel": None, "idx": None, "t": 0}


def _get_panel():
    """行情面板加载要 1-2 分钟，缓存 10 分钟，别每次请求都重载。"""
    import time as _t
    if _panel_cache["panel"] is not None and _t.time() - _panel_cache["t"] < 600:
        return _panel_cache["panel"], _panel_cache["idx"]
    from aq.config import BENCHMARK
    from aq.data.loader import load_index, load_many
    from aq.data.universe import from_file
    from aq.engine.panel import build_panel
    uni = "universe_full.txt" if Path("universe_full.txt").exists() else "universe.txt"
    pn = build_panel(load_many(from_file(uni), verbose=False))
    ix = load_index(BENCHMARK)
    _panel_cache.update({"panel": pn, "idx": ix, "t": _t.time()})
    return pn, ix


@app.route("/api/paper")
def api_paper():
    """模拟盘 —— 把历史推荐按规则回放成一个虚拟组合。"""
    import yaml

    from aq.paper import pending_actions, simulate, summarize, to_dict
    try:
        pn, ix = _get_panel()
        R = (yaml.safe_load(Path("rules.yaml").read_text(encoding="utf-8")) or {}).get("risk") or {}
        init = float(request.args.get("cash", 100000))
        res = simulate(pn, R, init_cash=init)
        d = to_dict(res, pn)
        d["summary"] = summarize(res, pn, ix["close"])
        d["rules"] = {k: R.get(k) for k in
                      ("position_size", "stop_loss", "take_profit",
                       "max_hold_days", "max_positions")}
        d["actions"] = pending_actions(res, pn, R)
        d["blocked"] = res.blocked[-20:]
        return jsonify(clean(d))
    except Exception as e:                                     # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500


@app.route("/api/settings")
def api_settings():
    """当前配置。key 只回传是否设置和尾 4 位，**永不回传明文**。"""
    import yaml
    raw = Path("rules.yaml").read_text(encoding="utf-8") if Path("rules.yaml").exists() else ""
    try:
        y = yaml.safe_load(raw) or {}
    except Exception:                                          # noqa: BLE001
        y = {}
    keys = {}
    for k in ("GEMINI_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY", "ZHIPU_API_KEY"):
        v = os.environ.get(k) or ""
        keys[k] = {"set": bool(v), "tail": ("…" + v[-4:]) if len(v) > 4 else ""}
    return jsonify(clean({"raw": raw, "hard": y.get("hard_filters") or {},
                    "soft": y.get("soft_prefs") or {}, "risk": y.get("risk") or {},
                    "schedule": y.get("schedule") or {}, "keys": keys,
                    "env_path": str(ENV_FILE)}))


@app.route("/api/settings", methods=["POST"])
def api_settings_save():
    """保存配置。rules.yaml 改动前自动备份，改坏了能回滚。"""
    import shutil

    import yaml
    body = request.get_json(force=True) or {}

    if body.get("keys"):
        upd = {k: str(v).strip() for k, v in body["keys"].items()
               if isinstance(v, str) and v.strip()}
        if upd:
            write_env(upd)

    if body.get("raw") is not None:
        txt = str(body["raw"])
        try:
            yaml.safe_load(txt)            # 先验证再写，别把坏 YAML 落盘
        except Exception as e:             # noqa: BLE001
            return jsonify({"ok": False, "msg": f"YAML 语法错误：{str(e)[:160]}"}), 400
        p = Path("rules.yaml")
        if p.exists():
            shutil.copy2(p, p.with_suffix(".yaml.bak"))
        p.write_text(txt, encoding="utf-8")
        return jsonify({"ok": True, "msg": "已保存（旧版本备份为 rules.yaml.bak）"})

    fields = body.get("fields") or {}
    if fields:
        p = Path("rules.yaml")
        try:
            y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as e:             # noqa: BLE001
            return jsonify({"ok": False, "msg": f"现有 rules.yaml 无法解析：{e}"}), 400
        for path, val in fields.items():
            sec, _, k = path.partition(".")
            if not k:
                continue
            y.setdefault(sec, {})[k] = val
        shutil.copy2(p, p.with_suffix(".yaml.bak"))
        p.write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False,
                                    default_flow_style=False), encoding="utf-8")
        return jsonify({"ok": True, "msg": "已保存"})

    return jsonify({"ok": True, "msg": "无改动"})


@app.route("/api/run/<key>", methods=["POST"])
def api_run(key):
    if key not in TASKS:
        return jsonify({"ok": False, "msg": "未知任务"}), 400
    with _task_lock:
        if _task["name"]:
            return jsonify({"ok": False, "msg": f"已有任务在跑：{_task['name']}"}), 409
        _task.update({"name": key, "log": [], "rc": None, "started": time.time()})
    _broadcast(f"[开始] {TASKS[key]['label']}　{datetime.now():%H:%M:%S}")
    threading.Thread(target=_run, args=(key,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    p = _task.get("proc")
    if p and p.poll() is None:
        p.terminate()
        _broadcast("[已请求终止]")
        return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "没有正在运行的任务"})


@app.route("/api/log")
def api_log():
    """SSE 实时日志。先补发已有内容，再推增量。"""
    def stream():
        q: queue.Queue = queue.Queue()
        for ln in list(_task["log"])[-300:]:
            q.put_nowait(ln)
        _subs.append(q)
        try:
            while True:
                try:
                    ln = q.get(timeout=20)
                    yield f"data: {json.dumps({'line': ln})}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            if q in _subs:
                _subs.remove(q)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股量化 · 控制台</title><style>
:root{--paper:#f5f6f8;--card:#fff;--card2:#eef1f5;--text:#161b23;--t2:#4d5765;--t3:#7a8492;
--line:#e2e6ec;--line2:#c9cfd8;--brass:#9a6c1c;--brass-s:#f4e8d0;--up:#c23a3a;--down:#1f8a5f;
--sidebar:248px;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
--cjk:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif}
@media(prefers-color-scheme:dark){:root{--paper:#0d1117;--card:#151a21;--card2:#1c222b;
--text:#e6eaf0;--t2:#a5afbd;--t3:#78828f;--line:#232a34;--line2:#333c48;--brass:#d6a24e;
--brass-s:#332810;--up:#e05a5a;--down:#3aab7d}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--text);
font-family:var(--cjk);line-height:1.65;font-size:14px}
.app{display:flex;min-height:100vh}
aside{width:var(--sidebar);background:var(--card);border-right:1px solid var(--line);
position:fixed;height:100vh;overflow-y:auto;padding:22px 0;flex-shrink:0}
.brand{padding:0 22px 18px;border-bottom:1px solid var(--line);margin-bottom:14px}
.brand h1{font-size:16px;margin:0;letter-spacing:-.01em}
.brand .v{font-family:var(--mono);font-size:10px;color:var(--t3);letter-spacing:.1em;margin-top:3px}
.ngroup{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;
color:var(--t3);padding:14px 22px 6px}
.nitem{display:flex;align-items:center;gap:9px;padding:8px 22px;cursor:pointer;
color:var(--t2);border-left:2px solid transparent;font-size:13.5px}
.nitem:hover{background:var(--card2);color:var(--text)}
.nitem.on{background:var(--brass-s);color:var(--brass);border-left-color:var(--brass);font-weight:600}
.nitem .dot{width:5px;height:5px;border-radius:50%;background:currentColor;opacity:.55}
main{margin-left:var(--sidebar);flex:1;padding:28px 32px 64px;min-width:0}
.page{display:none}.page.on{display:block}
h2{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
.lede{color:var(--t2);font-size:13.5px;margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 18px}
.card .l{font-family:var(--mono);font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;
color:var(--t3);margin-bottom:6px}
.card .v{font-family:var(--mono);font-size:25px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1}
.card .d{font-size:12px;color:var(--t2);margin-top:4px}
.sec{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:20px;margin-bottom:16px}
.sec h3{font-size:15px;margin:0 0 12px;padding-bottom:9px;border-bottom:1px solid var(--line)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:520px;font-size:13px}
th,td{text-align:left;padding:7px 12px 7px 0;border-bottom:1px solid var(--line)}
th{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;
color:var(--t3);font-weight:500;white-space:nowrap}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.up{color:var(--up)}.down{color:var(--down)}
.pill{font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:3px;font-weight:600}
.pill.on{background:var(--brass-s);color:var(--brass)}
.pill.off{background:var(--card2);color:var(--t3)}
button{font-family:var(--cjk);font-size:13px;padding:8px 16px;border-radius:6px;
border:1px solid var(--line2);background:var(--card);color:var(--text);cursor:pointer}
button:hover:not(:disabled){border-color:var(--brass);color:var(--brass)}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:var(--brass);border-color:var(--brass);color:#fff}
button.primary:hover:not(:disabled){opacity:.88;color:#fff}
button:focus-visible{outline:2px solid var(--brass);outline-offset:2px}
.runs{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.run{border:1px solid var(--line);border-radius:7px;padding:14px}
.run .t{font-weight:600;margin-bottom:4px}
.run .d{font-size:12.5px;color:var(--t2);margin-bottom:10px;min-height:34px}
#console{background:#0b0f14;color:#c8d3e0;font-family:var(--mono);font-size:11.5px;
padding:14px;border-radius:7px;height:380px;overflow-y:auto;white-space:pre-wrap;
line-height:1.5;border:1px solid var(--line2)}
#console .warn{color:#e0a34e}#console .err{color:#e06a6a}#console .ok{color:#5fbf8f}
.pick{border:1px solid var(--line2);border-radius:8px;padding:18px;margin-bottom:12px;background:var(--card)}
.pick-h{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:9px}
.pick-h .c{font-family:var(--mono);font-size:17px;font-weight:600}
.pick-h .nm{font-size:16px;font-weight:650}
.pick-h .px{margin-left:auto;font-family:var(--mono);color:var(--t2)}
.pick p{margin:5px 0;font-size:13.5px;color:var(--t2)}.pick b{color:var(--text)}
.tag{font-family:var(--mono);font-size:10px;padding:2px 7px;border-radius:3px;font-weight:600}
.tag.sell{background:var(--up);color:#fff}
.tag.warn{background:var(--brass-s);color:var(--brass)}
.tag.hold{background:var(--card2);color:var(--t3)}
tr.act-sell td{background:color-mix(in srgb,var(--up) 8%,transparent)}
tr.act-warn td{background:color-mix(in srgb,var(--brass) 7%,transparent)}
.note{background:var(--card2);border-left:2px solid var(--brass);padding:10px 14px;
font-size:12.5px;color:var(--t2);margin:10px 0}
.bar{height:5px;background:var(--card2);border-radius:3px;overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%;background:var(--brass)}
details{margin-top:10px}summary{cursor:pointer;font-size:13px;color:var(--brass)}
details pre{white-space:pre-wrap;font-family:var(--cjk);font-size:13px;color:var(--t2);
background:var(--card2);padding:12px;border-radius:6px;margin-top:8px;max-height:340px;overflow-y:auto}
.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.form.col{grid-template-columns:1fr}
.fld{display:flex;flex-direction:column;gap:4px}
.fld label{font-size:12.5px;font-weight:600}
.fld input,.fld textarea{font-family:var(--cjk);font-size:13px;padding:7px 10px;
border:1px solid var(--line2);border-radius:5px;background:var(--paper);color:var(--text)}
.fld input:focus,.fld textarea:focus,#raw:focus{outline:2px solid var(--brass);outline-offset:0;border-color:var(--brass)}
.fld .hint{font-size:11.5px;color:var(--t3)}
#raw{border:1px solid var(--line2);border-radius:6px;padding:12px;background:var(--paper);
color:var(--text);resize:vertical}
@media(max-width:820px){aside{position:static;width:100%;height:auto}
.app{flex-direction:column}main{margin-left:0;padding:20px}}
</style></head><body>
<div class="app">
<aside>
  <div class="brand"><h1>A股量化选股</h1><div class="v">LOCAL CONSOLE</div></div>
  <div class="ngroup">驾驶舱</div>
  <div class="nitem on" data-p="overview"><span class="dot"></span>总览</div>
  <div class="nitem" data-p="picks"><span class="dot"></span>本周推荐</div>
  <div class="ngroup">研究</div>
  <div class="nitem" data-p="candidates"><span class="dot"></span>候选池</div>
  <div class="nitem" data-p="factors"><span class="dot"></span>因子结论</div>
  <div class="ngroup">跟踪</div>
  <div class="nitem" data-p="paper"><span class="dot"></span>模拟盘</div>
  <div class="nitem" data-p="journal"><span class="dot"></span>战绩记账</div>
  <div class="ngroup">运行</div>
  <div class="nitem" data-p="run"><span class="dot"></span>任务与日志</div>
  <div class="ngroup">配置</div>
  <div class="nitem" data-p="settings"><span class="dot"></span>设置</div>
</aside>
<main>
  <section class="page on" id="p-overview"></section>
  <section class="page" id="p-picks"></section>
  <section class="page" id="p-candidates"></section>
  <section class="page" id="p-factors"></section>
  <section class="page" id="p-paper"></section>
  <section class="page" id="p-journal"></section>
  <section class="page" id="p-run"></section>
  <section class="page" id="p-settings"></section>
</main></div>
<script>
const $=(s,r=document)=>r.querySelector(s), E=s=>document.createElement(s);
const pct=(x,n=2)=>x==null||isNaN(x)?'—':(x>=0?'+':'')+(x*100).toFixed(n)+'%';
const cls=x=>x==null?'':x>0?'up':x<0?'down':'';
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let ST={};

document.querySelectorAll('.nitem').forEach(n=>n.onclick=()=>{
  document.querySelectorAll('.nitem').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('on'));
  n.classList.add('on'); $('#p-'+n.dataset.p).classList.add('on'); load(n.dataset.p);
});

async function J(u,o){const r=await fetch(u,o);return r.json()}

async function load(p){
  if(p==='overview')return renderOverview();
  if(p==='picks')return renderPicks();
  if(p==='candidates')return renderCands();
  if(p==='factors')return renderFactors();
  if(p==='paper')return renderPaper();
  if(p==='journal')return renderJournal();
  if(p==='run')return renderRun();
  if(p==='settings')return renderSettings();
}

async function renderOverview(){
  const s=await J('/api/status'), j=await J('/api/journal');
  ST=s;
  const keyOn=Object.entries(s.keys).filter(([,v])=>v).map(([k])=>k.split('_')[0]);
  $('#p-overview').innerHTML=`
  <h2>总览</h2><p class="lede">数据截至 ${s.px_latest||'—'} · 全部来自本地，页面不联网</p>
  <div class="cards">
    <div class="card"><div class="l">股票池</div><div class="v">${s.stocks}</div>
      <div class="d">两融 ${s.margin} 只</div></div>
    <div class="card"><div class="l">研报 / 股东户数</div>
      <div class="v" style="font-size:19px">${(s.reports/1e4).toFixed(1)}万 / ${(s.holders/1e4).toFixed(1)}万</div>
      <div class="d">价值陷阱排查依据</div></div>
    <div class="card"><div class="l">因子层样本外</div><div class="v">0.88</div>
      <div class="d">夏普 · 年化 +23.2%</div></div>
    <div class="card"><div class="l">LLM 层证据</div>
      <div class="v" style="color:var(--t3)">${j.mature}/30</div>
      <div class="d">达 30 个样本才有初步结论</div>
      <div class="bar"><i style="width:${Math.min(100,j.mature/30*100)}%"></i></div></div>
  </div>
  <div class="sec"><h3>接口与情绪源</h3><div class="scroll"><table>
    <thead><tr><th>项目</th><th>状态</th><th>说明</th></tr></thead><tbody>
    ${Object.entries(s.keys).map(([k,v])=>`<tr><td>${k}</td>
      <td><span class="pill ${v?'on':'off'}">${v?'已设置':'未设置'}</span></td>
      <td style="color:var(--t2)">${k==='GEMINI_API_KEY'?'免费额度是需求的 250 倍':
        k==='XAI_API_KEY'?'可官方检索 X，替代爬虫':'备选'}</td></tr>`).join('')}
    ${Object.entries(s.feeds).map(([k,v])=>`<tr><td>情绪源 ${k}</td>
      <td><span class="pill ${v?'on':'off'}">${v?v+' 个文件':'无数据'}</span></td>
      <td style="color:var(--t2)">不进因子打分，仅记账验证</td></tr>`).join('')}
    </tbody></table></div></div>
  <div class="note">因子层有统计支撑（2804 只全市场、6 折滚动前推、多重检验）。
    <b>LLM 层与情绪源目前没有任何有效性证据</b>，正在前瞻记账中 —— 在样本达到 30 之前，任何单笔盈亏都是噪声。</div>`;
}

async function renderPicks(){
  const d=await J('/api/picks');
  const box=$('#p-picks');
  if(!d.picks.length){box.innerHTML=`<h2>本周推荐</h2><p class="lede">还没有扫描记录。去「任务与日志」运行一次扫描。</p>`;return}
  box.innerHTML=`<h2>本周推荐</h2><p class="lede">基准日 ${d.date} · 只使用该日及以前的数据</p>`+
  d.picks.map(p=>{const f=p.forecast||{};return `<div class="pick"><div class="pick-h">
    <span class="c">${esc(p.symbol)}</span><span class="nm">${esc(p.name||'')}</span>
    <span class="pill on">信心 ${esc(p.score)}/10</span>
    ${f.price_at_scan?`<span class="px">${f.price_at_scan.toFixed(2)} 元</span>`:''}</div>
    <p><b>理由</b> ${esc(p.reason)}</p>
    ${p.risks?`<p><b>风险</b> ${esc(p.risks.join('；'))}</p>`:''}
    ${p.entry_note?`<p><b>进场</b> ${esc(p.entry_note)}</p>`:''}
    ${f.enough?`<div class="note">历史类比 ${f.n_analogs.toLocaleString()} 个样本 ·
      预期最高价中位 <b>${f.peak_price_q.p50.toFixed(2)}</b>（${pct(f.mfe_q.p50,1)}）·
      见顶中位第 ${f.days_to_peak_median.toFixed(0)} 日 ·
      摸到 +15% ${(f.p_hit_15*100).toFixed(1)}% · 触发止损 ${(f.p_stop_15*100).toFixed(1)}% ·
      期望收益 ${pct(f.expected_ret)}</div>`:''}</div>`}).join('')+
  (d.market_view?`<div class="note"><b>大盘观点</b>　${esc(d.market_view)}</div>`:'')+
  (d.rejected&&d.rejected.length?`<div class="sec"><h3>被剔除的</h3><div class="scroll"><table>
    <thead><tr><th>代码</th><th>理由</th></tr></thead><tbody>
    ${d.rejected.map(r=>`<tr><td>${esc(r.symbol)}</td><td style="color:var(--t2)">${esc(r.why)}</td></tr>`).join('')}
    </tbody></table></div></div>`:'')+
  (d.views?Object.entries(d.views).map(([k,v])=>`<details><summary>展开：${k} 分析师全文</summary>
    <pre>${esc(v)}</pre></details>`).join(''):'')+
  (d.debate?`<details><summary>展开：多空辩论全文</summary><pre>${esc(d.debate)}</pre></details>`:'');
}

async function renderCands(){
  const d=await J('/api/candidates');
  if(!d.facts.length){$('#p-candidates').innerHTML='<h2>候选池</h2><p class="lede">还没有数据。</p>';return}
  $('#p-candidates').innerHTML=`<h2>候选池</h2>
  <p class="lede">${d.date} · 因子排序前 ${d.facts.length} 只，喂给分析师的原始信息包</p>
  <div class="sec"><div class="scroll"><table><thead><tr>
    <th>#</th><th>代码</th><th>名称</th><th class="n">现价</th><th class="n">20日</th>
    <th class="n">60日</th><th class="n">波动率</th><th class="n">RSI</th>
    <th class="n">区间位置</th><th class="n">成交额</th><th class="n">EPS修正</th>
  </tr></thead><tbody>${d.facts.map(f=>`<tr>
    <td class="n">${f.factor_rank??''}</td><td>${esc(f.symbol)}</td><td>${esc(f.name||'')}</td>
    <td class="n">${f.price?.toFixed(2)??'—'}</td>
    <td class="n ${cls(f.ret_20d)}">${pct(f.ret_20d,1)}</td>
    <td class="n ${cls(f.ret_60d)}">${pct(f.ret_60d,1)}</td>
    <td class="n">${pct(f.vol_60d_ann,0)}</td>
    <td class="n">${f.rsi_14?.toFixed(0)??'—'}</td>
    <td class="n">${f.pos_in_60d_range?.toFixed(2)??'—'}</td>
    <td class="n">${f.amount_20d?(f.amount_20d/1e8).toFixed(2)+'亿':'—'}</td>
    <td class="n ${cls(f.analyst_eps_revision)}">${f.analyst_eps_revision!=null?pct(f.analyst_eps_revision,1):'—'}</td>
  </tr>`).join('')}</tbody></table></div></div>`;
}

async function renderFactors(){
  const f=await J('/api/factors'), a=f.adopted, b=f.bench;
  const mx=Math.max(...f.windows.map(w=>w[1]));
  $('#p-factors').innerHTML=`<h2>因子结论</h2>
  <p class="lede">2804 只全市场 · 6 折滚动前推 · 多重检验校正</p>
  <div class="cards">
    <div class="card"><div class="l">采用</div><div class="v" style="font-size:15px">${a.name}</div></div>
    <div class="card"><div class="l">样本外年化</div><div class="v up">${pct(a.cagr)}</div>
      <div class="d">沪深300 ${pct(b.cagr)}</div></div>
    <div class="card"><div class="l">夏普</div><div class="v">${a.sharpe}</div>
      <div class="d">卡玛 ${a.calmar} · 盈利月 ${(a.pos_month*100).toFixed(0)}%</div></div>
    <div class="card"><div class="l">最大回撤</div><div class="v down">${pct(a.mdd)}</div>
      <div class="d">沪深300 ${pct(b.mdd)}</div></div>
  </div>
  <div class="sec"><h3>参数高原 · 反转窗口的样本外夏普</h3>
  <div class="scroll"><table><thead><tr><th>回看窗口</th><th class="n">夏普</th><th>分布</th></tr></thead>
  <tbody>${f.windows.map(([w,s])=>`<tr><td>${w} 日</td><td class="n">${s.toFixed(2)}</td>
    <td><div class="bar" style="max-width:260px"><i style="width:${s/mx*100}%"></i></div></td></tr>`).join('')}
  </tbody></table></div>
  <div class="note">30–120 日整片有效，只有两端塌陷 —— <b>是高原不是孤峰</b>，所以用 30/60/90 等权，
    而不是挑夏普最高的 30 日（那是选择偏差）。</div></div>
  <div class="sec"><h3>被拒绝的候选</h3><div class="scroll"><table>
  <thead><tr><th>因子</th><th class="n">增量 t 值</th><th>原因</th></tr></thead><tbody>
  ${f.rejected.map(r=>`<tr><td>${esc(r.n)}</td>
    <td class="n">${r.t==null?'—':r.t.toFixed(2)}</td>
    <td style="color:var(--t2)">${esc(r.why)}</td></tr>`).join('')}
  </tbody></table></div>
  <div class="note">8 个候选，1 个通过。<b>这个通过率是正常的</b> —— 拦掉的每一个，
    都是一次本来会发生的实盘亏损。</div></div>`;
}

async function renderJournal(){
  const j=await J('/api/journal');
  const st=Object.entries(j.stats);
  $('#p-journal').innerHTML=`<h2>战绩记账</h2>
  <p class="lede">执行假设与回测一致：推荐日次一交易日开盘买入</p>
  ${st.length?`<div class="sec"><h3>汇总</h3><div class="scroll"><table>
    <thead><tr><th>持有</th><th class="n">样本</th><th class="n">平均收益</th>
    <th class="n">胜率</th><th class="n">超额</th></tr></thead><tbody>
    ${st.map(([h,v])=>`<tr><td>${h} 日</td><td class="n">${v.n}</td>
      <td class="n ${cls(v.avg)}">${pct(v.avg)}</td><td class="n">${(v.win*100).toFixed(0)}%</td>
      <td class="n ${cls(v.exc)}">${v.exc==null?'—':pct(v.exc)}</td></tr>`).join('')}
    </tbody></table></div></div>`:'<p class="lede">还没有已到期的样本。</p>'}
  ${j.mature<30?`<div class="note">⚠️ 已到期样本 ${j.mature} 个，<b>远不足以判断系统是否有效</b>。
    按每周 2–3 只算需累计约 3 个月。现在的胜率请当噪声看。</div>`:''}
  <div class="sec"><h3>逐笔记录</h3><div class="scroll"><table><thead><tr>
    <th>扫描日</th><th>代码</th><th>名称</th><th class="n">信心</th><th class="n">扫描价</th>
    <th class="n">买入价</th><th class="n">5日</th><th class="n">10日</th><th class="n">20日</th>
    <th class="n">超额20日</th></tr></thead><tbody>
    ${j.rows.map(r=>`<tr><td>${esc(r.scan_date)}</td><td>${esc(r.symbol)}</td>
      <td>${esc(r.name)}</td><td class="n">${r.score??'—'}</td>
      <td class="n">${r.price_at_scan?.toFixed(2)??'—'}</td>
      <td class="n">${r.entry_px?.toFixed(2)??'—'}</td>
      <td class="n ${cls(r.ret_5d)}">${pct(r.ret_5d,1)}</td>
      <td class="n ${cls(r.ret_10d)}">${pct(r.ret_10d,1)}</td>
      <td class="n ${cls(r.ret_20d)}">${pct(r.ret_20d,1)}</td>
      <td class="n ${cls(r.excess_20d)}">${pct(r.excess_20d,1)}</td></tr>`).join('')}
    </tbody></table></div></div>`;
}

let esOn=false;
async function renderRun(){
  const s=await J('/api/status');
  $('#p-run').innerHTML=`<h2>任务与日志</h2>
  <p class="lede">同一时刻只允许一个任务运行 —— 并发写 journal 会互相覆盖</p>
  <div class="sec"><h3>可运行的任务</h3><div class="runs" id="runs"></div>
    <div style="margin-top:14px"><button id="stop">终止当前任务</button></div></div>
  <div class="sec"><h3>实时日志</h3><div id="console"></div></div>`;
  const box=$('#runs');
  const D={update:'增量拉取行情、指数、国际环境。约 12 分钟。',
    scan:'硬规则 → 因子 → 五分析师 → 辩论 → 推荐。需要 LLM key。',
    scan_nollm:'只出因子候选池，不花 token。',
    journal:'给历史推荐补算 5/10/20 日前瞻收益。'};
  Object.entries(s.tasks).forEach(([k,label])=>{
    const d=E('div');d.className='run';
    d.innerHTML=`<div class="t">${label}</div><div class="d">${D[k]||''}</div>`;
    const b=E('button');b.textContent='运行';b.className=k==='scan'?'primary':'';
    b.disabled=!!s.running;
    b.onclick=async()=>{b.disabled=true;
      const r=await J('/api/run/'+k,{method:'POST'});
      if(!r.ok){alert(r.msg);b.disabled=false}else{renderRun()}};
    d.appendChild(b);box.appendChild(d);
  });
  $('#stop').onclick=async()=>{await J('/api/stop',{method:'POST'})};
  if(!esOn){esOn=true;
    const es=new EventSource('/api/log');
    es.onmessage=e=>{const c=$('#console');if(!c)return;
      const ln=JSON.parse(e.data).line, s=E('div');
      s.textContent=ln;
      if(/失败|错误|Error|Traceback|!/.test(ln))s.className='err';
      else if(/⚠️|警告|注意/.test(ln))s.className='warn';
      else if(/\[完成\]|完成|通过/.test(ln))s.className='ok';
      c.appendChild(s);c.scrollTop=c.scrollHeight};
  }
}

async function renderSettings(){
  const d=await J('/api/settings');
  const H=d.hard||{}, R=d.risk||{}, S=d.schedule||{}, SP=d.soft||{};
  const num=(id,label,val,hint,step)=>`<div class="fld"><label for="${id}">${label}</label>
    <input id="${id}" type="number" step="${step||'any'}" value="${val??''}">
    <span class="hint">${hint||''}</span></div>`;
  const list=a=>Array.isArray(a)?a.join('\n'):'';
  $('#p-settings').innerHTML=`<h2>设置</h2>
  <p class="lede">改动写入 <code>rules.yaml</code> 与 <code>${d.env_path}</code>。
    保存前会自动备份为 <code>rules.yaml.bak</code>，改坏了能回滚。</p>

  <div class="sec"><h3>API Key</h3>
  <p style="color:var(--t2);font-size:13px;margin:0 0 12px">
    写入项目内 <code>.env</code>（权限 600，已 gitignore），不写进全局 shell ——
    全局环境变量任何进程都能读。留空表示不修改。</p>
  <div class="form">
  ${Object.entries(d.keys).map(([k,v])=>`<div class="fld">
    <label for="k-${k}">${k}</label>
    <input id="k-${k}" type="password" placeholder="${v.set?'已设置 '+v.tail+'（留空不改）':'未设置'}">
    <span class="hint">${k==='GEMINI_API_KEY'?'免费，aistudio.google.com':
      k==='XAI_API_KEY'?'付费，可官方检索 X：console.x.ai':
      k==='DEEPSEEK_API_KEY'?'约 ¥0.15/次扫描':'glm-4-flash 免费'}</span></div>`).join('')}
  </div><div style="margin-top:12px"><button class="primary" id="save-keys">保存 Key</button></div></div>

  <div class="sec"><h3>硬约束（在 LLM 之前执行）</h3>
  <p style="color:var(--t2);font-size:13px;margin:0 0 12px">
    这些是代码级铁律，不是提示词 —— LLM 很擅长为任何标的编理由，硬规则必须挡在它前面。</p>
  <div class="form">
    ${num('h-min_amount_20d','20日均成交额下限（元）',H.min_amount_20d,'流动性门槛，默认 1 亿',1e7)}
    ${num('h-min_mcap','总市值下限（元）',H.min_mcap,'默认 50 亿',1e8)}
    ${num('h-max_consecutive_limit_up','连续涨停上限',H.max_consecutive_limit_up,'超过就排除，你设的是 1',1)}
    ${num('h-exclude_new_days','次新股排除天数',H.exclude_new_days,'上市不满 N 个交易日不碰',10)}
    ${num('h-min_vol_60d','60日波动率下限',H.min_vol_60d,'留空=不限。想要弹性可设 0.35',0.05)}
    ${num('h-max_vol_60d','60日波动率上限',H.max_vol_60d,'留空=不限。防妖股最后一棒',0.05)}
  </div></div>

  <div class="sec"><h3>风控</h3>
  <div class="form">
    ${num('r-max_positions','最多同时持有','','数据显示 1-2 只夏普明显更差',1)}
    ${num('r-init_cash','模拟盘本金（元）',R.init_cash,'决定买得起多贵的股票',10000)}
    ${num('r-position_size','单只仓位比例',R.position_size,
      '<span id="afford"></span>',0.01)}
    ${num('r-stop_loss','止损',R.stop_loss,'实测 15% 几乎不误杀赢家（仅0.2%）',0.01)}
    ${num('r-take_profit','止盈',R.take_profit,'留空=不止盈。实测 15% 以上差异在噪声内',0.01)}
    ${num('r-max_hold_days','最长持有（交易日）',R.max_hold_days,'实测越长越好，别短于 7 日',1)}
    ${num('r-monthly_drawdown_halt','月度回撤熔断',R.monthly_drawdown_halt,'超过就停开新仓',0.01)}
  </div></div>

  <div class="sec"><h3>扫描</h3>
  <div class="form">
    ${num('s-n_candidates_to_llm','交给 LLM 的候选数',S.n_candidates_to_llm,'默认 20，越多越贵',1)}
    ${num('s-n_recommendations','最多推荐几只',S.n_recommendations,'LLM 可以少推，不会凑数',1)}
  </div></div>

  <div class="sec"><h3>你的偏好（写进提示词）</h3>
  <p style="color:var(--t2);font-size:13px;margin:0 0 12px">每行一条。这部分只有你能写 ——
    写得越具体，系统越是在复现你的判断，而不是模型自由发挥。</p>
  <div class="form col">
    <div class="fld"><label for="sp-likes">喜欢的形态</label>
      <textarea id="sp-likes" rows="4">${esc(list(SP.likes))}</textarea></div>
    <div class="fld"><label for="sp-avoids">回避的情况</label>
      <textarea id="sp-avoids" rows="4">${esc(list(SP.avoids))}</textarea></div>
    <div class="fld"><label for="sp-fast">「来钱快」的定义</label>
      <textarea id="sp-fast" rows="4">${esc(list(SP.fast_money_definition))}</textarea></div>
  </div></div>

  <div style="margin:16px 0"><button class="primary" id="save-all">保存全部设置</button>
    <span id="msg" style="margin-left:12px;color:var(--t2)"></span></div>

  <div class="sec"><h3>直接编辑 rules.yaml</h3>
  <p style="color:var(--t2);font-size:13px;margin:0 0 10px">
    上面的表单只覆盖常用项。要改行业黑名单、个股黑名单这类，在这里直接改 ——
    保存前会做 YAML 语法校验，语法错了不会落盘。</p>
  <textarea id="raw" rows="20" style="width:100%;font-family:var(--mono);font-size:12px">${esc(d.raw)}</textarea>
  <div style="margin-top:10px"><button id="save-raw">保存 YAML</button></div></div>`;

  const gv=id=>{const e=$('#'+id);if(!e)return undefined;const v=e.value.trim();
    return v===''?null:(isNaN(+v)?v:+v)};
  const lines=id=>$('#'+id).value.split('\n').map(x=>x.trim()).filter(Boolean);
  const say=(m,ok=true)=>{const e=$('#msg');e.textContent=m;
    e.style.color=ok?'var(--down)':'var(--up)';setTimeout(()=>e.textContent='',4000)};

  $('#save-keys').onclick=async()=>{
    const keys={};
    Object.keys(d.keys).forEach(k=>{const v=$('#k-'+k).value.trim();if(v)keys[k]=v});
    if(!Object.keys(keys).length)return say('没有填写任何 key',false);
    const r=await J('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({keys})});
    say(r.ok?'Key 已保存，重启服务后生效':r.msg,r.ok);
  };
  $('#save-all').onclick=async()=>{
    const f={};
    [['hard_filters','h-',['min_amount_20d','min_mcap','max_consecutive_limit_up',
      'exclude_new_days','min_vol_60d','max_vol_60d']],
     ['risk','r-',['max_positions','position_size','stop_loss','take_profit',
      'max_hold_days','monthly_drawdown_halt']],
     ['schedule','s-',['n_candidates_to_llm','n_recommendations']]
    ].forEach(([sec,pre,ks])=>ks.forEach(k=>{const v=gv(pre+k);
      if(v!==undefined)f[sec+'.'+k]=v}));
    f['soft_prefs.likes']=lines('sp-likes');
    f['soft_prefs.avoids']=lines('sp-avoids');
    f['soft_prefs.fast_money_definition']=lines('sp-fast');
    const r=await J('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({fields:f})});
    say(r.ok?r.msg:r.msg,r.ok); if(r.ok)setTimeout(renderSettings,600);
  };
  $('#save-raw').onclick=async()=>{
    const r=await J('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({raw:$('#raw').value})});
    say(r.ok?r.msg:r.msg,r.ok); if(r.ok)setTimeout(renderSettings,600);
  };
  const mp=$('#r-max_positions'); if(mp) mp.value=R.max_positions??5;
}

function sparkline(pts, w, h){
  if(!pts || pts.length < 2) return '';
  const vs = pts.map(p=>p.v), lo = Math.min(...vs), hi = Math.max(...vs);
  const rng = (hi-lo) || 1, n = pts.length;
  const X = i => 34 + i/(n-1)*(w-46);
  const Y = v => 10 + (1-(v-lo)/rng)*(h-30);
  const line = pts.map((p,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(p.v).toFixed(1)}`).join(' ');
  const area = line + ` L${X(n-1).toFixed(1)},${h-20} L${X(0).toFixed(1)},${h-20} Z`;
  const base = pts[0].v, last = pts[n-1].v;
  const col = last >= base ? 'var(--up)' : 'var(--down)';
  const zeroY = Y(base).toFixed(1);
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:${h}px" role="img"
    aria-label="模拟盘净值曲线">
    <line x1="34" y1="${zeroY}" x2="${w-12}" y2="${zeroY}" stroke="var(--line2)"
      stroke-width="1" stroke-dasharray="3 3"/>
    <path d="${area}" fill="${col}" opacity=".08"/>
    <path d="${line}" fill="none" stroke="${col}" stroke-width="1.8"
      stroke-linejoin="round"/>
    <circle cx="${X(n-1).toFixed(1)}" cy="${Y(last).toFixed(1)}" r="3.2" fill="${col}"/>
    <text x="4" y="14" font-size="9" fill="var(--t3)" font-family="var(--mono)">${hi.toFixed(0)}</text>
    <text x="4" y="${h-16}" font-size="9" fill="var(--t3)" font-family="var(--mono)">${lo.toFixed(0)}</text>
    <text x="34" y="${h-4}" font-size="9" fill="var(--t3)" font-family="var(--mono)">${pts[0].d}</text>
    <text x="${w-12}" y="${h-4}" font-size="9" fill="var(--t3)" text-anchor="end"
      font-family="var(--mono)">${pts[n-1].d}</text>
  </svg>`;
}

async function renderPaper(){
  const box = $('#p-paper');
  box.innerHTML = '<h2>模拟盘</h2><p class="lede">加载行情中（首次约 1-2 分钟）…</p>';
  const d = await J('/api/paper');
  if(d.error){ box.innerHTML = `<h2>模拟盘</h2><div class="note">出错：${esc(d.error)}</div>`; return; }
  const S = d.summary || {}, R = d.rules || {};
  const rulesTxt = `每只 ${((R.position_size||0)*100).toFixed(0)}% 仓位 · `
    + `止损 ${R.stop_loss?(R.stop_loss*100).toFixed(0)+'%':'无'} · `
    + `止盈 ${R.take_profit?(R.take_profit*100).toFixed(0)+'%':'无'} · `
    + `最长持有 ${R.max_hold_days||'不限'} 日 · 最多 ${R.max_positions||5} 只`;

  if(S.empty){
    box.innerHTML = `<h2>模拟盘</h2><p class="lede">${rulesTxt}</p>
      <div class="note"><b>尚未开始计算</b><br>${
        (d.blocked||[]).map(esc).join('<br>')||'还没有推荐记录'}</div>`;
    return;
  }

  const tot = S.total_return, exc = S.excess;
  const tr = t => `<tr><td>${esc(t.symbol)}</td><td>${esc(t.name)}</td>
    <td>${esc(t.open_date)}</td><td class="n">${t.open_px}</td>
    <td class="n">${t.shares.toLocaleString()}</td>
    <td>${esc(t.close_date)||'—'}</td>
    <td class="n">${t.close_px||t.mark_px||'—'}</td>
    <td class="n ${cls(t.pnl)}">${t.pnl>=0?'+':''}${t.pnl.toLocaleString()}</td>
    <td class="n ${cls(t.ret)}">${pct(t.ret,1)}</td>
    <td style="color:var(--t2)">${esc(t.reason)}</td></tr>`;
  const HEAD = `<thead><tr><th>代码</th><th>名称</th><th>买入日</th><th class="n">买入价</th>
    <th class="n">股数</th><th>卖出日</th><th class="n">现价/卖价</th>
    <th class="n">盈亏</th><th class="n">收益率</th><th>状态</th></tr></thead>`;

  const A = d.actions || [];
  const nSell = A.filter(a=>a.urgency==='sell').length;
  const nWarn = A.filter(a=>a.urgency==='warn').length;
  const actRow = a => `<tr class="act-${a.urgency}">
    <td><span class="tag ${a.urgency}">${esc(a.action)}</span></td>
    <td>${esc(a.symbol)}</td><td>${esc(a.name)}</td>
    <td class="n">${a.held_days}日</td>
    <td class="n">${a.open_px}</td><td class="n">${a.last_px}</td>
    <td class="n ${cls(a.ret)}">${pct(a.ret,1)}</td>
    <td class="n">${a.stop_px??'—'}</td>
    <td style="color:var(--t2)">${esc(a.why)}</td></tr>`;

  box.innerHTML = `<h2>模拟盘</h2>
  <p class="lede">${esc(S.start)} 起 · ${S.n_days} 个交易日 · ${rulesTxt}</p>
  ${A.length?`<div class="sec" style="${nSell?'border-color:var(--up)':''}">
    <h3>下一交易日操作　${nSell?`<span class="tag sell">${nSell} 只需卖出</span>`:
      nWarn?`<span class="tag warn">${nWarn} 只需留意</span>`:
      `<span class="tag hold">全部持有</span>`}</h3>
    <div class="scroll"><table><thead><tr>
      <th>操作</th><th>代码</th><th>名称</th><th class="n">持有</th>
      <th class="n">成本</th><th class="n">现价</th><th class="n">盈亏</th>
      <th class="n">止损线</th><th>说明</th></tr></thead>
      <tbody>${A.map(actRow).join('')}</tbody></table></div>
    <div class="note" style="margin-bottom:0">判定用<b>最新收盘价</b>，
      执行在<b>次一交易日开盘</b> —— 与回测口径一致。
      系统只做提醒，<b>不会替你下单</b>。</div></div>`:''}
  <div class="cards">
    <div class="card"><div class="l">当前权益</div>
      <div class="v">${S.equity.toLocaleString(undefined,{maximumFractionDigits:0})}</div>
      <div class="d">初始 ${S.init_cash.toLocaleString()} · 现金 ${S.cash.toLocaleString(undefined,{maximumFractionDigits:0})}</div></div>
    <div class="card"><div class="l">总收益</div>
      <div class="v ${cls(tot)}">${pct(tot)}</div>
      <div class="d">沪深300 ${S.bench_return!=null?pct(S.bench_return):'—'}</div></div>
    <div class="card"><div class="l">超额收益</div>
      <div class="v ${cls(exc)}">${exc!=null?pct(exc):'—'}</div>
      <div class="d">最大回撤 ${pct(S.max_drawdown)}</div></div>
    <div class="card"><div class="l">交易</div>
      <div class="v">${S.n_closed}<span style="font-size:14px;color:var(--t3)">/${S.n_trades}</span></div>
      <div class="d">已平/总 · 胜率 ${S.win_rate!=null?(S.win_rate*100).toFixed(0)+'%':'—'}</div></div>
  </div>
  <div class="sec"><h3>净值曲线</h3>${sparkline(d.equity, 900, 220)}
    <div style="display:flex;gap:20px;margin-top:10px;font-size:13px;color:var(--t2)">
      <span>已实现盈亏 <b class="${cls(S.realized_pnl)}">${S.realized_pnl>=0?'+':''}${S.realized_pnl.toLocaleString(undefined,{maximumFractionDigits:0})}</b></span>
      <span>浮动盈亏 <b class="${cls(S.unrealized_pnl)}">${S.unrealized_pnl>=0?'+':''}${S.unrealized_pnl.toLocaleString(undefined,{maximumFractionDigits:0})}</b></span>
    </div></div>
  ${S.too_short?`<div class="note">⚠️ 只有 ${S.n_days} 个交易日，<b>刻意不显示年化和夏普</b> ——
    把三个月的收益年化会得到荒谬的数字。样本够长（≥120 个交易日）才会出现这两项。</div>`:''}
  ${d.open.length?`<div class="sec"><h3>当前持仓 ${d.open.length} 只</h3>
    <div class="scroll"><table>${HEAD}<tbody>${d.open.map(tr).join('')}</tbody></table></div></div>`:''}
  ${d.closed.length?`<div class="sec"><h3>已平仓 ${d.closed.length} 笔</h3>
    <div class="scroll"><table>${HEAD}<tbody>${d.closed.slice().reverse().map(tr).join('')}</tbody></table></div></div>`:''}
  ${(d.pending||[]).length?`<div class="sec"><h3>待执行 ${d.pending.length} 只</h3>
    <p style="color:var(--t2);font-size:13px;margin:0 0 8px">已推荐但还没到执行日 ——
      执行价是推荐日的次一交易日开盘。</p>
    <div class="scroll"><table><thead><tr><th>代码</th><th>名称</th>
      <th>推荐日</th><th>状态</th></tr></thead><tbody>
      ${d.pending.map(p=>`<tr><td>${esc(p.symbol)}</td><td>${esc(p.name)}</td>
        <td>${esc(p.scan_date)}</td><td style="color:var(--t2)">${esc(p.why)}</td></tr>`).join('')}
    </tbody></table></div></div>`:''}
  ${(d.blocked||[]).length?`<div class="sec"><h3>受阻记录</h3>
    <p style="color:var(--t2);font-size:13px;margin:0 0 8px">涨停买不进、跌停卖不出、停牌、
      组合满员、预算不够一手 —— 这些在实盘同样会发生，模拟盘如实记录。
      <b>每只推荐必定落在「建仓 / 待执行 / 受阻」三者之一</b>，不会凭空消失。</p>
    <div class="scroll"><table><tbody>${d.blocked.map(b=>`<tr><td style="color:var(--t2)">${esc(b)}</td></tr>`).join('')}</tbody></table></div></div>`:''}
  <div class="note">成交假设与回测引擎完全一致：推荐日收盘出信号 → <b>次一交易日开盘</b>买入；
    T+1 制度；涨停买不进、跌停卖不出；止损止盈按<b>收盘判定、次日开盘执行</b>；
    计入佣金、印花税、过户费与滑点。</div>`;
}

renderOverview();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5111)
    ap.add_argument("--host", default="127.0.0.1",
                    help="默认只绑本机。对外提供服务需同时设置 PANEL_TOKEN")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)

    local_only = a.host in ("127.0.0.1", "localhost", "::1")
    if not local_only and not PANEL_TOKEN:
        # /api/run 能起子进程、/api/settings 能写文件。
        # 对外绑定又没口令，等于把服务器交给公网 —— 直接拒绝启动。
        print(f"拒绝启动：绑定到 {a.host} 属于对外提供服务，但没有设置 PANEL_TOKEN。\n"
              f"  两个选择：\n"
              f"    1) 设置口令：在 .env 里加一行 PANEL_TOKEN=你的口令\n"
              f"    2) 更省事也更安全 —— 保持绑本机，用 SSH 隧道访问：\n"
              f"       ssh -L {a.port}:127.0.0.1:{a.port} 你@服务器\n"
              f"       然后本地打开 http://127.0.0.1:{a.port}")
        raise SystemExit(2)

    url = f"http://127.0.0.1:{a.port}"
    print(f"控制台: {url}    （Ctrl-C 停止）")
    print("鉴权: " + ("已启用 PANEL_TOKEN" if PANEL_TOKEN else "关闭（仅本机可访问）"))
    if not local_only:
        print(f"⚠️ 绑定 {a.host} —— 已对外提供服务，请确认防火墙只放行你需要的来源")
    if not a.no_open and local_only:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
