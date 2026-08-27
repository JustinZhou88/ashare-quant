"""用 Playwright 抓 X 上指定博主的 A股相关提及。

    python tools/x_scrape.py --accounts accounts.txt --days 7

⚠️ 使用前必读
  1. X 的服务条款禁止自动化抓取。本脚本复用你**自己浏览器的登录态**做个人研究，
     抓取频率刻意压得很低。即便如此，账号仍有被限制的风险 —— 建议用小号。
  2. 抓到的东西**不能回测**（拿不到历史推文），所以它不进因子打分，
     只写进 LLM 上下文并被 journal 记账，等积累够样本再判断有没有用。
  3. 默认按「**反向拥挤指标**」解释：某只票被集中提及往往意味着
     流动性出口正在打开。是正是负最终由记账数据决定，不由预设决定。
  4. 如果它真的被验证有效，请换成 X 官方 API（付费）—— 爬虫不可能长期稳定。

首次运行会打开浏览器让你扫码/登录，登录态存在 .x_profile/ 供后续复用。

输出：sentiment_feed/x_influencer/YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROFILE = Path(".x_profile")
OUT_DIR = Path("sentiment_feed/x_influencer")
# 不能用 \b：Python 的 \b 把中文也算词字符，"688386值得关注" 里
# "6" 和 "值" 之间没有词边界，会漏掉紧挨中文的股票代码。
# 改用数字前后瞻，既避开中文问题，又不会把长数字串的一部分误当代码。
CODE_RE = re.compile(r"(?<!\d)(6\d{5}|00\d{4}|30\d{4}|68[89]\d{3})(?!\d)")


def load_name_map() -> dict[str, str]:
    """股票简称 -> 代码。用于识别正文里只提名字不提代码的情况。"""
    try:
        import pandas as pd
        f = Path("data_cache/_symbols.csv")
        if not f.exists():
            return {}
        d = pd.read_csv(f, dtype={"symbol": str})
        return {str(r["name"]).strip(): str(r["symbol"]).zfill(6)
                for _, r in d.iterrows() if isinstance(r["name"], str) and len(str(r["name"])) >= 2}
    except Exception:                                          # noqa: BLE001
        return {}


def extract_symbols(text: str, name_map: dict[str, str]) -> set[str]:
    hits = set(CODE_RE.findall(text))
    for name, code in name_map.items():
        if name in text:
            hits.add(code)
    return hits


def logged_in(ctx, page) -> bool:
    """判断是否真的登录了 X。

    **不能只看 URL**：实测未登录时 X 直接在 `https://x.com/` 渲染登录墙，
    URL 里没有 login / i/flow，光看 URL 会误判成已登录，然后抓到 0 条还找不到原因。
    真正可靠的判据是 `auth_token` 这个 cookie 在不在。
    """
    try:
        names = {c.get("name") for c in ctx.cookies()}
        if "auth_token" in names:
            return True
    except Exception:                                          # noqa: BLE001
        pass
    try:
        body = page.inner_text("body")[:400]
        if "Continue with" in body or "Sign in to X" in body or "登录" in body[:120]:
            return False
    except Exception:                                          # noqa: BLE001
        pass
    return "login" not in page.url and "i/flow" not in page.url


def resolve_profile(args) -> tuple[str, list[str]]:
    """决定用哪个浏览器配置目录。

    --use-my-chrome：直接指向你本机 Chrome 的配置，现成的 X 登录态直接可用，
    省掉扫码登录这一步。代价是 **Chrome 必须先完全退出** ——
    Chrome 用文件锁保护配置目录，同一目录不允许两个进程同时打开。

    默认（不加这个开关）：用项目内的独立配置 .x_profile/，
    首次要登录一次，但和你日常浏览完全隔离，也不用关 Chrome。
    """
    if not args.use_my_chrome:
        PROFILE.mkdir(exist_ok=True)
        return str(PROFILE), []
    real = Path.home() / "Library/Application Support/Google/Chrome"
    if not real.exists():
        print(f"找不到 Chrome 配置目录 {real}，退回独立配置。")
        PROFILE.mkdir(exist_ok=True)
        return str(PROFILE), []

    # Chrome 对配置目录加了文件锁，运行中就没法直接用。
    # 但锁只在原目录 —— 把登录态相关的几个文件**复制**一份出来就能绕开，
    # 而且完全不影响你正在用的 Chrome（只读复制，不碰原文件）。
    # 只拷必需文件，不拷缓存：整个 profile 可能有好几个 GB。
    import shutil
    clone = PROFILE.parent / ".x_profile_clone"
    if clone.exists():
        shutil.rmtree(clone, ignore_errors=True)
    (clone / "Default").mkdir(parents=True, exist_ok=True)
    ok = []
    for rel in ["Local State", "Default/Cookies", "Default/Preferences",
                "Default/Login Data", "Default/Network/Cookies",
                "Default/Secure Preferences"]:
        src = real / rel
        if not src.exists():
            continue
        dst = clone / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            ok.append(rel)
        except Exception as e:                                 # noqa: BLE001
            print(f"  ! 复制 {rel} 失败: {type(e).__name__}")
    print(f"  已复制 Chrome 登录态 {len(ok)} 个文件到临时配置（原 Chrome 不受影响）")
    return str(clone), ["--profile-directory=Default"]


def discover(args) -> None:
    """发现模式：搜索 A股关键词，统计哪些账号最常提及个股。

    为什么用这个而不是让人凭印象列名单：
    你要的是「在 A股话题上真的活跃」的账号，这件事**可以测**，
    不该靠记忆。搜索结果按互动度排序，提及个股次数多的自然浮上来。

    输出 accounts_discovered.txt，你自己筛一遍再决定放不放进 accounts.txt。
    """
    from playwright.sync_api import sync_playwright
    from collections import Counter

    name_map = load_name_map()
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    author_hits: Counter[str] = Counter()
    author_posts: Counter[str] = Counter()
    samples: dict[str, str] = {}

    PROFILE.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        profile_dir, extra = resolve_profile(args)
        kw = {"headless": args.headless, "args": extra,
              "viewport": {"width": 1280, "height": 900}}
        if args.channel:
            kw["channel"] = args.channel
        ctx = pw.chromium.launch_persistent_context(profile_dir, **kw)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://x.com/home", timeout=60000)
        page.wait_for_timeout(int(args.delay * 1000))
        if not logged_in(ctx, page):
            if not sys.stdin.isatty():
                print("未检测到 X 登录态，且当前不是交互终端 —— 直接退出，"
                      "不做无意义的等待。\n"
                      "请在浏览器里登录 X 后重跑；或去掉 --use-my-chrome，"
                      "用独立配置手动登录一次。")
                ctx.close()
                return
            print("需要登录。请在浏览器里完成登录，然后回车继续 ...")
            input()

        for q in queries:
            print(f"  搜索「{q}」...", flush=True)
            try:
                page.goto(f"https://x.com/search?q={q}&f=live", timeout=60000)
                page.wait_for_timeout(int(args.delay * 1000))
                for _ in range(args.max_scroll):
                    for art in page.query_selector_all("article"):
                        try:
                            txt = art.inner_text()
                        except Exception:                      # noqa: BLE001
                            continue
                        if not txt:
                            continue
                        m = re.search(r"@([A-Za-z0-9_]{2,15})", txt)
                        if not m:
                            continue
                        who = m.group(1)
                        author_posts[who] += 1
                        codes = extract_symbols(txt, name_map)
                        if codes:
                            author_hits[who] += len(codes)
                            samples.setdefault(who, txt[:90].replace("\n", " "))
                    page.mouse.wheel(0, 2400)
                    page.wait_for_timeout(int(args.delay * 1000))
            except Exception as e:                             # noqa: BLE001
                print(f"    ! 「{q}」失败: {type(e).__name__}: {str(e)[:70]}")
        ctx.close()

    ranked = sorted(author_hits.items(), key=lambda x: -x[1])
    out = Path("accounts_discovered.txt")
    lines = ["# 发现模式的结果 —— 按「提及个股次数」排序，不是按粉丝数。",
             "# 粉丝多不代表 A股内容多；这里排的是真的在聊个股的账号。",
             "# 请你自己看一眼再决定放不放进 accounts.txt。", ""]
    for who, n in ranked[:40]:
        lines.append(f"# {who}: 提及个股 {n} 次 / 共 {author_posts[who]} 条 "
                     f"| {samples.get(who,'')[:60]}")
        lines.append(who)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n发现 {len(ranked)} 个账号 -> {out}")
    for who, n in ranked[:12]:
        print(f"  @{who:20s} 提及个股 {n:3d} 次 / 共 {author_posts[who]:3d} 条")
    if not ranked:
        print("  没抓到 —— 可能未登录，或搜索词太窄。试试 --queries 换几个词。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", default="accounts.txt",
                    help="每行一个 X 用户名（不带 @）")
    ap.add_argument("--days", type=int, default=7, help="只保留最近 N 天的推文")
    ap.add_argument("--max-scroll", type=int, default=8, help="每个账号滚动几屏")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--delay", type=float, default=3.0, help="操作间隔秒数，别调小")
    ap.add_argument("--discover", action="store_true",
                    help="发现模式：搜索 A股关键词，按提及频次排出值得关注的账号")
    ap.add_argument("--queries", default="A股,涨停,超跌反弹,主力资金,龙虎榜",
                    help="发现模式的搜索词，逗号分隔")
    ap.add_argument("--use-my-chrome", action="store_true",
                    help="直接用你本机 Chrome 的现有登录态（不用再登一次）。"
                         "⚠️ 必须先完全退出 Chrome —— 同一个配置目录不能被两个进程同时占用")
    ap.add_argument("--channel", default="chrome",
                    help="用本机已装浏览器：chrome / msedge；留空则用 playwright 自带内核")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("需要先装 playwright：\n"
              "  pip install playwright && python -m playwright install chromium")
        return

    if args.discover:
        discover(args)
        return

    acc_file = Path(args.accounts)
    if not acc_file.exists():
        print(f"找不到 {acc_file}。建一个文本文件，每行一个 X 用户名（不带 @）。")
        return
    accounts = [x.strip().lstrip("@") for x in
                acc_file.read_text(encoding="utf-8").splitlines()
                if x.strip() and not x.startswith("#")]
    if not accounts:
        print("账号列表为空。")
        return

    name_map = load_name_map()
    cutoff = datetime.now() - timedelta(days=args.days)
    mentions: Counter[str] = Counter()
    samples: dict[str, str] = {}
    n_posts = 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        # macOS 13 上 playwright 1.6x 不再提供自带 chromium 内核，
        # 改用本机已安装的 Chrome（channel="chrome"）—— 顺带能复用你已有的登录态。
        profile_dir, extra = resolve_profile(args)
        launch_kw = {"headless": args.headless, "args": extra,
                     "viewport": {"width": 1280, "height": 900}}
        if args.channel:
            launch_kw["channel"] = args.channel
        try:
            ctx = pw.chromium.launch_persistent_context(profile_dir, **launch_kw)
        except Exception as e:
            print(f"启动浏览器失败: {type(e).__name__}: {str(e)[:140]}")
            if args.use_my_chrome:
                print("最常见原因：Chrome 还开着。完全退出 Chrome（⌘Q）后重试。")
            else:
                print("如果提示找不到 chrome，试试 --channel msedge。")
            return
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        page.goto("https://x.com/home", timeout=60000)
        page.wait_for_timeout(int(args.delay * 1000))
        if not logged_in(ctx, page):
            if not sys.stdin.isatty():
                print("未检测到 X 登录态，且当前不是交互终端 —— 直接退出。\n"
                      "请先在 Chrome 里登录 X，或去掉 --use-my-chrome 手动登录一次。")
                ctx.close()
                return
            print("需要登录。请在打开的浏览器里完成登录，然后回到这里按回车继续 ...")
            input()

        for acc in accounts:
            print(f"  抓取 @{acc} ...", flush=True)
            try:
                page.goto(f"https://x.com/{acc}", timeout=60000)
                page.wait_for_timeout(int(args.delay * 1000))
                for _ in range(args.max_scroll):
                    for art in page.query_selector_all("article"):
                        try:
                            txt = art.inner_text()
                        except Exception:                      # noqa: BLE001
                            continue
                        if not txt:
                            continue
                        te = art.query_selector("time")
                        dt = te.get_attribute("datetime") if te else None
                        if dt:
                            try:
                                if datetime.fromisoformat(dt.replace("Z", "+00:00")) \
                                        .replace(tzinfo=None) < cutoff:
                                    continue
                            except Exception:                  # noqa: BLE001
                                pass
                        n_posts += 1
                        for code in extract_symbols(txt, name_map):
                            mentions[code] += 1
                            samples.setdefault(code, f"@{acc}: {txt[:80]}")
                    page.mouse.wheel(0, 2400)
                    page.wait_for_timeout(int(args.delay * 1000))
            except Exception as e:                             # noqa: BLE001
                print(f"    ! @{acc} 失败: {type(e).__name__}: {str(e)[:70]}")

        ctx.close()

    today = datetime.now().strftime("%Y-%m-%d")
    rows = [{"date": today, "symbol": c, "count": n,
             "note": f"X提及{n}次 | {samples.get(c,'')[:70]}"}
            for c, n in mentions.most_common()]
    out = OUT_DIR / f"{today}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                   encoding="utf-8")
    print(f"\n扫描 {n_posts} 条推文，识别 {len(rows)} 只股票 -> {out}")
    if rows:
        print("提及最多：")
        for r in rows[:8]:
            print(f"  {r['symbol']}  {r['count']} 次")
    print("\n提醒：这些数据尚未经过任何验证，scan_weekly 只会把它作为参考信息"
          "写进上下文，并记账跟踪。不要据此直接下单。")


if __name__ == "__main__":
    main()
