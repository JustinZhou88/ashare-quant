"""校验 web_server.py 里内嵌的前端代码 —— 每次改完面板必跑。

为什么需要这个：面板的 HTML/JS 是以 Python 字符串形式内嵌的，
Python 的转义和 JS 的转义会互相干扰。踩过一次真实的坑：
在 Python 里写 `"\\n"` 本意是换行，实际产生了字面的反斜杠-n，
注入 JS 后是语法错误，**整个 <script> 块失效，页面所有交互全死**，
但服务端一切正常、接口全 200 —— 从后端完全看不出问题。

肉眼审查抓不住这类错误，必须让解析器来判。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent / "web_server.py"


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    # 必须检查**所有** script 块。踩过的坑：加了登录页之后，
    # 只取第一个块会检到登录页那 300 字符，主面板 24000 字符的 JS 完全没被检查。
    blocks = re.findall(r"<script>(.*?)</script>", src, re.S)
    if not blocks:
        print("✗ 没找到 <script> 块")
        return 1
    js = "\n;\n".join(blocks)
    print(f"  找到 {len(blocks)} 个 script 块")

    node = shutil.which("node")
    if not node:
        print("! 未安装 node，跳过语法解析（强烈建议装一个：brew install node）")
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        # 顶层有 await 之类的话按模块解析；这里是普通脚本，直接 --check
        f.write(js)
        tmp = f.name
    try:
        r = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
    finally:
        Path(tmp).unlink(missing_ok=True)

    if r.returncode == 0:
        n_fn = len(re.findall(r"function\s+\w+|=>", js))
        print(f"✓ 前端 JS 语法通过（{len(js):,} 字符，{n_fn} 个函数/箭头）")
        return 0

    print("✗ 前端 JS 语法错误 —— 页面所有交互都会失效：")
    for ln in (r.stderr or "").splitlines()[:12]:
        print("   ", ln)
    return 1




def check_apis(port: int = 5111) -> int:
    """校验所有接口返回的是**合法 JSON**。

    为什么单独检查：pandas 的 NaN 经 jsonify 会输出裸 `NaN`，
    那不是合法 JSON。浏览器 JSON.parse 抛异常 -> 页面空白，
    但接口返回 200、后端日志干干净净 —— 从服务端完全看不出问题。
    实测「总览」和「战绩记账」两页就是这么挂掉的。
    """
    import json as _json
    import urllib.request

    paths = ["/api/status", "/api/picks", "/api/journal", "/api/candidates",
             "/api/factors", "/api/settings"]
    bad = 0
    for p in paths:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{p}", timeout=90) as r:
                raw = r.read().decode("utf-8")
        except Exception as e:                                 # noqa: BLE001
            print(f"  ✗ {p}  请求失败 {type(e).__name__}")
            bad += 1
            continue
        try:
            _json.loads(raw)                 # 严格模式：不接受 NaN/Infinity
            print(f"  ✓ {p}  {len(raw):>8,} 字节")
        except ValueError as e:
            tokens = [t for t in ("NaN", "Infinity", "-Infinity") if t in raw]
            print(f"  ✗ {p}  非法 JSON：{e}"
                  + (f"（含 {', '.join(tokens)}）" if tokens else ""))
            bad += 1
    return bad


if __name__ == "__main__":
    rc = main()
    if "--api" in sys.argv:
        print("\n接口 JSON 合法性:")
        rc += check_apis()
    sys.exit(rc)
