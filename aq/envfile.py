"""从项目根目录的 .env 加载 API key。

为什么不写进 ~/.zshrc：全局环境变量**任何进程都能读**，
包括你随手跑的第三方脚本和 npm 包。放在项目内的 .env（权限 600 + gitignore）
只有明确加载它的代码能看到，泄露面小得多。

已存在的环境变量优先 —— 命令行临时 export 能覆盖文件里的值，方便调试。
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load(path: Path | str | None = None, override: bool = False) -> list[str]:
    """加载 .env，返回本次实际设置的变量名列表。"""
    p = Path(path) if path else ENV_FILE
    if not p.exists():
        return []
    done = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k or not v:
            continue
        if override or not os.environ.get(k):
            os.environ[k] = v
            done.append(k)
    return done


def write(updates: dict[str, str], path: Path | str | None = None) -> None:
    """写回 .env，保留注释与未涉及的行。空值表示删除该项。"""
    p = Path(path) if path else ENV_FILE
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    seen = set()
    out = []
    for raw in lines:
        s = raw.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in updates:
                seen.add(k)
                v = updates[k]
                out.append(f"{k}={v}" if v else f"# {k}=")
                continue
        out.append(raw)
    for k, v in updates.items():
        if k not in seen and v:
            out.append(f"{k}={v}")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    p.chmod(0o600)          # 别让同机其他用户读到
    load(p, override=True)
