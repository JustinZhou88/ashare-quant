"""LLM 客户端（多 provider：Gemini / DeepSeek / 智谱 / 通义）。

**没有付费 API 也能用**：Gemini 在 AI Studio 有永久免费额度
（1500 次/天，无需信用卡），而每周扫描只用 6 次调用 —— 免费额度是需求的 250 倍。
智谱 glm-4-flash 同样免费。所以不必去自动化网页版 AI。

三件必须内建的事：
  1. **磁盘缓存** —— 同一个 prompt 不重复付费。回测/调试时反复跑同一周，
     没有缓存会烧掉大量 token。
  2. **成本统计** —— 每次扫描花了多少钱要能看见。看不见的成本会失控。
  3. **dry-run** —— 没有 API key 也能把整条流水线跑通，
     这样你可以先确认信息包和流程对不对，再决定花钱。

不用 openai SDK，直接 requests —— 少一个依赖，也方便看清发出去的到底是什么。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ..envfile import load as _load_env

# 一进来就加载 .env —— 这样 CLI、web 面板、任何入口都能拿到 key，
# 不用each脚本各写一遍。已存在的环境变量优先，方便临时覆盖调试。
_load_env()

CACHE_DIR = Path("llm_cache")

# 计费（元/百万 token），仅用于**估算**。Gemini 免费额度内成本为 0。
PRICE_IN = 2.0
PRICE_OUT = 8.0


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def cost_cny(self) -> float:
        return self.tokens_in / 1e6 * PRICE_IN + self.tokens_out / 1e6 * PRICE_OUT

    def report(self) -> str:
        return (f"LLM 调用 {self.calls} 次（命中缓存 {self.cached} 次），"
                f"输入 {self.tokens_in:,} tok，输出 {self.tokens_out:,} tok，"
                f"估算成本 ¥{self.cost_cny:.2f}")


# 各家配置。Gemini 是**永久免费**的（AI Studio 拿 key，无需信用卡），
# 免费额度 1500 次/天，而每周扫描只用 6 次 —— 对本项目完全够用。
PROVIDERS = {
    "gemini": {
        "env": "GEMINI_API_KEY",
        "model": "gemini-3.5-flash",   # 实测可用；gemini-3-flash 不在免费列表里
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "format": "gemini",
        "signup": "https://aistudio.google.com （免费，无需信用卡）",
        "deep_model": "gemini-3.5-flash",   # 辩论与最终决策用更强的模型
    },
    "xai": {
        "env": "XAI_API_KEY",
        "model": "grok-4.5",
        "url": "https://api.x.ai/v1/chat/completions",
        "format": "openai",
        "signup": "https://console.x.ai （付费；独有优势是能实时检索 X）",
    },
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "format": "openai",
        "signup": "https://platform.deepseek.com （需充值，但每周扫描约 ¥0.15）",
    },
    "zhipu": {
        "env": "ZHIPU_API_KEY",
        "model": "glm-4-flash",
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "format": "openai",
        "signup": "https://open.bigmodel.cn （glm-4-flash 免费）",
    },
    "qwen": {
        "env": "DASHSCOPE_API_KEY",
        "model": "qwen-plus",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "format": "openai",
        "signup": "https://bailian.console.aliyun.com （新用户有免费额度）",
    },
}


def detect_provider() -> str | None:
    """按环境变量自动选一个可用的 provider。"""
    for name, cfg in PROVIDERS.items():
        if os.environ.get(cfg["env"]):
            return name
    return None


@dataclass
class LLM:
    provider: str | None = None       # None = 自动探测环境变量
    model: str | None = None
    api_key: str | None = None
    temperature: float = 0.3
    timeout: float = 180.0
    dry_run: bool = False
    usage: Usage = field(default_factory=Usage)
    base_url: str = ""
    fmt: str = "openai"

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = detect_provider() or "gemini"
        cfg = PROVIDERS.get(self.provider)
        if cfg is None:
            raise ValueError(f"未知 provider: {self.provider}，可选 {list(PROVIDERS)}")
        self.model = self.model or cfg["model"]
        self.fmt = cfg["format"]
        self.base_url = cfg["url"].format(model=self.model)
        if self.api_key is None:
            self.api_key = os.environ.get(cfg["env"])
        if not self.api_key and not self.dry_run:
            self.dry_run = True
        CACHE_DIR.mkdir(exist_ok=True)

    def missing_key_hint(self) -> str:
        lines = ["未检测到任何 API key。任选一家配置环境变量即可："]
        for name, c in PROVIDERS.items():
            lines.append(f"  export {c['env']}=...    # {name}: {c['signup']}")
        return "\n".join(lines)

    # ------------------------------------------------------------ 缓存
    def _key(self, system: str, user: str) -> str:
        h = hashlib.sha256()
        h.update(f"{self.model}|{self.temperature}|{system}|{user}".encode())
        return h.hexdigest()[:32]

    def _cached(self, k: str) -> str | None:
        f = CACHE_DIR / f"{k}.txt"
        return f.read_text(encoding="utf-8") if f.exists() else None

    def _store(self, k: str, v: str) -> None:
        (CACHE_DIR / f"{k}.txt").write_text(v, encoding="utf-8")

    # ------------------------------------------------------------ 调用
    def chat(self, system: str, user: str, use_cache: bool = True) -> str:
        k = self._key(system, user)
        if use_cache:
            hit = self._cached(k)
            if hit is not None:
                self.usage.cached += 1
                return hit

        if self.dry_run:
            out = _stub_response(system, user)
            self._store(k, out)
            return out

        if self.fmt == "gemini":
            url = f"{self.base_url}?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": self.temperature},
            }
        else:
            url = self.base_url
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": self.temperature,
            }

        last = None
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout,
                                  headers=headers)
                r.raise_for_status()
                js = r.json()
                if self.fmt == "gemini":
                    out = js["candidates"][0]["content"]["parts"][0]["text"]
                    u = js.get("usageMetadata") or {}
                    ti = int(u.get("promptTokenCount", 0))
                    to = int(u.get("candidatesTokenCount", 0))
                else:
                    out = js["choices"][0]["message"]["content"]
                    u = js.get("usage") or {}
                    ti = int(u.get("prompt_tokens", 0))
                    to = int(u.get("completion_tokens", 0))
                self.usage.calls += 1
                self.usage.tokens_in += ti
                self.usage.tokens_out += to
                self._store(k, out)
                return out
            except Exception as e:                             # noqa: BLE001
                last = e
                # 429 = 触发速率限制，退避久一点（免费额度常见）
                sleep = 10 if ("429" in str(last) or "503" in str(last)) else 2 * (attempt + 1)
                time.sleep(sleep)
        raise RuntimeError(f"{self.provider} 调用失败: "
                           f"{type(last).__name__}: {str(last)[:150]}")


def parse_json(text: str) -> dict | list | None:
    """从模型回复里抠出 JSON。模型经常裹在 ```json ``` 里或前后带解释。"""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith(("{", "[")):
                try:
                    return json.loads(p)
                except Exception:                              # noqa: BLE001
                    continue
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:                                  # noqa: BLE001
                pass
    return None


def _stub_response(system: str, user: str) -> str:
    """dry-run 占位回复：结构与真实输出一致，内容标注为占位。

    故意**不做任何真实判断** —— 免得你在 dry-run 下看到"推荐"就当真。
    """
    codes = []
    for line in user.splitlines():
        if line.startswith("### "):
            parts = line[4:].split()
            if parts:
                codes.append(parts[0])
    codes = codes[:5] or ["000000"]
    if "JSON" not in system and "json" not in system:
        return "[DRY-RUN 占位输出] 未配置任何 API key，未做真实分析。"
    return json.dumps({
        "picks": [{"symbol": c, "score": 0, "reason": "[DRY-RUN] 未做真实分析",
                   "risks": ["未配置 API key"], "entry_note": "-"} for c in codes[:3]],
        "note": "[DRY-RUN] 这是占位输出，不是投资判断。配置 GEMINI_API_KEY（免费）后重跑。",
    }, ensure_ascii=False, indent=2)
