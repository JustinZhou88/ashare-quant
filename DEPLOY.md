# 部署到自己的服务器

## 一、把项目搬过去

`data_cache/`（约 1.5 GB）**不要传**——服务器上重新下载更快，也免得传坏。

```bash
# 本机：只打包代码和配置
tar --exclude='.venv' --exclude='data_cache' --exclude='journal' \
    --exclude='llm_cache' --exclude='__pycache__' --exclude='.x_profile*' \
    -czf aq.tar.gz ~/Desktop/ashare-quant

scp aq.tar.gz 你@服务器:~/
```

`journal/` 要不要传取决于你：**里面是历史推荐记录**，传了就能延续战绩统计，
不传等于从零开始。建议传——三个月的样本积累不该白费。

```bash
scp -r ~/Desktop/ashare-quant/journal 你@服务器:~/ashare-quant/
```

## 二、服务器上装环境

需要 **Python 3.11+**（3.14 实测可用）和 **Node.js**（只用于 `check_web.py` 的
JS 语法校验，不装也能跑，只是少一道检查）。

```bash
cd ~/ashare-quant
python3 -m venv .venv
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    pandas numpy requests flask pyyaml akshare

# akshare 的 mini_racer 是 20MB 二进制，网络不稳时单独装并允许续传
.venv/bin/pip install --resume-retries 10 mini-racer
```

**国内服务器如果挂了代理**：腾讯、东财、清华镜像都是国内站点，走代理反而更慢
更容易断。用 `env -u HTTP_PROXY -u HTTPS_PROXY <命令>` 绕开。

## 三、配置

```bash
cp .env.example .env 2>/dev/null || cat > .env <<'EOF'
GEMINI_API_KEY=你的key
# PANEL_TOKEN=对外访问时必须设置
EOF
chmod 600 .env
```

`.env` 权限必须是 600 —— 同机其他用户能读到就等于泄露。自检会检查这一条。

## 四、首次拉数据

```bash
.venv/bin/python download_bulk.py --workers 6      # 行情，约 70 分钟
.venv/bin/python download_margin.py --workers 6    # 两融，约 80 分钟
.venv/bin/python -m aq.data.altdata                # 研报/股东户数（可选）
```

跑完自检确认：

```bash
.venv/bin/python selfcheck.py
```

**16 项全绿再往下走。** 有红的先修，别带着已知问题上线。

## 五、启动面板

### 方案 A：SSH 隧道（推荐）

服务器上什么都不用改，保持只绑本机：

```bash
# 服务器
.venv/bin/python web_server.py --no-open

# 本机
ssh -L 5111:127.0.0.1:5111 你@服务器
# 然后浏览器开 http://127.0.0.1:5111
```

**零暴露面，不需要口令，也不用配防火墙。** 个人用这个最省事。

### 方案 B：对外提供服务

必须先设 `PANEL_TOKEN`，否则程序拒绝启动：

```bash
echo 'PANEL_TOKEN=一个足够长的随机口令' >> .env
.venv/bin/python web_server.py --host 0.0.0.0 --no-open
```

再配防火墙只放行你的 IP：

```bash
sudo ufw allow from 你的IP to any port 5111
```

⚠️ **口令是唯一的门。** `/api/run` 能在服务器上起子进程、`/api/settings` 能写文件——
口令泄露等于把服务器交出去。

### 开机自启（systemd）

```ini
# /etc/systemd/system/aq-panel.service
[Unit]
Description=A股量化面板
After=network.target

[Service]
User=你的用户名
WorkingDirectory=/home/你的用户名/ashare-quant
ExecStart=/home/你的用户名/ashare-quant/.venv/bin/python web_server.py --no-open
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now aq-panel
sudo journalctl -u aq-panel -f      # 看日志
```

## 六、每周例行

收盘后（**15:05 之后**，否则会拿到未完成的日线）：

```bash
.venv/bin/python update_data.py                          # 约 24 分钟
.venv/bin/python scan_weekly.py --universe universe_full.txt
.venv/bin/python scan_weekly.py --journal                # 补算前瞻收益
```

或者在面板「运行 → 任务与日志」里点按钮，日志实时滚。

### 自动化（可选）

```cron
# 每周五 15:30 更新数据并扫描
30 15 * * 5 cd ~/ashare-quant && .venv/bin/python update_data.py >> logs/cron.log 2>&1
0  16 * * 5 cd ~/ashare-quant && .venv/bin/python scan_weekly.py --universe universe_full.txt >> logs/cron.log 2>&1
```

先手动跑通再上 cron。cron 环境变量少，`.env` 会自动加载所以 key 没问题，
但 `PATH` 可能不同，命令一律写绝对路径。

## 七、这个系统当前的真实状态

| 层 | 证据 | 能信吗 |
|---|---|---|
| 因子层 | 2804 只全市场、6 折滚动前推、多重检验校正。样本外年化 +23.2%、夏普 0.89 | ✅ 有统计支撑 |
| 历史类比预测 | 纯统计，给分布不给点估计 | ✅ 口径可复核 |
| **LLM 层** | **零证据**。无法回测（模型语料含历史答案），只能前瞻记账 | ❌ **需约 30 个已到期样本，即三个月** |
| 外部情绪源 | Polymarket 实测与沪深300 同步相关仅 −0.025；股吧/X 尚无历史 | ❌ 仅记账观察 |

**在记账样本达到 30 之前，任何单笔盈亏都是噪声。** 面板首页那个
「LLM 层证据 n/30」的进度条就是为此存在的。

## 八、踩过的坑（改代码前先看）

这个项目出过四次**静默失效**——看起来一切正常、接口 200、日志干净，
但实际什么都没做。改任何东西之后，跑这两条：

```bash
.venv/bin/python selfcheck.py        # 16 项健康检查
.venv/bin/python check_web.py --api  # 前端 JS 语法 + 接口 JSON 合法性
```

具体的坑：

1. **`except Exception: break` 吞掉代码错误** —— 少了 `import json`，NameError
   被吞成「接口没数据」。数据管道里的异常处理必须打印类型和消息。
2. **缓存判断 off-by-one** —— 请求 `start=2017-01-01` 但缓存最早是 `2017-01-02`，
   判定失败导致每次重下 14 万条。发现「慢」时**先看 CPU 占用**：1% 说明在等网络。
3. **Python 字符串里的 `\\n` 注入进 JS** —— 变成字面量反斜杠-n，整个
   `<script>` 块语法错误，页面所有交互失效，但服务端完全正常。
4. **pandas 的 NaN 经 jsonify 输出裸 `NaN`** —— 不是合法 JSON，
   浏览器 `JSON.parse` 抛异常 → 页面空白。所有 API 出口都要过 `clean()`。

还有两个 A股特有的数据坑：

5. **腾讯行情对科创板返回「股」不是「手」** —— 成交额被放大 100 倍，
   流动性过滤形同虚设。验证方法：用 `流通市值/价格` 反推流通股数算隐含换手率，
   超过 100% 就是单位错了。
6. **后复权价和不复权价不能混用** —— 股数和显示价用不复权（真实成交价），
   收益率和止损判定用后复权（含分红）。混用会导致股数算错、止损在错误位置触发。
