# A股量化系统 · 部署包

解压后按 `DEPLOY.md` 操作。这里是三分钟速通版。

## 包里有什么

| 内容 | 说明 |
|---|---|
| 代码 | `aq/`（核心库）、根目录各 `.py` 脚本、`tools/`（爬虫工具） |
| **行情数据** | `data_cache/` 2807 只 × 2016 年至今，已含 2026-08-04 |
| **另类数据** | 两融 2520 只、研报 14.6 万条、股东户数 17.4 万条 |
| **推荐记录** | `journal/` —— 你的战绩起点，别丢 |
| 配置 | `rules.yaml`（你的交易规则）、`.env.example` |
| 文档 | `DEPLOY.md` 部署详解、`README.md` 项目说明 |

| **`.env`（含 API key）** | 按你的要求已包含，权限 600 |

**不含 `.venv`** —— 虚拟环境的二进制不跨机器，必须在服务器上重建。

⚠️ **这个压缩包里有你的 API key。** 别转发、别上传到公开仓库、别放网盘共享目录。
传完删掉本地副本：`rm ~/Desktop/ashare-quant-deploy.tar.gz`

## 三步启动

```bash
tar -xzf ashare-quant-*.tar.gz && cd ashare-quant

python3 -m venv .venv
.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    pandas numpy requests flask pyyaml akshare
.venv/bin/pip install --resume-retries 10 mini-racer

chmod 600 .env      # 包里已含 key，只需确认权限

.venv/bin/python selfcheck.py        # 必须 16/16 通过
.venv/bin/python web_server.py --no-open
```

本机再开一个终端做隧道，然后浏览器访问 `http://127.0.0.1:5111`：

```bash
ssh -L 5111:127.0.0.1:5111 你@服务器
```

## 数据新鲜度

打包时数据截至 **2026-08-04**。部署后第一件事是补上这期间的行情：

```bash
.venv/bin/python update_data.py
```

隔得越久跑得越慢，但增量逻辑会自动处理，不用担心重复或缺失。

## 一句话现状

**因子层有统计支撑**（2804 只全市场、6 折前推，样本外年化 +23.2%、夏普 0.89）；
**LLM 层零证据**，记账才 2/30 个样本。在进度条走到 30 之前，
任何单笔盈亏都是噪声——这也是面板首页那个进度条存在的意义。
