# A-Share Strategy Research Pipeline (ashare-quant)

A production-grade quantitative research framework and pipeline designed specifically for China's A-share market, implementing the complete cycle: **"AI Batch Generation → Backtesting → Trade Diagnostics → Walk-Forward Validation → Iteration"**.

Conventional toolchains like TradingView + Pine Script fail on the A-share market due to unique market mechanics: **T+1 settlement, 10%/20% daily price limits, one-sided stamp duty, and lack of short-selling mechanisms**. This project embeds these rules directly into a vectorized backtesting engine and adds the most critical missing link in retail quant research: **multiple testing correction and statistical p-hacking controls**.

---

## 🎯 Why Multiple Testing Correction Is Critical

Traditional quant workflows often claim: *"We backtested 432 strategies and picked the most profitable one."* That sentence is the textbook definition of **data snooping and overfitting**.

Searching across 400 strategy candidates on the same historical slice and selecting the maximum will yield an annualized Sharpe Ratio of **2.5 to 3.5 even if all 400 strategies are pure random Gaussian noise**. What looks like a holy grail is simply the maximum of white noise.

In this pipeline, the top-ranked strategy on the leaderboard is **not a conclusion, but an unverified hypothesis**. It must pass three statistical hurdles:

| Validation Test | Question Answered | Implementation |
|---|---|---|
| **Deflated Sharpe Ratio (DSR)** | After discounting the selection bias of searching 424 candidates, is the Sharpe ratio still statistically significant? | `aq/validation/deflated.py` |
| **Walk-Forward Analysis (WFA)** | Does selecting strategies based on in-sample performance possess any predictive out-of-sample power? | `aq/validation/wfa.py` |
| **Random Benchmark** | Given identical market exposure and turnover, what performance would pure random coin-flipping achieve? | `aq/validation/random_bench.py` |

The most informative metric from WFA is the **In-Sample / Out-of-Sample Sharpe Rank Correlation**. If this correlation stays near zero over time, your entire screening pipeline is selecting noise — a signal that you need new independent information sources rather than tweaking parameters.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd ~/Desktop/ashare-quant
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Download Historical Market Data
```bash
.venv/bin/python download.py --n 120 --start 2015-01-01
```

### 3. Run Strategy Screening & Multi-Testing Validation
```bash
.venv/bin/python run_screen.py --start 2015-01-01
```
Upon completion, review results in `results/report.md`.

---

## 🔄 Four-Step Workflow Architecture

### ① Strategy Harvesting → `aq/strategies/`
- **15 Strategy Families**: Moving averages, channel breakouts, Donchian, Turtle, Bollinger Bands, RSI, KDJ, MACD, Volume-Price, ATR Chandelier, Trend Pullback, etc.
- **Parameter Grid**: Evaluates **424 candidate strategies** across the asset universe.
- **Adding Custom Strategies**: Define standard vector functions in `library.py` registered into `FAMILIES`, then register parameter grids in `grid.py`:
  ```python
  def my_strategy(P, n=20, k=2.0) -> pd.DataFrame:   # Returns binary 0/1 position matrix
      """Return value represents target position held at Day t close. Engine executes at Day t+1 open."""
  ```
- Vectorized across all stocks simultaneously on `DataFrame` (Date × Symbol) without per-symbol loops.

### ② A-Share Compliant Backtesting Engine → `aq/engine/`
Every market rule is covered by rigorous automated unit tests (`tests/test_engine.py`, 21 test suites):
- **T Day Close Signal → T+1 Day Open Execution**: Avoids look-ahead bias by never executing trades at the signal close price.
- **T+1 Settlement**: Stocks bought on day T cannot be sold until day T+1.
- **Limit Up / Down Execution Constraints**: Buys at opening limit-up and sells at opening limit-down are rejected (evaluated using unadjusted non-restated prices).
- **Suspension Handling**: Suspended tickers cannot be traded; existing positions remain frozen.
- **Realistic Friction & Transaction Costs**:
  - Brokerage commission: 0.025% (bid/ask)
  - Stamp duty: 0.05% (one-sided on sell)
  - Transfer fee: 0.001% (bid/ask)
  - Slippage: 0.05% (bid/ask)
  - Total round-trip drag: **~20 basis points (0.20%)**.
- **Tiered Price Limits by Board**: Main Board (10%), ChiNext & STAR Market (20%), ST (5%), BSE (30%).
- **Stop-Loss Modeling**: Evaluated against close price, executed at next open (intraday stop-losses on daily bars are mathematically unrealistic).
- **Dual Price Tracking**: Backward-adjusted prices for return calculations; unadjusted raw prices for limit-up/down boundary checks.

### ③ Trade Attribution & Diagnostics → `aq/analysis.py`
Exports detailed trade logs (`results/trades_best.csv`): entry/exit prices, holding days, Maximum Favorable Excursion (MFE), Maximum Adverse Excursion (MAE), market regime tags, and benchmark excess return.

Automated reporting diagnostics:
- **Profit Concentration**: Contribution percentage from the top 5 winning trades (is the strategy profitable without them?).
- **Market Regime Partitioning**: Performance grid across Bull / Range / Bear markets × High / Medium / Low volatility.
- **Yearly Return Distribution**: Verifies if returns were concentrated entirely in outlier bull years (e.g. 2015 or 2020).
- **MAE / MFE Distribution**: Guides optimal stop-loss placement and identifies unrealized profit give-backs.
- **Parameter Robustness**: Identifies whether parameter spaces form smooth plateaus (robust) or isolated needle peaks (overfitted).

### Integrating External Signals (LLM Multi-Agent / Alternative Data)
This system functions as an **impartial referee**. Any signal provider (multi-agent LLM systems, factor models, or external scrapers) that outputs a daily holding matrix can be tested under identical rules:
```bash
.venv/bin/python run_screen.py --external signals/ --knowledge-cutoff 2025-01-01
```

> [!WARNING]
> **Look-Ahead Bias in LLM Backtesting**:
> When backtesting LLMs on historical periods, the future is already embedded inside the model's weights. Asking an LLM to make decisions on 2020 news suffers from implicit training contamination (the model knows which companies survived or rebounded).
> This contamination cannot be cured by Walk-Forward Analysis. The only scientifically sound approach is testing LLM agents **exclusively after the model's knowledge cutoff date**. The `--knowledge-cutoff` flag warns when signal dates violate this boundary.

### ④ Iteration & Parameter Tuning
```bash
.venv/bin/python run_screen.py --start 2015-01-01 --stop-loss 0.08 --top-k 5
```

Key CLI Flags:
| Parameter | Description |
|---|---|
| `--start / --end` | Backtest date range. |
| `--min-amount` | Liquidity cutoff (average daily turnover), default: 30,000,000 RMB. |
| `--is-years / --oos-years` | In-sample / Out-of-sample window sizes for Walk-Forward Analysis. |
| `--top-k` | Selects equal-weight ensemble of top K strategies per fold (>1 improves stability). |
| `--max-positions` | Maximum concurrent positions (1/K allocation each), default: 10. |
| `--stop-loss` | Fixed stop-loss threshold (e.g., `0.08`). |
| `--no-random` | Skips Monte Carlo random benchmark to speed up runs. |

---

## ⚠️ Known Biases & Inherent Limitations

1. **Survivorship Bias (Primary Limitation)**: Public free data sources generally only contain currently active listings; delisted companies are excluded, which systematically inflates long-term historical returns.
2. **Simplified Capital Allocation**: Fixed 1/K position sizing for up to K holdings (default 10) with idle capital held as uninvested cash without interest.
3. **Daily Bar Granularity**: Does not simulate order book depth, tick-level slippage, or queue priority at limit boards. Unsuitable for HFT or intraday strategies.
4. **Long-Only**: Short selling is restricted for typical retail accounts; short positions are disabled in the engine.
5. **Lot Sizing & Minimum Commission**: Standard 100-share board lot rounding and 5 RMB minimum commission floor are omitted (negligible for capital > 100k RMB and daily turnover > 30M RMB).

---

## 📂 Repository Structure

```text
├── aq/
│   ├── config.py              # A-share transaction costs, taxes, and exchange rules
│   ├── data/
│   │   ├── loader.py          # Market data downloader & caching
│   │   ├── universe.py        # Stock pool generation & survivorship documentation
│   │   ├── altdata.py         # Margin trading, research reports, shareholder data
│   │   ├── guba.py            # East Money sentiment crawler & parser
│   │   └── grok_x.py          # Social media sentiment integration
│   ├── engine/
│   │   ├── panel.py           # Vectorized A-share backtesting engine
│   │   └── metrics.py         # Performance, risk, and overfitting metrics
│   ├── strategies/            # Technical indicators, 15 strategy families, parameter grids
│   ├── validation/
│   │   ├── deflated.py        # Deflated Sharpe Ratio (DSR) multiple-testing correction
│   │   ├── wfa.py             # Walk-Forward Analysis (WFA) rolling engine
│   │   └── random_bench.py    # Monte Carlo random trade benchmark
│   └── agents/                # LLM multi-agent analysts (macro, industry, technical, risk)
├── tools/                     # Data scraping and external API adapters
├── tests/                     # Test suite validating A-share engine mechanics (21 tests)
├── journal/                   # Live paper-trading recommendation logs and audit history
├── results/                   # Backtest reports, leaderboards, and trade performance CSVs
├── results_factor/            # Multi-factor research and ensemble reports
├── data_cache/                # Cached historical A-share daily K-lines and alternative data
├── web_server.py              # Real-time web dashboard backend (Flask / Werkzeug)
├── dashboard.html             # Interactive quant dashboard frontend
├── scan_weekly.py             # Automated weekly factor & sentiment screening runner
├── update_data.py             # Incremental daily market data updater
└── selfcheck.py               # 16-point automated system health check
```

---

## 📜 Disclaimer

This project is a **quantitative research and educational tool**, not financial advice or an investment recommendation. Backtest results do not guarantee future returns. The author is not liable for any financial losses incurred from trading decisions based on this code.

*If you test 424 strategies and discover none pass the Deflated Sharpe or WFA tests — that is a completely honest and realistic outcome. Public technical indicators are widely traded and their edge has largely decayed. Knowing that with statistical rigor is the core value of this framework.*
