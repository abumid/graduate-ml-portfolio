# Real-Time Futures Trading Bot (Backtesting + Testnet Execution)

This project showcases my engineering skills in quantitative finance: building a backtesting workflow for a rule-based futures strategy and connecting it to a real-time execution prototype using exchange APIs.

> **Disclaimer:** This is a research and skills-demonstration project using **Binance Futures testnet**. It is not financial advice.

## What this project demonstrates
- **Python engineering:** class-based design, reproducible notebook workflows
- **Market data pipelines:** historical data extraction + real-time streaming via WebSockets
- **Async programming:** event-driven processing using `asyncio`
- **Trading mechanics:** leverage settings, order placement, and basic execution flow (testnet)

## Notebooks
### 1) `backtesting.ipynb` — Futures Strategy Backtesting (Leverage & Costs)
Implements a research-style backtesting workflow to compare a rule-based long/short strategy vs a baseline.
- Vectorized return computation
- Transaction cost approximation
- Leverage sensitivity experiments and performance visualization

**Limitations:** bar-based execution assumptions; no slippage model; parameter exploration is for hypothesis generation.

### 2) `NB_02_Futures_API_Updated.ipynb` — Futures API + Real-Time Streaming (Testnet)
Demonstrates hands-on work with Binance Futures APIs:
- Account and trading-mode checks (e.g., margin type, leverage, positions)
- Historical data retrieval
- Real-time market data streaming via WebSockets (`BinanceSocketManager`)
- Minimal trader prototype (stream → signal → execute) on **testnet**

**Limitations:** simplified reliability layer (reconnect/idempotency not production-grade).

## How to run (high level)
- Open notebooks in Jupyter/VS Code
- Configure API keys for Binance Futures **testnet**
- Run notebooks top-to-bottom

## Next steps (planned)
- Walk-forward evaluation (time-split validation)
- Add risk controls (max position size, stop logic, kill switch)
- Add execution reliability (reconnect, retries, order idempotency)
