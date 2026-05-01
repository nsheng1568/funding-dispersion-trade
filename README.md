# Funding Dispersion Trade

**Market-neutral funding rate strategy on Hyperliquid perpetuals — running live with real capital.**
[View the live dashboard →](https://nsheng-funding-dispersion.streamlit.app/)

---

## A handwritten note from Nathan

I thought for a bit about a project that would be non-trivial and interesting, and that would showcase my expertise in overseeing the development of quantitative trading systems from research, to production, to monitoring. In addition to the tight time constraint, not having access to any production-grade tooling also posed a significant challenge. In particular, I had very limited access to real-time data for live trading or signal generation, to compute power for training sophisticated models, or to data infrastructure for ETLs.

Under these constraints, I felt the most tractable path to pursue was to develop a low-frequency trading strategy in Hyperliquid perpetuals, where historical and real-time data are available robustly and for free. At my previous role, I had always wanted to work on a market-neutral funding dispersion trade, but never got the resourcing for it. I summarize below how I spent my 3 days.

**Tuesday** — I gathered data and performed initial exploration and ideation. I settled on the following investment thesis: Perpetuals funding rates are fairly sticky, and therefore provide a straightforward source of edge, so long as you can control costs and price risk. Crypto prices are highly correlated and can be parsimoniously modeled as having some "crypto beta" plus some idiosyncratic noise. If we further assume that the idiosyncratic noise has mean zero for all assets, then we can harvest funding rates by hedging out crypto beta, leaving only idiosyncratic asset exposure.

**Wednesday** — I performed signals research and backtesting related to this investment thesis. Because of the low trading frequency and the short horizon of data, the backtest results were highly sensitive to inputs, and readings of returns/Sharpe/etc. were not terribly useful. However, I did manage to produce evidence that broadly supports my thesis — with extremely naive modeling and execution strategies, the backtests yielded around 7% funding APR net of transaction costs, at around 25% vol. (Actual backtest realized returns varied significantly around the 7% mark due to the high vol.) A highly unattractive risk-adjusted rate of return, to be sure, but nonetheless a reasonable baseline on which future research could build.

**Thursday** — I implemented the Hyperliquid exchange connectivity, strategy execution, and strategy monitoring frontend. Several pieces here leave much to be desired — alerting, error handling, pipeline robustness, and portfolio risk checks would all be necessary before calling this project truly rigorous and production-ready. Nonetheless, I did succeed in getting the trading strategy and monitoring frontend running live with actual capital, with a lightweight automated pipeline.

With more time, I would explore the following avenues, in this order:

1. Investigate more sophisticated portfolio construction techniques to allow for trading more than two assets at a time, with transaction-cost aware portfolio transitions.
2. Explore further modeling of crypto factor risks as well as funding rate evolution to improve edge.
3. Improve robustness of the trade execution stack.

---

## Architecture

```
Hyperliquid API ──► src/data/pipeline.py ──► data/*.parquet
CoinGecko API  ──┘                                │
                                                  ▼
                              src/models/betas.py        (PCA market factor, per-coin betas)
                              src/models/signal.py       (EWMA + OLS composite signal)
                                                  │
                                                  ▼
                              src/trading/portfolio.py   (beta-neutral order construction)
                              src/trading/executor.py    (Hyperliquid order execution)
                              src/trading/rebalance.py   (CLI: dry-run or live)
                                                  │
                          ┌───────────────────────┴──────────────────────┐
                          ▼                                               ▼
              pages/1_Strategy_State.py                    .github/workflows/etl.yml
              pages/2_Analytics.py                         scripts/etl.py
              (Streamlit dashboard)                        (daily snapshot cron, midnight UTC)
```

**Data flow:** The pipeline fetches historical funding rates and 8h OHLCV prices from Hyperliquid and stores them as parquet files. Beta estimation runs PCA on BTC/ETH/SOL returns to extract a market factor, then regresses each coin against it. The signal combines two EWMA half-lives (168h, 72h) and a rolling direct regression, risk-adjusted by per-coin idiosyncratic volatility. Portfolio construction picks the lowest-signal coin (long) and highest-signal coin (short), sizes beta-neutrally to a 40% annualized vol target, and generates a minimal order delta against the current position.

The daily ETL cron (GitHub Actions) snapshots equity, positions, and signals to `data/history/` CSVs, which are committed to the repo and read by the Streamlit dashboard. No private key is required for the ETL — only for live execution.

---

## Running it

### Prerequisites

```bash
pip install -r requirements.txt
cp .env.example .env       # add your Hyperliquid API wallet private key
```

`HL_PRIVATE_KEY` is only needed for live trade execution. The dashboard and ETL work without it.

### 1. Build the historical dataset (one-time)

```bash
python -m src.data.pipeline
```

Fetches funding rates and prices from Hyperliquid for the full universe (~30 coins). Takes several minutes due to API rate limits. Outputs `data/universe.parquet`, `data/funding_rates.parquet`, `data/prices.parquet`.

### 2. Estimate betas (re-run whenever you want to refresh)

```bash
python -m src.models.betas
```

Fits the PCA market factor and per-coin betas on the full price history. Outputs `data/coin_betas.parquet`.

### 3. Run the dashboard

```bash
streamlit run app.py
```

Opens the live monitoring dashboard on `localhost:8501`. Page 1 shows live positions and equity history; Page 2 shows signal rankings, beta summary, and signal history.

### 4. Rebalance (dry run first, then live)

```bash
python -m src.trading.rebalance           # prints the proposed order plan, no orders sent
python -m src.trading.rebalance --execute  # sends live orders to Hyperliquid
```

### 5. ETL snapshot (runs automatically via GitHub Actions)

```bash
python scripts/etl.py
```

Appends one row to each of the history CSVs (`equity.csv`, `positions.csv`, `signals.csv`, `target.csv`). The GitHub Actions workflow runs this daily at midnight UTC and commits the result.

### Research notebooks

The `notebooks/` directory contains the full research pipeline in order:

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Funding rate EDA — distributions, autocorrelation, seasonality |
| `02_beta_research.ipynb` | PCA market factor construction and beta estimation |
| `03_signal_research.ipynb` | EWMA half-life grid search and direct regression calibration |
| `04_backtest.ipynb` | OOS backtest with P&L decomposition, beta stability check, cost sensitivity |
