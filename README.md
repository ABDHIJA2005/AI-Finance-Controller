# AI Finance Controller & QuantMarket Engine

**AI Finance Controller & QuantMarket** is an end-to-end financial engineering and quantitative research platform combining:
1. **AI Finance Controller (`src/reconciliation/`)**: An autonomous, explainable multi-source financial reconciliation engine that resolves discrepancies across Invoices, Accounting General Ledgers, Payment Gateways, and Bank Feeds using deterministic policy gating, tool-augmented LLM review, and zero-hallucination guardrails.
2. **QuantMarket Engine (`src/`)**: An empirical quantitative framework for systematic strategy formulation, technical feature engineering, friction-aware backtesting ($T \to T+1$ execution, slippage, brokerage fees), out-of-sample partition testing, and Monte Carlo bootstrap risk analysis.

---

## 1. Project Overview & Objectives

The platform provides an empirical, mathematically sound platform bridging systematic quantitative trading research and autonomous corporate financial operations:
- **Autonomous Financial Operations**: Multi-source reconciliation pipeline with transparent audit trails, explainable decision factors, and automated exception routing.
- **Empirical Market Signal Research**: Identification and statistical validation of market inefficiencies without data snooping or curve fitting.
- **Friction-Aware Simulation**: Chronological backtesting strictly enforcing zero look-ahead bias and realistic market execution costs.
- **Granular Risk & Robustness Analytics**: Sortino, Calmar, drawdown recovery dynamics, and 5,000-path Monte Carlo bootstrap simulations.

---

## 2. System Architecture

```text
AI-Finance-Controller / quantmarket/
├── data/               # Raw, processed, and cached market datasets & reconciliation feeds
├── src/                # Core production Python source code
│   ├── strategies/     # Quantitative trading strategies (Breakout, Momentum, MA Crossover)
│   └── reconciliation/ # AI Finance Controller multi-source reconciliation engine & tools
├── dashboard/          # Interactive operations console (HTML5 / Vanilla JS / FastAPI)
├── tests/              # Comprehensive test suite covering all modules (104 Pytest cases)
├── notebooks/          # Exploratory Data Analysis (EDA) & research notebooks
├── scripts/            # Validation runners for backtests, signals, and Monte Carlo
├── README.md           # Project documentation and architecture guide
├── requirements.txt    # Managed project dependencies
├── .gitignore          # Git exclusion rules
└── AGENTS.md           # Engineering guidelines, statistical rigor, and research standards
```

---

## 3. Planned Core Modules

1. **Data Ingestion & Hygiene (`src/data/`)**:
   - Automated ingestion of historical price, volume, and market data from verified APIs.
   - Point-in-time data cleaning, missing value handling, and timestamp alignment.

2. **Signal & Feature Engineering (`src/signals/` & `src/strategies/`)**:
   - Quantitative feature extraction (trend, momentum, volatility, mean-reversion, statistical arbitrage).
   - Point-in-time indicator calculation strictly using information available prior to time $t$.

3. **Backtesting Simulation Engine (`src/backtest/`)**:
   - Chronological simulation engine supporting vectorized and event-driven backtesting.
   - Execution modeling incorporating transaction costs, slippage, and position limits.

4. **Risk & Performance Analytics (`src/risk/`)**:
   - Standard quantitative performance metrics: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown.
   - Statistical risk metrics: Value-at-Risk (VaR), Expected Shortfall (CVaR), exposure metrics, and stress-testing.

5. **Interactive Operations Console (`dashboard/`)**:
   - Web console powered by FastAPI and modern HTML5/CSS for real-time audit trail analysis, match/exception routing, and factor breakdown.

6. **AI Finance Controller (`src/reconciliation/`)**:
   - Multi-source transaction reconciliation (Invoice, General Ledger, Payment Gateway, Bank) with tool-based AI review and strict zero-hallucination validation.

---

## 4. Emphasis on Statistical Rigor & Reproducibility

- **Zero Look-Ahead Bias**: Signals and indicators strictly utilize historical data known at or prior to execution time.
- **Chronological Validation**: Data splits maintain strict time-series ordering without shuffling or future leakage.
- **Empirical Authenticity**: All research is conducted exclusively on verified, reproducible empirical market data without synthetic fabrication or cherry-picked backtest periods.
- **Friction-Aware Performance**: Strategies are benchmarked after accounting for transaction fees, slippage, and execution constraints.

---

## 5. Data Pipeline

The QuantMarket historical market data pipeline (`src/data_loader.py`, `src/data_cleaner.py`, `src/storage.py`) provides reproducible, point-in-time data ingestion, non-destructive hygiene validation, and high-performance Parquet storage.

### Data Source & Instrument
- **Primary Public Data Provider**: Yahoo Finance (`yfinance` API).
- **Target Instrument**: NIFTY 50 Index (Yahoo Finance ticker: `^NSEI`).
- **Sampling Interval**: Configurable (defaults to daily `'1d'`).
- **Date Range**: Fully configurable (`start_date` and `end_date`), supporting multi-year historical periods up to the current available date.

### Standardized OHLCV Schema
- `open` *(float)*: Opening price level.
- `high` *(float)*: Period high price level.
- `low` *(float)*: Period low price level.
- `close` *(float)*: Period closing price level.
- `adj_close` *(float)*: Adjusted closing price level.
- `volume` *(int/float)*: Trading volume. Treated as **optional** for index instruments like NIFTY 50 (`^NSEI`), where volume may be zero or unrecorded.

### Raw vs. Processed Storage Architecture
Data is segregated into two isolated storage tiers in Apache Parquet format:
- `data/raw/`: Preserves exact downloaded time-series as received from yfinance.
- `data/processed/`: Contains point-in-time validated and cleaned datasets ready for quantitative analysis.

Files are named deterministically:
`<SANITIZED_SYMBOL>_<START_DATE>_<END_DATE>_<INTERVAL>.parquet`
*(Example: `NSEI_20150101_20260825_1d.parquet`)*

### Data Hygiene & Validation Rules
1. **Non-Empty Check**: Verifies downloaded DataFrame contains valid observations.
2. **Chronological Ordering**: Enforces strictly monotonic increasing DatetimeIndex.
3. **Duplicate Timestamp Hygiene**: Identifies duplicate timestamps and retains first occurrence.
4. **Price Positivity**: Enforces positive price bounds ($Open, High, Low, Close > 0$).
5. **Logical OHLC Bounds**: Validates $High \ge \max(Open, Close)$, $Low \le \min(Open, Close)$, and $High \ge Low$.
6. **Zero Data Fabrication Policy**: Missing or invalid records are logged and removed; data is **never** artificially forward-filled or replaced with synthetic values without explicit quantitative justification.

### Known Limitations of Data Source
- **Index Volume**: Yahoo Finance often reports `volume = 0` for index tickers such as `^NSEI`. Volume-dependent indicators are kept separate from core price signals.
- **API Dependencies**: Yahoo Finance endpoints are subject to rate limiting, network latency, and structural changes.
- **Market Adjustments**: Index historical prices reflect benchmark index levels rather than individual stock corporate action adjustments.

> [!NOTE]
> **Research Disclaimer**: QuantMarket is designed strictly for quantitative research and educational evaluation. Historical data quality from public endpoints does not guarantee live-market execution accuracy.

---

## 6. Feature Engineering

The feature engineering layer (`src/features.py`) provides point-in-time calculation of technical market indicators and returns with strict zero look-ahead bias.

### Implemented Technical Features
- **Daily Simple Returns (`return`)**:
  $$r_t = \frac{\text{Close}_t}{\text{Close}_{t-1}} - 1$$
- **Log Returns (`log_return`)**:
  $$\text{log\_r}_t = \ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right)$$
- **Simple Moving Average (`sma_20`, `sma_50`, `sma_200`)**:
  $$\text{SMA}_{N, t} = \frac{1}{N} \sum_{i=0}^{N-1} \text{Close}_{t-i}$$
- **Exponential Moving Average (`ema_20`, `ema_50`)**:
  $$\text{EMA}_{N, t} = \alpha \cdot \text{Close}_t + (1 - \alpha) \cdot \text{EMA}_{N, t-1}, \quad \alpha = \frac{2}{N+1}$$
- **Annualized Rolling Volatility (`volatility_20`, `volatility_50`)**:
  $$\sigma_{N, t} = \text{std}(r_{t-N+1:t}) \times \sqrt{252}$$
  *(The factor $\sqrt{252}$ represents the standard annualization multiplier assuming 252 trading days per calendar year).*
- **Relative Strength Index (`rsi_14`)**:
  Wilder's RSI bounded strictly in $[0, 100]$, calculated via exponential moving averages of gains and losses with smoothing factor $\alpha = \frac{1}{14}$.
- **Average True Range (`atr_14`)**:
  Wilder's smoothing of True Range ($TR_t = \max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)$).
- **Rolling High/Low (`rolling_high_20`, `rolling_low_20`)**:
  Rolling 20-period maximum and minimum close prices including current observation $t$.
- **Shifted Previous-Window High/Low (`previous_high_20`, `previous_low_20`)**:
  Rolling 20-period maximum and minimum close prices shifted by 1 period (`.shift(1)`), representing $\max(\text{Close}_{t-20} \dots \text{Close}_{t-1})$ strictly **excluding** current observation $t$.

### Prevention of Look-Ahead Bias
All features at time $t$ strictly utilize information known at or before time $t$. For breakout strategies, `previous_high_20` and `previous_low_20` explicitly apply `.shift(1)` so that today's price level ($\text{Close}_t$) cannot influence the historical breakout threshold.

### Initial Rolling Period Behavior
Rolling window calculations (e.g., SMA-200, Volatility-50, RSI-14) naturally produce initial `NaN` values for the first $N-1$ periods. These missing values reflect the required historical lookback and are preserved without artificial zero-filling.

> [!IMPORTANT]
> **Analytical Features vs. Signals**: All computed metrics are descriptive statistical and analytical features. They do not constitute automated trading signals or predictive forecast outputs until processed by strategy modules.

---

## 7. Signal Engine

The signal engine (`src/signal_engine.py` and `src/strategies/`) evaluates whether predefined market conditions are satisfied at time $t$ based strictly on information available on or before time $t$.

### Signal vs. Trade Execution
A **market signal** indicates whether an analytical condition is satisfied ($1$ for long condition, $0$ for flat). A signal is **NOT** a trade execution or backtest. Execution timing, slippage, order mechanics, and PnL accounting belong exclusively to the backtesting engine.

### Standardized Signal Conventions
1. **Position State (`position_state`)**:
   - `1.0`: Currently Long (condition satisfied).
   - `0.0`: Currently Flat (exit condition satisfied or out of position).
   - `NaN`: Initial warmup period (insufficient historical data).
2. **Signal Event (`signal_event`)**:
   - `1.0`: **New Entry Event** (transition from flat `0` to long `1`).
   - `-1.0`: **Exit Event** (transition from long `1` to flat `0`).
   - `0.0`: **Continuation** (no position state transition).

*Initial State Policy*: For the first valid observation after warmup, if `position_state = 1`, it is flagged as an initial entry (`1.0`); if `0`, no event (`0.0`).

### Initial Strategy Implementations
1. **Moving-Average Crossover (`src/strategies/moving_average.py`)**:
   - **Entry Rule**: $\text{SMA}_{\text{fast}} > \text{SMA}_{\text{slow}} \implies \text{position\_state} = 1.0$
   - **Exit Rule**: $\text{SMA}_{\text{fast}} \le \text{SMA}_{\text{slow}} \implies \text{position\_state} = 0.0$
   - Default parameters: `fast_window=20`, `slow_window=50`.
2. **Momentum Strategy (`src/strategies/momentum.py`)**:
   - **Entry Rule**: $\text{momentum}_N(t) = \frac{\text{Close}_t}{\text{Close}_{t-N}} - 1 > \text{threshold} \implies \text{position\_state} = 1.0$
   - **Exit Rule**: $\text{momentum}_N(t) \le \text{threshold} \implies \text{position\_state} = 0.0$
   - Default parameters: `lookback=20`, `threshold=0.0`.
3. **N-Day Breakout Strategy (`src/strategies/breakout.py`)**:
   - **Entry Rule**: $\text{Close}_t > \text{previous\_high}_N(t) \implies \text{position\_state} = 1.0$
   - **Exit Rule**: $\text{Close}_t < \text{previous\_low}_N(t) \implies \text{position\_state} = 0.0$
   - Default parameters: `lookback=20`.

### Look-Ahead Bias Prevention in Breakout Thresholds
For the N-day breakout strategy, today's price level ($\text{Close}_t$) is **NEVER** used to define today's breakout threshold. Today's breakout threshold is strictly `previous_high_N(t)`, which is defined as $\max(\text{Close}_{t-N} \dots \text{Close}_{t-1})$ via explicit 1-period shifting (`.shift(1)`).

> [!CAUTION]
> **Research Disclaimer**: The signal engine evaluates condition satisfaction only and does **NOT** establish whether any strategy is profitable. Strategy profitability, returns, drawdowns, and risk-adjusted metrics are evaluated separately by the backtesting engine.

---

## 8. Backtesting Engine

The backtesting engine (`src/backtest.py`) provides chronological portfolio simulation, trade execution modeling, and friction accounting.

### Execution Timing Conventions
- **Strategy Signal Execution**:
  $$\text{Signal generated at Close}(t) \implies \text{Trade executed at Open}(t+1)$$
  Signals generated on day $t$ cannot execute on day $t$. If a signal occurs on the final day $N-1$ Close, it is not executed due to the absence of a subsequent trading day.
- **Buy-and-Hold Benchmark Execution**:
  $$\text{Entered at first available Open}(t=0) \implies \text{Held through end of test period}$$

### Execution Frictions & Pricing Models
- **BUY Execution Price**: $\text{Open}_{t+1} \times (1 + \text{slippage\_rate})$
- **SELL Execution Price**: $\text{Open}_{t+1} \times (1 - \text{slippage\_rate})$
- **Transaction Costs**: Proportional cost rate ($\text{gross\_value} \times \text{transaction\_cost\_rate}$) deducted from cash on both entries and exits.
- Default friction assumptions: $\text{transaction\_cost\_rate} = 5\text{ bps}$ ($0.0005$), $\text{slippage\_rate} = 5\text{ bps}$ ($0.0005$).

### Position Sizing & Portfolio Accounting
- **Position Sizing**: Capital allocated $= \text{Total Equity}_t \times \text{position\_size}$ ($1.0 = 100\%$). Quantity $= \lfloor \text{Capital Allocated} / \text{Effective Price} \rfloor$.
- **Daily Portfolio Valuation**:
  $$\text{Position Value}_t = \text{Quantity}_t \times \text{Close}_t$$
  $$\text{Total Equity}_t = \text{Cash}_t + \text{Position Value}_t$$
- **End-of-Test Handling**: Unclosed positions at the end of the test period are valued mark-to-market at the final available Close price for reporting purposes, without fabricating unexecuted trade logs.

### Theoretical Index Unit Assumption
For the NIFTY 50 research dataset (`^NSEI`), backtest simulations model theoretical index units. This provides empirical strategy benchmarking and does not model futures contract lot sizes or margin accounting.

> [!WARNING]
> **Performance Simulation Disclaimer**: Backtest results are historical simulations and are **not** evidence of future profitability or live-market execution accuracy.

---

## 9. Performance & Risk Analytics

The performance analytics module (`src/metrics.py`) evaluates `BacktestResult` objects to quantify risk-adjusted returns, drawdown recovery, trade statistics, and transaction cost impact.

### Implemented Quantitative Metrics
1. **Total Return**:
   $$\text{Total Return} = \frac{\text{Final Equity}}{\text{Initial Capital}} - 1$$
2. **Compound Annual Growth Rate (CAGR)**:
   $$\text{CAGR} = \left(\frac{\text{Final Equity}}{\text{Initial Capital}}\right)^{\frac{1}{\text{Years}}} - 1, \quad \text{Years} = \frac{(T_{\text{end}} - T_{\text{start}})_{\text{days}}}{365.25}$$
3. **Annualized Volatility**:
   $$\text{Annualized Volatility} = \text{std}(\text{daily\_returns}) \times \sqrt{252}$$
4. **Maximum Drawdown & Recovery Analysis**:
   $$\text{Drawdown}_t = \frac{\text{Equity}_t}{\text{Running Peak}_t} - 1, \quad \text{Max Drawdown} = \min(\text{Drawdown}_t)$$
   Tracks drawdown peak date, trough date, and recovery date (when equity recovers to peak level).
5. **Sharpe Ratio (Annualized)**:
   $$\text{Sharpe} = \frac{\text{mean}(\text{daily\_returns} - R_f)}{\text{std}(\text{daily\_returns})} \times \sqrt{252}, \quad R_f = 0.0$$
6. **Sortino Ratio (Annualized)**:
   $$\text{Sortino} = \frac{\text{mean}(\text{daily\_returns} - \text{MAR})}{\text{Downside Deviation}} \times \sqrt{252}, \quad \text{MAR} = 0.0$$
7. **Calmar Ratio**:
   $$\text{Calmar} = \frac{\text{CAGR}}{|\text{Max Drawdown}|}$$
8. **Completed Round-Trip Trade Matching**:
   Trades are matched by pairing each BUY entry with its subsequent SELL exit (`match_round_trip_trades`).
   $$\text{Net Round-Trip PnL} = \text{Net Cash Flow}_{\text{SELL}} + \text{Net Cash Flow}_{\text{BUY}}$$
   - **Win Rate**: $\frac{\text{Winning Round-Trips}}{\text{Completed Round-Trips}}$
   - **Profit Factor**: $\frac{\sum \text{Winning PnL}}{|\sum \text{Losing PnL}|}$

### Key Assumptions & Methodological Rigor
- **Trading Days Annualization**: Standard 252 trading days per calendar year.
- **Risk-Free Rate & MAR**: Defaulted to $0.0$ baseline without fabricating artificial yield rates.
- **Why Total Return Alone is Insufficient**: High total return can hide severe drawdowns or excessive turnover costs. Risk-adjusted metrics (Sharpe, Sortino, Calmar, Max Drawdown) provide a balanced evaluation.

> [!IMPORTANT]
> **Research Disclaimer**: All metrics represent descriptive historical evaluation on past market data. Historical backtest performance metrics are **not** evidence or guarantees of future live-market profitability.

---

## 10. Out-of-Sample Testing & Robustness Analysis

The out-of-sample testing module (`src/oos.py`) evaluates strategy performance persistence and degradation across non-overlapping chronological partitions without parameter optimization or curve fitting.

### Chronological Period Partitioning
The dataset is partitioned into three distinct time-series periods:
1. **Development Period**: `2015-01-01` through `2021-12-31`
2. **Validation Period**: `2022-01-01` through `2023-12-31`
3. **Out-of-Sample (OOS) Test Period**: `2024-01-01` through latest available date

### Feature Warm-Up & Portfolio Boundary Architecture
- **Feature & Signal Calculation**: Point-in-time features and strategy signals are computed on the **full** chronological dataset prior to slicing. This preserves indicator warm-up history (e.g., SMA-200, 20-day previous highs) across period boundaries.
- **Portfolio Boundary Isolation**: Each partition (Dev, Val, OOS) is evaluated as an **independent** backtest simulation starting with a fresh initial capital balance ($100,000). Positions, cash, or accumulated equity are **NOT** carried over period boundaries, allowing unbiased historical evaluation.

### Fixed Strategy Specifications (No Curve Fitting)
All strategies are evaluated using pre-specified, fixed parameters across all three periods:
- **MA Crossover**: `fast_window = 20`, `slow_window = 50`
- **Momentum**: `lookback = 20`, `threshold = 0.0`
- **Breakout**: `lookback = 20`

### Performance Degradation & Signal Stability Metrics
- **Degradation Deltas**: Calculates $\Delta \text{CAGR} = \text{CAGR}_{\text{OOS}} - \text{CAGR}_{\text{Dev}}$, $\Delta \text{Sharpe} = \text{Sharpe}_{\text{OOS}} - \text{Sharpe}_{\text{Dev}}$, and $\Delta \text{MaxDD} = \text{MaxDD}_{\text{OOS}} - \text{MaxDD}_{\text{Dev}}$.
- **Signal Stability & Exposure**: Tracks entry/exit event counts and market exposure percentage per period to identify regime shifts or structural changes.

> [!CAUTION]
> **Out-of-Sample Evaluation Disclaimer**: The out-of-sample period is treated as chronologically held-out evaluation data under fixed, pre-specified strategy parameters. Strategy parameters were **not** optimized using observations from the out-of-sample period. Out-of-sample performance remains descriptive empirical evidence and is not a guarantee of future live-market trading performance.

---

## 11. Monte Carlo Risk Analysis

The Monte Carlo risk analysis module (`src/monte_carlo.py`) evaluates backtest outcomes via empirical bootstrap resampling to estimate the distribution of possible portfolio equity paths, terminal wealth outcomes, and maximum drawdown risk.

### Bootstrap Resampling Methodologies
1. **Trade-Level Bootstrap (Primary Analysis)**:
   - Resamples realized round-trip percentage returns ($r_1 \dots r_K$) with replacement (`bootstrap_trade_returns`).
   - Evaluates portfolio path distributions resulting from alternative ordering/sequences of realized trade outcomes.
2. **Daily-Return Bootstrap (Secondary Analysis)**:
   - Resamples daily portfolio return series ($R_1 \dots R_T$) with replacement (`bootstrap_daily_returns`), supporting optional block bootstrap (`block_size`).

### Simulation Specifications & Reproducibility
- **Simulations Count**: 5,000 iterations per strategy and period scope.
- **Random Seed**: Fixed random seed (`random_seed = 42`) for 100% deterministic reproducibility.
- **Initial Capital**: Fresh $100,000.00$ base per simulation.

### Empirical Sample Size Integrity & Percentile Interpretation
- **Empirical Sample Size Reporting**: Every Monte Carlo result explicitly reports the underlying number of completed trade observations ($K$). Bootstrapping 5,000 simulations resamples existing observations and does **NOT** increase the underlying empirical sample size or historical information.
- **Simulated Outcome Percentiles**: Percentile values (5th, 25th, 50th, 75th, 95th) are **simulated outcome percentiles** under the bootstrap model, **NOT** statistical confidence intervals.
- **5th Percentile Downside Scenarios**:
  - `5th Percentile Terminal Equity`: Represents an adverse downside outcome (95% of paths achieved higher final wealth).
  - `5th Percentile Max Drawdown`: Represents a severe downside drawdown (only 5% of simulated paths suffered a drawdown equal to or worse than this negative level).

### Serial Dependence & Methodological Limitations
- Simple empirical bootstrap resampling destroys historical temporal ordering, thereby underestimating risks associated with volatility clustering, regime shifts, or autocorrelation.

> [!WARNING]
> **Risk Analysis Disclaimer**: Monte Carlo simulation does **NOT** predict future market prices. It estimates outcome distributions under empirical resampling assumptions. A favorable Monte Carlo outcome does not prove future profitability, and an unfavorable outcome does not prove future failure. Monte Carlo results are **never** used to select strategies or tune strategy parameters.

---

## 12. AI Finance Controller — Multi-Source Reconciliation Engine

QuantMarket incorporates an explainable, multi-source financial reconciliation prototype located in [`src/reconciliation/`](file:///c:/Users/abdhi/OneDrive/Desktop/projects/quantmarket/src/reconciliation). The system ingests and reconciles high-volume transactions across four financial sources:
1. **Customer Invoices** (`invoice`)
2. **Internal Accounting General Ledger Payables** (`ledger`)
3. **Payment Gateway Settlements** (`gateway`)
4. **Bank Statement Feeds** (`bank`)

### Core Pipeline Architecture

```text
DATA INGESTION
→ NORMALIZATION
→ EXACT MATCHING & CANDIDATE GENERATION
→ RULE/FUZZY SCORING & POLICY ROUTING
→ AI REVIEW FOR AMBIGUOUS CASES (Agent Tools + Zero-Hallucination Policy)
→ STRICT VALIDATION (Pydantic Schema + Candidate Containment + Evidence Grounding)
→ MATCH / EXCEPTION ROUTING
→ COMPLETE AUDIT TRAIL
→ INDEPENDENT GROUND-TRUTH EVALUATION
→ OPERATIONS DASHBOARD
```

### Policy Routing & Gating Rules
- **Score $\ge 0.95$ (Single Top Candidate)**: Automatic deterministic match (`method = "deterministic"`).
- **Score $\ge 0.95$ (Multiple Plausible Candidates)**: Multiple candidate / duplicate ambiguity cannot be auto-matched; routed through `AIReviewer` or held as `MULTIPLE_CANDIDATES` exception.
- **$0.75 \le \text{Score} < 0.95$**: Ambiguous cases (e.g. gateway fee deductions, small settlement date shifts) are routed strictly to the `AIReviewer`.
- **Score $< 0.75$ or Missing**: Low confidence or missing transactions are routed directly to human-review `EXCEPTION`.

### AI Agent Tools & Guardrails
The `AIReviewer` utilizes an encapsulated tool suite (`ReconciliationAgentTools`) exposing:
- `get_transaction(record_id)`
- `search_transactions(query)`
- `get_candidates(record_id)`
- `compare_records(record_a_id, record_b_id)`
- `calculate_difference(amount_a, amount_b)`
- `mark_match(source_id, target_id, confidence, evidence, reason_codes)`
- `create_exception(source_id, reason_codes, confidence, evidence, recommended_action)`
- `get_reconciliation_status()`
- `generate_report()`

**Zero-Hallucination Enforcement**:
1. Decisions must strictly validate against the Pydantic `AIReviewDecision` schema.
2. `matched_record_id` must strictly belong to the supplied candidate set (unsupplied candidate IDs are rejected).
3. Evidence strings must ground factual statements in supplied records (citing unsupplied IDs triggers validation failure).
4. Any validation failure routes the item into an operational `EXCEPTION` rather than forcing a match.

### Independent Ground-Truth Evaluation & Metrics
A separate `data/reconciliation/ground_truth.json` is generated with hidden relationships and defect truth. The reconciliation engine never sees this truth during matching. The evaluation engine (`evaluation.py`) measures:
- **Precision**: $\frac{\text{Correct Matches}}{\text{Total System Matches}}$
- **Recall (Reconcilable Universe)**: $\frac{\text{Correct Matches}}{\text{Total Ground-Truth Reconcilable Invoices}}$ (explicitly documenting numerator and denominator).
- **False-Match Rate**: $\frac{\text{Incorrect Matches}}{\text{Total Invoices}}$.
- **Exception Rate**: $\frac{\text{Exceptions}}{\text{Total Invoices}}$.
- **Unresolved Financial Exposure**: Exact INR sum of pending exception items held for controller review.
- **Throughput & Speed**: Records per second and total execution time.

### Four Distinct System Outcomes
1. `deterministic`: Automatic high-confidence deterministic match.
2. `live_llm`: Live OpenAI review of bounded ambiguous candidate sets. A missing or failed API configuration creates an explicit human-review exception; it never simulates an AI decision.
3. `exception`: Human-review exception case with explicit reason codes and recommended operational action.

### Launching the Interactive Console

```powershell
uvicorn src.reconciliation.api:app --reload --port 8000
```
Open `http://127.0.0.1:8000` to access the interactive operations console. Click **Run Seeded Demo** to process 120 invoices across 228 records, inspect top KPI cards, view the live pipeline gate flow, filter exceptions by reason code, and use the **"Why did you match this?"** drawer to audit deterministic factor decomposition and AI tool traces.

> [!WARNING]
> **Prototype Disclaimer**: The AI Finance Controller is an assisted reconciliation proof-of-concept for educational and research evaluation. It does not claim autonomous or production-ready financial operations.
