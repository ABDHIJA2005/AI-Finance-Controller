# AGENTS.md — QuantMarket Development Guidelines

This document outlines the strict engineering, mathematical, and research standards governing development within the **QuantMarket** codebase.

---

## 1. Core Engineering Standards

- **Python Version**: Target Python 3.10+ clean, modern syntax.
- **Code Architecture**: Keep modules focused, decoupled, and single-purpose (`data`, `signals`, `backtest`, `risk`).
- **Type Annotations**: Enforce explicit type hints (`typing` / built-in generics) across all function signatures and class definitions.
- **Docstrings**: Provide detailed docstrings (NumPy format) for all public functions, classes, and methods, documenting parameters, return types, mathematical definitions, and assumptions.
- **Testing**: Maintain comprehensive unit tests (`pytest`) covering matrix operations, signal logic, backtest accounting, and risk metrics. Run all tests following any functional change.

---

## 2. Quantitative & Research Rigor

- **No Fabricated Financial Results**: Never generate, fabricate, or hardcode fake market data or backtest performance numbers. All signals and strategy metrics must derive from reproducible processing of verified empirical market data.
- **Zero Look-Ahead Bias**:
  - Information available at time $t$ MUST strictly consist of data known on or before time $t$.
  - Feature normalization, rolling statistics, signal scaling, and indicators must use point-in-time calculation (e.g. expanding or rolling windows without future lookahead).
- **Chronological Time-Series Validation**:
  - Never shuffle time-series data randomly across train/validation/test splits.
  - Maintain strict chronological ordering throughout data ingestion, feature generation, model validation, and strategy backtesting.
- **Period Separation**:
  - Explicitly partition historical datasets into distinct **Development (In-Sample)**, **Validation**, and **Out-of-Sample Test** timeframes to prevent overfitting and data leakage.
- **Market Frictions & Realistic Execution**:
  - All strategy backtests must explicitly incorporate realistic transaction costs (brokerage/exchange fees) and slippage models.
  - Account for execution delays and order execution mechanics.
- **Reproducibility & Assumptions**:
  - Set explicit random seeds for any stochastic simulations (e.g., Monte Carlo risk modeling).
  - Explicitly document all financial, statistical, and operational assumptions made in strategy and risk formulas.
- **Data Provenance**:
  - Clearly document all external data sources, dataset identifiers, timestamps, time zones, and preprocessing transformations.

---

## 3. Workflow & Verification Rules

1. **Inspect before changing**: Inspect existing schemas, functions, and tests before making modifications.
2. **Modular Implementation**: Implement core analytical components in `src/` prior to wrapping them in notebook visualizations or dashboard UI elements.
3. **Continuous Testing**: Execute `pytest` after introducing new calculations or modifying analytical modules.
