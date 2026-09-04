"""
QuantMarket Source Package.
"""

from src.backtest import (
    Backtester,
    BacktestResult,
    TradeRecord,
    run_buy_and_hold_benchmark,
)
from src.data_cleaner import DataCleaner, ValidationResult
from src.data_loader import DataLoader, DataLoaderError
from src.features import (
    add_features,
    calculate_atr,
    calculate_ema,
    calculate_returns,
    calculate_rolling_high_low,
    calculate_rolling_volatility,
    calculate_rsi,
    calculate_sma,
)
from src.metrics import (
    PerformanceMetrics,
    RoundTripTrade,
    compare_performance,
    evaluate_backtest,
    match_round_trip_trades,
)
from src.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    bootstrap_daily_returns,
    bootstrap_trade_returns,
    calculate_path_max_drawdown,
    compare_monte_carlo_results,
    run_monte_carlo_simulation,
)
from src.oos import (
    MultiPeriodBacktestResult,
    PeriodPartition,
    calculate_degradation,
    calculate_signal_stability,
    compare_robustness,
    run_multi_period_backtest,
    split_chronologically,
)
from src.signal_engine import detect_signal_events, validate_signal_values
from src.storage import StorageManager
from src.strategies import (
    generate_breakout_signals,
    generate_ma_crossover_signals,
    generate_momentum_signals,
)

__all__ = [
    "DataLoader",
    "DataLoaderError",
    "DataCleaner",
    "ValidationResult",
    "StorageManager",
    "add_features",
    "calculate_returns",
    "calculate_sma",
    "calculate_ema",
    "calculate_rolling_volatility",
    "calculate_rsi",
    "calculate_atr",
    "calculate_rolling_high_low",
    "validate_signal_values",
    "detect_signal_events",
    "generate_ma_crossover_signals",
    "generate_momentum_signals",
    "generate_breakout_signals",
    "Backtester",
    "BacktestResult",
    "TradeRecord",
    "run_buy_and_hold_benchmark",
    "PerformanceMetrics",
    "RoundTripTrade",
    "evaluate_backtest",
    "compare_performance",
    "match_round_trip_trades",
    "PeriodPartition",
    "MultiPeriodBacktestResult",
    "split_chronologically",
    "run_multi_period_backtest",
    "calculate_degradation",
    "calculate_signal_stability",
    "compare_robustness",
    "MonteCarloConfig",
    "MonteCarloResult",
    "bootstrap_trade_returns",
    "bootstrap_daily_returns",
    "calculate_path_max_drawdown",
    "run_monte_carlo_simulation",
    "compare_monte_carlo_results",
]
