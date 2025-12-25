"""
A股量化交易系统 (A-Share Quantitative Trading System)

一个基于Python的A股量化交易框架，支持数据获取、因子计算、
策略回测和绑定分析等功能。

主要模块:
    - data_loader: 数据获取与ETL处理
    - features: 因子计算引擎
    - strategy: 策略逻辑实现
    - backtest: VectorBT回测流程
    - optimizer: 投资组合优化
    - utils: 通用工具函数
"""

__version__ = "0.1.0"
__author__ = "Quant Developer"

from .data_loader import (
    DataHandler,
    AkshareDataLoader,
    AShareDataCleaner,
    DataLoader,
    DataCleaner,
    DownloadResult,
)
from .features import (
    FeatureEngine,
    TechnicalFeatures,
    AlphaFeatures,
    FactorCalculator,
    z_score_normalize,
)
from .strategy import (
    BaseStrategy,
    MACrossStrategy,
    RSIStrategy,
    CompositeStrategy,
    MultiFactorStrategy,
    SignalType,
    TradeSignal,
)
from .backtest import BacktestEngine, VBTProBacktester
from .optimizer import (
    optimize_max_sharpe,
    PortfolioOptimizer,
    OptimizationResult,
    calculate_expected_returns,
)
from .utils import (
    setup_logging,
    load_config,
    send_pushplus_msg,
    # GroupBy 向量化工具
    groupby_rolling,
    groupby_shift,
    groupby_pct_change,
    groupby_rank,
    groupby_zscore,
    groupby_winsorize,
    groupby_neutralize,
    groupby_cumsum,
    groupby_cumprod,
    groupby_ewm,
    groupby_diff,
    # PyPortfolioOpt 权重优化
    optimize_weights,
    optimize_portfolio_from_prices,
    calculate_shrinkage_covariance,
    calculate_expected_returns_mean,
    PortfolioWeightOptimizer,
)

__all__ = [
    # 数据处理
    "DataHandler",
    "AkshareDataLoader",
    "AShareDataCleaner",
    "DataLoader",
    "DataCleaner",
    "DownloadResult",
    # 特征工程
    "FeatureEngine",
    "TechnicalFeatures",
    "AlphaFeatures",
    "FactorCalculator",
    "z_score_normalize",
    # 策略
    "BaseStrategy",
    "MACrossStrategy",
    "RSIStrategy",
    "CompositeStrategy",
    "MultiFactorStrategy",
    "SignalType",
    "TradeSignal",
    # 回测
    "BacktestEngine",
    "VBTProBacktester",
    # 组合优化
    "optimize_max_sharpe",
    "PortfolioOptimizer",
    "OptimizationResult",
    "calculate_expected_returns",
    # 工具
    "setup_logging",
    "load_config",
    "send_pushplus_msg",
    # GroupBy 向量化
    "groupby_rolling",
    "groupby_shift",
    "groupby_pct_change",
    "groupby_rank",
    "groupby_zscore",
    "groupby_winsorize",
    "groupby_neutralize",
    "groupby_cumsum",
    "groupby_cumprod",
    "groupby_ewm",
    "groupby_diff",
    # PyPortfolioOpt 权重优化
    "optimize_weights",
    "optimize_portfolio_from_prices",
    "calculate_shrinkage_covariance",
    "calculate_expected_returns_mean",
    "PortfolioWeightOptimizer",
]

