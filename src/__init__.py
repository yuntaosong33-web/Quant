"""
A股量化交易系统 (A-Share Quantitative Trading System)

一个基于Python的A股量化交易框架，支持数据获取、因子计算、
策略回测和绑定分析等功能。

主要模块:
    - data_loader: 数据获取与ETL处理
    - tushare/: Tushare 数据加载器（重构后的模块化版本）
    - strategy/: 策略逻辑实现（重构后的模块化版本）
    - features: 因子计算引擎
    - backtest: VectorBT回测流程
    - optimizer: 投资组合优化
    - utils: 通用工具函数
    - daily_runner: 每日更新运行器
    - backtest_runner: 回测执行逻辑
    - report_generator: 报告生成器
"""

__version__ = "0.2.0"
__author__ = "Quant Developer"

# ========================================
# 数据加载模块
# ========================================
from .data_loader import (
    DataHandler,
    AShareDataCleaner,
    DataLoader,
    DataCleaner,
    DownloadResult,
)

# Tushare 数据加载器（优先使用新模块，回退到旧模块）
try:
    from .tushare import TushareClient as TushareDataLoader
    from .tushare import create_tushare_loader
except ImportError:
    from .tushare_loader import TushareDataLoader, create_tushare_loader

# ========================================
# 特征工程模块
# ========================================
from .features import (
    FeatureEngine,
    TechnicalFeatures,
    AlphaFeatures,
    FactorCalculator,
    z_score_normalize,
)

# ========================================
# 策略模块（优先使用新模块，回退到旧模块）
# ========================================
try:
    from .strategy import (
        BaseStrategy,
        MACrossStrategy,
        RSIStrategy,
        MultiFactorStrategy,
        SignalType,
        TradeSignal,
    )
    # CompositeStrategy 可能不在新模块中
    try:
        from .strategy import CompositeStrategy
    except ImportError:
        CompositeStrategy = None
except ImportError:
    # 回退到旧的 strategy.py
    from .strategy import (
        BaseStrategy,
        MACrossStrategy,
        RSIStrategy,
        CompositeStrategy,
        MultiFactorStrategy,
        SignalType,
        TradeSignal,
    )

# ========================================
# 回测模块
# ========================================
from .backtest import BacktestEngine, VBTProBacktester

# ========================================
# 组合优化模块
# ========================================
from .optimizer import (
    optimize_max_sharpe,
    PortfolioOptimizer,
    OptimizationResult,
    calculate_expected_returns,
)

# ========================================
# 工具模块
# ========================================
from .utils import (
    setup_logging,
    load_config,
    send_pushplus_msg,
    # 数据标准化
    DataStandardizer,
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

# ========================================
# 新增模块（重构版）
# ========================================
try:
    from .report_generator import ReportGenerator
except ImportError:
    ReportGenerator = None

try:
    from .daily_runner import DailyUpdateRunner, run_daily_update
except ImportError:
    DailyUpdateRunner = None
    run_daily_update = None

try:
    from .backtest_runner import run_backtest as run_backtest_new
except ImportError:
    run_backtest_new = None

__all__ = [
    # 数据处理
    "DataHandler",
    "AShareDataCleaner",
    "DataLoader",
    "DataCleaner",
    "DownloadResult",
    "TushareDataLoader",
    "create_tushare_loader",
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
    # 数据标准化
    "DataStandardizer",
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
    # 新增模块（重构版）
    "ReportGenerator",
    "DailyUpdateRunner",
    "run_daily_update",
    "run_backtest_new",
]

