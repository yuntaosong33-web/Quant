"""
通用工具模块

提供日志配置、配置文件加载、数据验证、GroupBy 向量化计算、投资组合优化等通用功能。

本模块已拆分为以下子模块：
- config: 日志和配置文件管理
- dataframe: DataFrame 数据操作工具
- timer: 计时器工具
- messaging: 消息推送工具
- groupby: Pandas GroupBy 向量化计算
- portfolio: PyPortfolioOpt 投资组合优化
- standardizer: 数据源标准化工具

为保持向后兼容性，所有函数和类仍可从本模块直接导入。
"""

# 从子模块导入所有公共接口
from .config import (
    setup_logging,
    load_config,
    save_config,
)

from .dataframe import (
    validate_dataframe,
    resample_ohlcv,
    calculate_returns,
    rolling_zscore,
    winsorize,
    neutralize,
    create_dir_structure,
)

from .timer import Timer

from .messaging import (
    send_pushplus_msg,
    format_number,
)

from .groupby import (
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
    groupby_apply_multiple,
    cross_sectional_regression,
)

from .portfolio import (
    calculate_shrinkage_covariance,
    calculate_expected_returns_mean,
    optimize_weights,
    optimize_portfolio_from_prices,
    PortfolioWeightOptimizer,
)

from .standardizer import DataStandardizer


# 导出所有公共接口
__all__ = [
    # config
    "setup_logging",
    "load_config",
    "save_config",
    # dataframe
    "validate_dataframe",
    "resample_ohlcv",
    "calculate_returns",
    "rolling_zscore",
    "winsorize",
    "neutralize",
    "create_dir_structure",
    # timer
    "Timer",
    # messaging
    "send_pushplus_msg",
    "format_number",
    # groupby
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
    "groupby_apply_multiple",
    "cross_sectional_regression",
    # portfolio
    "calculate_shrinkage_covariance",
    "calculate_expected_returns_mean",
    "optimize_weights",
    "optimize_portfolio_from_prices",
    "PortfolioWeightOptimizer",
    # standardizer
    "DataStandardizer",
]
