"""
因子计算模块

该模块提供技术指标和因子计算功能，支持自定义因子扩展。
所有计算采用向量化操作以保证性能。

主要组件:
    - FeatureEngine: 因子计算引擎抽象基类
    - TechnicalFeatures: 技术指标计算器
    - AlphaFeatures: Alpha因子计算器
    - FactorCalculator: 多因子计算器 (从原始模块导入)
    - SentimentEngine: 情绪分析引擎 (从原始模块导入)
    - numba_utils: Numba优化的底层计算函数
    - normalize: Z-Score标准化函数
    - ic: 因子IC计算函数

Performance Notes
-----------------
- 使用 numba JIT 编译加速滚动窗口计算
- 避免使用 groupby().apply()，优先使用 transform() 或向量化操作
- 对于大规模数据（全市场A股），使用 numba 优化的底层函数
"""

# 核心类（新模块化版本）
from .base import FeatureEngine
from .technical import TechnicalFeatures
from .alpha import AlphaFeatures

# 标准化和IC计算函数（新拆分版本）
from .normalize import z_score_normalize, lag_fundamental_data, safe_zscore
from .ic import calculate_factor_ic, calculate_forward_returns

# Numba 工具函数
from .numba_utils import (
    rolling_std_1d,
    rolling_mean_1d,
    calculate_rsi_numba,
    calculate_returns_1d,
    grouped_rolling_std,
    grouped_rsi,
    calculate_atr_numba,
    calculate_volatility_numba,
    calculate_kdj_numba,
)

# 从原始模块导入大型类（向后兼容）
try:
    from ..features import FactorCalculator
except ImportError:
    try:
        from features import FactorCalculator
    except ImportError:
        FactorCalculator = None

# 导入情绪分析引擎
try:
    from ..sentiment_analyzer import SentimentEngine
except ImportError:
    try:
        from sentiment_analyzer import SentimentEngine
    except ImportError:
        SentimentEngine = None

__all__ = [
    # 核心类
    'FeatureEngine',
    'TechnicalFeatures',
    'AlphaFeatures',
    'FactorCalculator',
    'SentimentEngine',
    # 标准化函数
    'z_score_normalize',
    'lag_fundamental_data',
    'safe_zscore',
    # IC 计算函数
    'calculate_factor_ic',
    'calculate_forward_returns',
    # Numba 工具函数
    'rolling_std_1d',
    'rolling_mean_1d',
    'calculate_rsi_numba',
    'calculate_returns_1d',
    'grouped_rolling_std',
    'grouped_rsi',
    'calculate_atr_numba',
    'calculate_volatility_numba',
    'calculate_kdj_numba',
]

