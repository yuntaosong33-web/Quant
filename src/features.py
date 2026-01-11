"""
因子计算模块 - 兼容层

该模块已重构拆分到 src/features/ 子模块中。
为保持向后兼容性，本文件重新导出所有公共接口。

新代码请直接从子模块导入：
    from src.features.technical import TechnicalFeatures
    from src.features.alpha import AlphaFeatures
    from src.features.normalize import z_score_normalize

或从包导入：
    from src.features import TechnicalFeatures, AlphaFeatures
"""

# 重新导出所有公共接口
from .features import (
    # base
    FeatureEngine,
    # technical
    TechnicalFeatures,
    # alpha
    AlphaFeatures,
    # normalize
    z_score_normalize,
    # ic
    calculate_factor_ic,
    calculate_forward_returns,
    # numba_utils (常用的)
    _rolling_std_1d,
    _rolling_mean_1d,
    _calculate_rsi_numba,
    _calculate_returns_1d,
)

__all__ = [
    "FeatureEngine",
    "TechnicalFeatures",
    "AlphaFeatures",
    "z_score_normalize",
    "calculate_factor_ic",
    "calculate_forward_returns",
    "_rolling_std_1d",
    "_rolling_mean_1d",
    "_calculate_rsi_numba",
    "_calculate_returns_1d",
]
