"""
VectorBT回测流程模块

该模块提供基于VectorBT的回测功能，支持策略绩效评估和分析。
采用向量化回测引擎确保高性能。

本模块已拆分为以下子模块：
- result: 回测结果数据类
- engine: 回测引擎（权重驱动多资产组合版）
- analyzer: 绩效分析器
- vbt_backtester: VectorBT Pro 全市场回测器

为保持向后兼容性，所有类仍可从本模块直接导入。
"""

import logging

# VectorBT 可用性检查
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logging.warning("vectorbt未安装，部分回测功能不可用")

# 从子模块导入所有公共接口
from .result import BacktestResult
from .engine import BacktestEngine
from .analyzer import PerformanceAnalyzer
from .vbt_backtester import VBTProBacktester


# 导出所有公共接口
__all__ = [
    "BacktestResult",
    "BacktestEngine",
    "PerformanceAnalyzer",
    "VBTProBacktester",
    "VBT_AVAILABLE",
]
