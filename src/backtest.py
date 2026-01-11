"""
VectorBT回测流程模块 - 兼容层

该模块已重构拆分到 src/backtest/ 子模块中。
为保持向后兼容性，本文件重新导出所有公共接口。

新代码请直接从子模块导入：
    from src.backtest.engine import BacktestEngine
    from src.backtest.analyzer import PerformanceAnalyzer
    from src.backtest.result import BacktestResult

或从包导入：
    from src.backtest import BacktestEngine, BacktestResult
"""

# 重新导出所有公共接口
from .backtest import (
    BacktestResult,
    BacktestEngine,
    PerformanceAnalyzer,
    VBTProBacktester,
    VBT_AVAILABLE,
)

__all__ = [
    "BacktestResult",
    "BacktestEngine",
    "PerformanceAnalyzer",
    "VBTProBacktester",
    "VBT_AVAILABLE",
]
