"""
数据加载模块 - 兼容层

该模块已重构拆分到 src/data_loader/ 子模块中。
为保持向后兼容性，本文件重新导出所有公共接口。

新代码请直接从子模块导入：
    from src.data_loader.base import DataHandler
    from src.data_loader.cleaner import AShareDataCleaner

或从包导入：
    from src.data_loader import DataHandler, AShareDataCleaner
"""

# 重新导出所有公共接口
from .data_loader import (
    DataHandler,
    AShareDataCleaner,
)

__all__ = [
    "DataHandler",
    "AShareDataCleaner",
]
