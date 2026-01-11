"""
计时器工具模块

提供上下文管理器形式的计时器。
"""

import logging
import time


class Timer:
    """
    计时器上下文管理器
    
    Examples
    --------
    >>> with Timer("数据加载"):
    ...     data = load_data()
    数据加载 耗时: 1.23秒
    """
    
    def __init__(self, name: str = "操作") -> None:
        """
        初始化计时器
        
        Parameters
        ----------
        name : str
            操作名称
        """
        self.name = name
        self._start_time = None
    
    def __enter__(self) -> "Timer":
        """进入上下文"""
        self._start_time = time.time()
        return self
    
    def __exit__(self, *args) -> None:
        """退出上下文"""
        elapsed = time.time() - self._start_time
        logging.info(f"{self.name} 耗时: {elapsed:.2f}秒")

