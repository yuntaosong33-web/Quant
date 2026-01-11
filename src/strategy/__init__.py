"""
策略模块

本模块提供各种交易策略的实现。

策略类层次：
- BaseStrategy: 抽象基类，定义策略接口
  - MACrossStrategy: 双均线交叉策略
  - RSIStrategy: RSI超买超卖策略
  - MultiFactorStrategy: 多因子选股策略

使用示例：
    >>> from src.strategy import MultiFactorStrategy, MACrossStrategy
    >>> 
    >>> # 多因子策略
    >>> strategy = MultiFactorStrategy("MF", {"top_n": 5})
    >>> positions = strategy.generate_target_positions(factor_data)
    >>> 
    >>> # 均线策略
    >>> ma_strategy = MACrossStrategy("MA", {"fast_window": 10, "slow_window": 30})
    >>> signals = ma_strategy.generate_signals(price_data)
"""
from .base import BaseStrategy, SignalType, TradeSignal
from .simple import MACrossStrategy, RSIStrategy
from .multi_factor import MultiFactorStrategy

__all__ = [
    # 基类与类型
    "BaseStrategy",
    "SignalType",
    "TradeSignal",
    # 简单策略
    "MACrossStrategy",
    "RSIStrategy",
    # 多因子策略
    "MultiFactorStrategy",
]

