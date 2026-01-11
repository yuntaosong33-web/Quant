"""
简单策略实现模块

该模块包含基础的交易策略实现：
- MACrossStrategy: 双均线交叉策略
- RSIStrategy: RSI超买超卖策略
- CompositeStrategy: 组合策略
"""

from typing import Optional, List, Dict, Any, Tuple
import logging

import pandas as pd
import numpy as np

from .base import BaseStrategy, TradeSignal, SignalType

logger = logging.getLogger(__name__)


class MACrossStrategy(BaseStrategy):
    """
    双均线交叉策略
    
    当短期均线上穿长期均线时买入，下穿时卖出。
    
    Parameters
    ----------
    short_window : int
        短期均线周期
    long_window : int
        长期均线周期
    
    Examples
    --------
    >>> config = {"short_window": 5, "long_window": 20}
    >>> strategy = MACrossStrategy("MA Cross", config)
    >>> signals = strategy.generate_signals(price_data)
    """
    
    def __init__(
        self,
        name: str = "MA Cross Strategy",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化均线交叉策略
        
        Parameters
        ----------
        name : str
            策略名称
        config : Optional[Dict[str, Any]]
            配置参数，包含 short_window 和 long_window
        """
        super().__init__(name, config)
        self.short_window = self.config.get("short_window", 5)
        self.long_window = self.config.get("long_window", 20)
        
        logger.info(
            f"均线交叉策略参数: 短期={self.short_window}, 长期={self.long_window}"
        )
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成均线交叉信号
        
        Parameters
        ----------
        data : pd.DataFrame
            价格数据，必须包含 'close' 列
        
        Returns
        -------
        pd.Series
            交易信号序列
        """
        close = data["close"]
        
        # 计算均线
        short_ma = close.rolling(window=self.short_window, min_periods=1).mean()
        long_ma = close.rolling(window=self.long_window, min_periods=1).mean()
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # 金叉买入信号
        golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        signals[golden_cross] = 1
        
        # 死叉卖出信号
        death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        signals[death_cross] = -1
        
        return signals
    
    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float
    ) -> float:
        """
        计算仓位大小
        
        使用固定比例仓位管理。
        
        Parameters
        ----------
        signal : TradeSignal
            交易信号
        portfolio_value : float
            组合价值
        
        Returns
        -------
        float
            建议仓位金额
        """
        base_size = portfolio_value * self._position_size
        adjusted_size = base_size * signal.strength
        
        return adjusted_size


class RSIStrategy(BaseStrategy):
    """
    RSI超买超卖策略
    
    当RSI低于超卖线时买入，高于超买线时卖出。
    
    Parameters
    ----------
    rsi_period : int
        RSI计算周期
    oversold : float
        超卖阈值
    overbought : float
        超买阈值
    """
    
    def __init__(
        self,
        name: str = "RSI Strategy",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化RSI策略
        
        Parameters
        ----------
        name : str
            策略名称
        config : Optional[Dict[str, Any]]
            配置参数
        """
        super().__init__(name, config)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.oversold = self.config.get("oversold", 30)
        self.overbought = self.config.get("overbought", 70)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成RSI信号
        
        Parameters
        ----------
        data : pd.DataFrame
            价格数据
        
        Returns
        -------
        pd.Series
            交易信号序列
        """
        close = data["close"]
        
        # 计算RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # 生成信号
        signals = pd.Series(0, index=data.index)
        
        # 超卖买入
        oversold_signal = (rsi < self.oversold) & (rsi.shift(1) >= self.oversold)
        signals[oversold_signal] = 1
        
        # 超买卖出
        overbought_signal = (rsi > self.overbought) & (rsi.shift(1) <= self.overbought)
        signals[overbought_signal] = -1
        
        return signals
    
    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float
    ) -> float:
        """计算仓位大小"""
        base_size = portfolio_value * self._position_size
        return base_size * signal.strength


class CompositeStrategy(BaseStrategy):
    """
    组合策略
    
    将多个策略组合在一起，通过加权投票生成综合信号。
    
    Parameters
    ----------
    strategies : List[Tuple[BaseStrategy, float]]
        策略和权重的列表
    """
    
    def __init__(
        self,
        name: str = "Composite Strategy",
        strategies: Optional[List[Tuple[BaseStrategy, float]]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化组合策略
        
        Parameters
        ----------
        name : str
            策略名称
        strategies : Optional[List[Tuple[BaseStrategy, float]]]
            子策略和权重列表
        config : Optional[Dict[str, Any]]
            配置参数
        """
        super().__init__(name, config)
        self.strategies = strategies or []
        self.threshold = self.config.get("threshold", 0.5)
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """
        添加子策略
        
        Parameters
        ----------
        strategy : BaseStrategy
            子策略实例
        weight : float
            策略权重
        """
        self.strategies.append((strategy, weight))
        logger.info(f"添加子策略: {strategy.name}, 权重: {weight}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成组合信号
        
        Parameters
        ----------
        data : pd.DataFrame
            价格数据
        
        Returns
        -------
        pd.Series
            加权组合信号
        """
        if not self.strategies:
            return pd.Series(0, index=data.index)
        
        total_weight = sum(w for _, w in self.strategies)
        weighted_signals = pd.Series(0.0, index=data.index)
        
        for strategy, weight in self.strategies:
            signals = strategy.generate_signals(data)
            weighted_signals += signals * (weight / total_weight)
        
        # 根据阈值转换为离散信号
        final_signals = pd.Series(0, index=data.index)
        final_signals[weighted_signals > self.threshold] = 1
        final_signals[weighted_signals < -self.threshold] = -1
        
        return final_signals
    
    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float
    ) -> float:
        """计算仓位大小"""
        base_size = portfolio_value * self._position_size
        return base_size * signal.strength


# 导出符号
__all__ = [
    "MACrossStrategy",
    "RSIStrategy",
    "CompositeStrategy",
]

