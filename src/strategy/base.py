"""
策略基类与类型定义模块

该模块定义交易策略的抽象接口和通用数据类型。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

import pandas as pd
import numpy as np

# 导入 LLM 熔断器异常（用于风控）
try:
    from src.llm_client import LLMCircuitBreakerError
except ImportError:
    try:
        from llm_client import LLMCircuitBreakerError
    except ImportError:
        # 定义回退类以避免导入错误
        class LLMCircuitBreakerError(RuntimeError):
            """LLM 熔断器触发异常（回退定义）"""
            pass

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """交易信号类型枚举"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    """
    交易信号数据类
    
    Attributes
    ----------
    timestamp : pd.Timestamp
        信号时间
    symbol : str
        股票代码
    signal_type : SignalType
        信号类型
    price : float
        信号价格
    strength : float
        信号强度 (0-1)
    reason : str
        信号原因说明
    """
    timestamp: pd.Timestamp
    symbol: str
    signal_type: SignalType
    price: float
    strength: float = 1.0
    reason: str = ""


class BaseStrategy(ABC):
    """
    策略抽象基类
    
    所有交易策略必须继承此类并实现抽象方法。
    定义策略的基本接口和通用功能。
    
    Attributes
    ----------
    name : str
        策略名称
    config : Dict[str, Any]
        策略配置参数
    
    Methods
    -------
    generate_signals(data)
        生成交易信号
    calculate_position_size(signal, portfolio_value)
        计算仓位大小
    on_data(data)
        数据更新时的回调
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化策略
        
        Parameters
        ----------
        name : str
            策略名称
        config : Optional[Dict[str, Any]]
            策略配置参数
        """
        self.name = name
        self.config = config or {}
        self._signals: List[TradeSignal] = []
        self._positions: Dict[str, float] = {}
        
        # 从配置加载参数
        self._max_positions = self.config.get("max_positions", 10)
        self._position_size = self.config.get("position_size", 0.1)
        self._stop_loss = self.config.get("stop_loss", 0.08)
        self._take_profit = self.config.get("take_profit", 0.20)
        
        # ATR 动态止损参数
        self._use_atr_stop_loss = self.config.get("use_atr_stop_loss", True)
        self._atr_period = self.config.get("atr_period", 14)
        self._atr_multiplier = self.config.get("atr_multiplier", 2.5)
        
        logger.info(f"策略 '{name}' 初始化完成")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Parameters
        ----------
        data : pd.DataFrame
            包含OHLCV和因子的数据框
        
        Returns
        -------
        pd.Series
            信号序列，1表示买入，-1表示卖出，0表示持有
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float
    ) -> float:
        """
        计算仓位大小
        
        Parameters
        ----------
        signal : TradeSignal
            交易信号
        portfolio_value : float
            当前组合价值
        
        Returns
        -------
        float
            建议仓位金额
        """
        pass
    
    def on_data(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        """
        数据更新时的回调方法
        
        Parameters
        ----------
        data : pd.DataFrame
            最新数据
        
        Returns
        -------
        Optional[TradeSignal]
            交易信号，如果没有信号则返回None
        """
        signals = self.generate_signals(data)
        
        if signals.iloc[-1] != 0:
            signal = TradeSignal(
                timestamp=data.index[-1],
                symbol=data.get("symbol", "UNKNOWN"),
                signal_type=SignalType.BUY if signals.iloc[-1] > 0 else SignalType.SELL,
                price=data["close"].iloc[-1],
                strength=abs(signals.iloc[-1]),
            )
            self._signals.append(signal)
            return signal
        
        return None
    
    def get_signals_df(self) -> pd.DataFrame:
        """
        获取信号历史DataFrame
        
        Returns
        -------
        pd.DataFrame
            信号历史记录
        """
        if not self._signals:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "symbol": s.symbol,
                "signal": s.signal_type.value,
                "price": s.price,
                "strength": s.strength,
                "reason": s.reason,
            }
            for s in self._signals
        ])
    
    def reset(self) -> None:
        """重置策略状态"""
        self._signals.clear()
        self._positions.clear()
        logger.info(f"策略 '{self.name}' 状态已重置")
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        计算 ATR（平均真实波幅）
        
        Parameters
        ----------
        high : pd.Series
            最高价序列
        low : pd.Series
            最低价序列
        close : pd.Series
            收盘价序列
        period : int
            计算周期，默认 14
        
        Returns
        -------
        pd.Series
            ATR 值序列
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_dynamic_stop_loss(
        self,
        data: pd.DataFrame,
        entry_prices: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        计算基于 ATR 的动态止损价格
        
        Parameters
        ----------
        data : pd.DataFrame
            价格数据
        entry_prices : Optional[pd.Series]
            入场价格序列
        
        Returns
        -------
        pd.Series
            动态止损价格序列
        """
        if not self._use_atr_stop_loss:
            entry = entry_prices if entry_prices is not None else data['close']
            return entry * (1 - self._stop_loss)
        
        atr = self.calculate_atr(
            data['high'],
            data['low'],
            data['close'],
            period=self._atr_period
        )
        
        stop_distance = self._atr_multiplier * atr
        
        entry = entry_prices if entry_prices is not None else data['close']
        stop_loss_price = entry - stop_distance
        stop_loss_price = stop_loss_price.clip(lower=0)
        
        return stop_loss_price
    
    def check_stop_loss_triggered(
        self,
        data: pd.DataFrame,
        entry_prices: pd.Series,
        current_prices: pd.Series
    ) -> pd.Series:
        """
        检查是否触发动态止损
        
        Parameters
        ----------
        data : pd.DataFrame
            价格数据
        entry_prices : pd.Series
            入场价格
        current_prices : pd.Series
            当前价格
        
        Returns
        -------
        pd.Series
            布尔序列，True 表示触发止损
        """
        stop_prices = self.calculate_dynamic_stop_loss(data, entry_prices)
        return current_prices < stop_prices


# 导出符号
__all__ = [
    "SignalType",
    "TradeSignal",
    "BaseStrategy",
    "LLMCircuitBreakerError",
]

