"""
策略逻辑实现模块

该模块定义交易策略的抽象接口和具体实现，支持多种策略类型。
使用抽象基类确保策略的一致性和可扩展性。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
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
        
        ATR = 过去 N 日 True Range 的移动平均
        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        
        Parameters
        ----------
        high : pd.Series
            最高价序列
        low : pd.Series
            最低价序列
        close : pd.Series
            收盘价序列
        period : int, optional
            计算周期，默认 14
        
        Returns
        -------
        pd.Series
            ATR 值序列
        """
        prev_close = close.shift(1)
        
        # 计算 True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算 ATR（使用 EWM 或 SMA）
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_dynamic_stop_loss(
        self,
        data: pd.DataFrame,
        entry_prices: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        计算基于 ATR 的动态止损价格
        
        止损价 = 入场价 - ATR_Multiplier * ATR(14)
        
        对于30万小资金账户，使用 ATR 止损比固定百分比更适合：
        - 波动大的股票止损更宽，避免被洗出
        - 波动小的股票止损更紧，保护利润
        
        Parameters
        ----------
        data : pd.DataFrame
            价格数据，必须包含 high, low, close 列
        entry_prices : Optional[pd.Series]
            入场价格序列。如果为 None，使用收盘价作为参考
        
        Returns
        -------
        pd.Series
            动态止损价格序列
        
        Examples
        --------
        >>> stop_prices = strategy.calculate_dynamic_stop_loss(ohlcv_data)
        >>> # 检查是否触发止损
        >>> triggered = data['close'] < stop_prices
        
        Notes
        -----
        默认配置：
        - ATR 周期: 14 日
        - ATR 乘数: 2.5
        - 止损价 = 入场价 - 2.5 * ATR(14)
        """
        if not self._use_atr_stop_loss:
            # 回退到固定百分比止损
            entry = entry_prices if entry_prices is not None else data['close']
            return entry * (1 - self._stop_loss)
        
        # 计算 ATR
        atr = self.calculate_atr(
            data['high'],
            data['low'],
            data['close'],
            period=self._atr_period
        )
        
        # 计算止损距离
        stop_distance = self._atr_multiplier * atr
        
        # 计算止损价格
        entry = entry_prices if entry_prices is not None else data['close']
        stop_loss_price = entry - stop_distance
        
        # 确保止损价格不为负
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
            价格数据（用于计算 ATR）
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
        """
        计算仓位大小（波动率倒数加权）
        
        Weight_i = (1 / Vol_i) / Sum(1 / Vol_j)
        
        如果无法获取波动率，则回退到等权重。
        """
        # 简单回退：直接返回等权金额（实际权重分配在 generate_target_weights 中处理）
        return portfolio_value / max(1, self.top_n)

    def generate_target_weights(
        self,
        factor_data: pd.DataFrame,
        current_weights: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        生成目标权重矩阵（核心逻辑优化版）
        
        优化点：
        1. 引入大盘择时：大盘跌破20日线时减半仓位
        2. 波动率倒数加权：降低妖股仓位
        3. 缓冲区逻辑：减少无效交易
        """
        logger.info(f"生成目标权重矩阵 (Buffer: {self.rebalance_buffer:.1%})...")
        
        # 获取所有交易日期
        dates = sorted(factor_data[self.date_col].unique())
        if not dates:
            return pd.DataFrame()
        
        # 确定调仓日期
        rebalance_dates = self.get_rebalance_dates(
            pd.DatetimeIndex(dates), 
            self.rebalance_frequency
        )
        
        # 初始化权重矩阵
        all_stocks = sorted(factor_data[self.stock_col].unique())
        target_weights = pd.DataFrame(
            0.0, 
            index=dates, 
            columns=all_stocks
        )
        
        # 逐个调仓日处理
        for date in rebalance_dates:
            # 1. 筛选当日候选股
            day_data = self.filter_stocks(factor_data, date)
            if day_data.empty:
                continue
            
            # 2. 计算综合得分
            scores = self.calculate_total_score(day_data)
            day_data['total_score'] = scores
            
            # 3. 选出 Top N
            # 增加一步：RSI 过滤 (RSI < 85)
            if 'rsi_20' in day_data.columns:
                day_data = day_data[day_data['rsi_20'] < 85]
            
            top_stocks_df = day_data.nlargest(self.top_n, 'total_score')
            selected_stocks = top_stocks_df[self.stock_col].tolist()
            
            if not selected_stocks:
                continue
                
            # 4. 计算权重 (波动率倒数加权)
            # 假设 vol_20 存在，否则用等权
            weights = {}
            if 'vol_20' in top_stocks_df.columns:
                inv_vol = 1.0 / top_stocks_df['vol_20'].replace(0, 0.01)
                vol_sum = inv_vol.sum()
                if vol_sum > 0:
                    for stock, iv in zip(selected_stocks, inv_vol):
                        weights[stock] = iv / vol_sum
                else:
                    # 回退等权
                    weight_each = 1.0 / len(selected_stocks)
                    weights = {s: weight_each for s in selected_stocks}
            else:
                 weight_each = 1.0 / len(selected_stocks)
                 weights = {s: weight_each for s in selected_stocks}
            
            # 5. 填充到权重矩阵
            for stock, w in weights.items():
                if stock in target_weights.columns:
                    target_weights.loc[date, stock] = w
                    
        # 填充非调仓日的权重（向前填充，模拟持有）
        target_weights = target_weights.ffill().fillna(0.0)
        
        return target_weights


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
        """
        计算仓位大小（波动率倒数加权）
        
        Weight_i = (1 / Vol_i) / Sum(1 / Vol_j)
        
        如果无法获取波动率，则回退到等权重。
        """
        # 简单回退：直接返回等权金额（实际权重分配在 generate_target_weights 中处理）
        return portfolio_value / max(1, self.top_n)

    def generate_target_weights(
        self,
        factor_data: pd.DataFrame,
        current_weights: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        生成目标权重矩阵（核心逻辑优化版）
        
        优化点：
        1. 引入大盘择时：大盘跌破20日线时减半仓位
        2. 波动率倒数加权：降低妖股仓位
        3. 缓冲区逻辑：减少无效交易
        """
        logger.info(f"生成目标权重矩阵 (Buffer: {self.rebalance_buffer:.1%})...")
        
        # 获取所有交易日期
        dates = sorted(factor_data[self.date_col].unique())
        if not dates:
            return pd.DataFrame()
        
        # 确定调仓日期
        rebalance_dates = self.get_rebalance_dates(
            pd.DatetimeIndex(dates), 
            self.rebalance_frequency
        )
        
        # 初始化权重矩阵
        all_stocks = sorted(factor_data[self.stock_col].unique())
        target_weights = pd.DataFrame(
            0.0, 
            index=dates, 
            columns=all_stocks
        )
        
        # 逐个调仓日处理
        for date in rebalance_dates:
            # 1. 筛选当日候选股
            day_data = self.filter_stocks(factor_data, date)
            if day_data.empty:
                continue
            
            # 2. 计算综合得分
            scores = self.calculate_total_score(day_data)
            day_data['total_score'] = scores
            
            # 3. 选出 Top N
            # 增加一步：RSI 过滤 (RSI < 85)
            if 'rsi_20' in day_data.columns:
                day_data = day_data[day_data['rsi_20'] < 85]
            
            top_stocks_df = day_data.nlargest(self.top_n, 'total_score')
            selected_stocks = top_stocks_df[self.stock_col].tolist()
            
            if not selected_stocks:
                continue
                
            # 4. 计算权重 (波动率倒数加权)
            # 假设 vol_20 存在，否则用等权
            weights = {}
            if 'vol_20' in top_stocks_df.columns:
                inv_vol = 1.0 / top_stocks_df['vol_20'].replace(0, 0.01)
                vol_sum = inv_vol.sum()
                if vol_sum > 0:
                    for stock, iv in zip(selected_stocks, inv_vol):
                        weights[stock] = iv / vol_sum
                else:
                    # 回退等权
                    weight_each = 1.0 / len(selected_stocks)
                    weights = {s: weight_each for s in selected_stocks}
            else:
                 weight_each = 1.0 / len(selected_stocks)
                 weights = {s: weight_each for s in selected_stocks}
            
            # 5. 填充到权重矩阵
            for stock, w in weights.items():
                if stock in target_weights.columns:
                    target_weights.loc[date, stock] = w
                    
        # 填充非调仓日的权重（向前填充，模拟持有）
        target_weights = target_weights.ffill().fillna(0.0)
        
        return target_weights


class MultiFactorStrategy(BaseStrategy):
    """
    多因子选股策略
    
    基于价值、质量和动量因子的综合打分进行选股。
    
    打分公式 (默认): 
    Final_Score = Quality_Weight * Quality_Z + Momentum_Weight * Momentum_Z + Size_Weight * Size_Z
    (+ Sentiment_Score * Sentiment_Weight if enabled)
    
    Parameters
    ----------
    name : str
        策略名称
    config : Optional[Dict[str, Any]]
        配置参数，包含：
        - value_weight: 价值因子权重 (默认 0.0)
        - quality_weight: 质量因子权重 (默认 0.3)
        - momentum_weight: 动量因子权重 (默认 0.7)
        - size_weight: 市值因子权重 (默认 0.0)
        - top_n: 选取股票数量
        - momentum_col: 动量因子列名 (默认 sharpe_20_zscore)
    
    Attributes
    ----------
    value_weight : float
        价值因子权重
    quality_weight : float
        质量因子权重
    momentum_weight : float
        动量因子权重
    top_n : int
        选取股票数量
    min_listing_days : int
        最小上市天数
    
    Examples
    --------
    >>> config = {
    ...     "value_weight": 0.4,
    ...     "quality_weight": 0.4,
    ...     "momentum_weight": 0.2,
    ...     "top_n": 30
    ... }
    >>> strategy = MultiFactorStrategy("Multi-Factor", config)
    >>> target_positions = strategy.generate_target_positions(factor_data)
    """
    
    def __init__(
        self,
        name: str = "Multi-Factor Strategy",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        初始化多因子策略
        
        Parameters
        ----------
        name : str
            策略名称
        config : Optional[Dict[str, Any]]
            策略配置参数
        """
        super().__init__(name, config)
        
        # 因子权重配置
        # 激进型小市值策略配置：
        # - value_weight: 借用位置放小市值因子
        # - quality_weight: 借用位置放换手率因子
        # - momentum_weight: RSI 动量因子
        # - size_weight: 独立的小市值因子权重
        self.value_weight: float = self.config.get("value_weight", 0.0)
        self.quality_weight: float = self.config.get("quality_weight", 0.3)
        self.momentum_weight: float = self.config.get("momentum_weight", 0.7)
        self.size_weight: float = self.config.get("size_weight", 0.0)  # 默认移除市值因子
        self.sentiment_weight: float = self.config.get("sentiment_weight", 0.2)  # 情绪进攻型权重
        
        # 选股参数配置
        self.top_n: int = self.config.get("top_n", 5)  # 默认激进持仓 5 只
        
        # 30万小资金账户适配：最大持仓数量限制为 8
        MAX_POSITIONS_LIMIT = 8
        if self.top_n > MAX_POSITIONS_LIMIT:
            logger.warning(
                f"配置的 top_n ({self.top_n}) 超过了小资金账户限制 ({MAX_POSITIONS_LIMIT})，"
                f"强制调整为 {MAX_POSITIONS_LIMIT}"
            )
            self.top_n = MAX_POSITIONS_LIMIT

        self.min_listing_days: int = self.config.get("min_listing_days", 126)  # 约6个月
        
        # 板块过滤配置
        self._exclude_chinext: bool = self.config.get("exclude_chinext", False)  # 排除创业板
        self._exclude_star: bool = self.config.get("exclude_star", False)  # 排除科创板
        
        # 因子列名配置（支持自定义列名）
        self.value_col: str = self.config.get("value_col", "value_zscore")
        self.quality_col: str = self.config.get("quality_col", "turnover_5d_zscore")
        self.momentum_col: str = self.config.get("momentum_col", "sharpe_20_zscore")  # 默认夏普动量
        self.size_col: str = self.config.get("size_col", "small_cap_zscore")
        
        # 日期和股票列名配置
        self.date_col: str = self.config.get("date_col", "date")
        self.stock_col: str = self.config.get("stock_col", "stock_code")
        
        # 调仓频率配置: 'monthly' 或 'weekly'
        self.rebalance_frequency: str = self.config.get("rebalance_frequency", "monthly")
        
        # ===== 再平衡缓冲区（Rebalance Buffer）=====
        # 用于避免小资金账户因微小调整产生的最低5元佣金磨损
        # 仅当 |w_new - w_old| > buffer_threshold 时才调整仓位
        # 例：5%缓冲区 = 30万 * 5% = 1.5万，按万三计算佣金4.5元，不足最低5元
        self.rebalance_buffer: float = self.config.get("rebalance_buffer", 0.05)
        
        # ===== [NEW] 持股惯性加分（Holding Bonus）=====
        # 当前持仓股票在选股时获得的得分加成（单位：标准差）
        # 作用：减少不必要的换手，让优质股票有更长的持有周期
        # 值越大，越倾向于持有当前股票
        self.holding_bonus: float = self.config.get("holding_bonus", 0.0)
        
        # ===== [NEW] 大盘风控参数（Market Risk Control）=====
        # 从配置读取，支持动态调整
        market_risk_config = self.config.get("market_risk", {})
        self._market_risk_enabled: bool = market_risk_config.get("enabled", True)
        self._market_risk_ma_period: int = market_risk_config.get("ma_period", 60)
        self._market_risk_drop_threshold: float = market_risk_config.get("drop_threshold", 0.05)
        self._market_risk_drop_lookback: int = market_risk_config.get("drop_lookback", 20)
        
        if self.rebalance_frequency not in ("monthly", "weekly"):
            logger.warning(
                f"不支持的调仓频率 '{self.rebalance_frequency}'，使用默认 'monthly'"
            )
            self.rebalance_frequency = "monthly"
        
        # 验证权重之和（包含新增的 size_weight）
        # 注意：sentiment_weight 是额外的 Alpha 因子权重，不计入基础权重归一化
        # 由用户配置保证合理性
        weight_sum = self.value_weight + self.quality_weight + self.momentum_weight + self.size_weight
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"基础因子权重之和为 {weight_sum}，建议权重之和为 1.0")
        
        # ===== LLM 情绪分析配置 =====
        # 从配置中获取 LLM 设置
        self._llm_config = self.config.get("llm", {})
        self._enable_sentiment_filter: bool = self._llm_config.get("enable_sentiment_filter", False)
        self._sentiment_threshold: float = self._llm_config.get("sentiment_threshold", -0.5)
        self._min_confidence: float = self._llm_config.get("min_confidence", 0.7)
        self._sentiment_buffer_multiplier: int = self._llm_config.get("sentiment_buffer_multiplier", 3)
        self._sentiment_engine = None
        
        # 过热熔断阈值（换手率 Z-Score）
        # [牛市进攻型] 大幅放宽：50.0 基本禁用熔断
        self.turnover_threshold: float = self.config.get("turnover_threshold", 50.0)
        
        # 波动率熔断阈值（年化波动率）
        # [牛市进攻型] 放宽波动率限制
        self.volatility_threshold: float = self.config.get("volatility_threshold", 5.0)  # 500% 年化波动率
        
        # ===== [NEW] 拥挤度板块轮动配置 =====
        # 利用 CrowdingFactorCalculator 计算行业拥挤度，实现板块轮动
        self._enable_crowding_rotation: bool = self.config.get("enable_crowding_rotation", False)
        self._crowding_exit_threshold: float = self.config.get("crowding_exit_threshold", 0.95)
        self._crowding_entry_threshold: float = self.config.get("crowding_entry_threshold", 0.50)
        self._crowding_calculator = None
        self._crowding_cache: Dict[str, pd.DataFrame] = {}
        
        # ===== [NEW] Alpha 因子开关 =====
        self._enable_alpha_factors: bool = self.config.get("enable_alpha_factors", True)
        
        # 初始化情绪分析引擎（如果启用）
        if self._enable_sentiment_filter:
            try:
                from src.features import SentimentEngine
                self._sentiment_engine = SentimentEngine(self._llm_config)
                logger.info(
                    f"情绪分析过滤已启用: threshold={self._sentiment_threshold}, "
                    f"min_confidence={self._min_confidence}, buffer_multiplier={self._sentiment_buffer_multiplier}"
                )
            except ImportError:
                logger.warning(
                    "无法导入 SentimentEngine，情绪分析过滤未启用。"
                    "请确保 src.features 模块可用。"
                )
                self._enable_sentiment_filter = False
            except Exception as e:
                logger.warning(f"初始化 SentimentEngine 失败: {e}")
                self._enable_sentiment_filter = False
        
        logger.info(
            f"多因子策略初始化: 价值权重={self.value_weight}, "
            f"质量权重={self.quality_weight}, 动量权重={self.momentum_weight}, "
            f"市值权重={self.size_weight}, 情绪权重={self.sentiment_weight}, "
            f"Top N={self.top_n}, 调仓频率={self.rebalance_frequency}, "
            f"再平衡缓冲区={self.rebalance_buffer:.1%}, "
            f"持股惯性加分={self.holding_bonus:.2f}, "
            f"情绪过滤={'启用' if self._enable_sentiment_filter else '禁用'}"
        )
    
    def calculate_total_score(
        self,
        data: pd.DataFrame,
        sentiment_scores: Optional[pd.Series] = None,
        return_components: bool = False
    ) -> pd.Series:
        """
        计算综合因子得分
        
        多因子策略公式：
        Total_Score = value_weight * Value_Z + quality_weight * Quality_Z 
                    + momentum_weight * Momentum_Z + size_weight * Size_Z
        
        情绪进攻型策略额外加分：
        Final_Score = Base_Score + sentiment_weight * Sentiment_Score * 3.0
        
        特殊处理：
        - 换手率因子引入"过热惩罚"：Z-Score > 2.0 时反向扣分（仅对 turnover 类因子）
        - 情绪因子量纲对齐：乘以 3.0 放大系数使其与技术因子匹配
        
        Parameters
        ----------
        data : pd.DataFrame
            包含因子 Z-Score 的数据框
        sentiment_scores : Optional[pd.Series]
            情绪分数序列（范围 -1 到 1），索引应为股票代码。
        return_components : bool
            是否返回各因子的得分分量（用于 Debug）
        
        Returns
        -------
        pd.Series
            综合得分序列
        
        Notes
        -----
        - 缺失的因子值会被视为 0
        - 使用向量化操作计算得分
        - 过热惩罚仅适用于包含 "turnover" 的质量因子列名
        """
        total_score = pd.Series(0.0, index=data.index)
        
        # 用于存储各因子分量（Debug 用）
        score_components = {}
        
        # ===== 价值因子 =====
        value_contribution = pd.Series(0.0, index=data.index)
        if self.value_col in data.columns and self.value_weight > 0:
            value_contribution = self.value_weight * data[self.value_col].fillna(0)
            total_score += value_contribution
            score_components['value'] = data[self.value_col].fillna(0)
        elif self.value_weight > 0:
            logger.warning(f"未找到价值因子列: {self.value_col}")
        
        # ===== 质量因子 =====
        # 判断是否为换手率类因子（需要过热惩罚）
        quality_contribution = pd.Series(0.0, index=data.index)
        is_turnover_factor = 'turnover' in self.quality_col.lower()
        
        if self.quality_col in data.columns and self.quality_weight > 0:
            raw_quality = data[self.quality_col].fillna(0)
            
            if is_turnover_factor:
                # 换手率因子：牛市进攻型配置
                # [关键修改] 将换手率视为正向因子（高人气）
                # Z-Score > 3.5 时才轻微惩罚，保留大部分高换手股
                # 牛市核心逻辑：高换手 = 高人气 = 趋势延续
                TURNOVER_PENALTY_THRESHOLD = 3.5  # 惩罚阈值放宽
                quality_score = np.where(
                    raw_quality > TURNOVER_PENALTY_THRESHOLD,
                    TURNOVER_PENALTY_THRESHOLD - (raw_quality - TURNOVER_PENALTY_THRESHOLD) * 0.5,  # 轻微惩罚
                    raw_quality  # 正常情况保持原值（正向加分）
                )
                overheat_count = (raw_quality > TURNOVER_PENALTY_THRESHOLD).sum()
                if overheat_count > 0:
                    logger.debug(f"换手率过热惩罚: {overheat_count} 只股票 Z-Score > {TURNOVER_PENALTY_THRESHOLD}")
            else:
                # 非换手率因子（如 ROE、IVOL）：直接使用原值
                quality_score = raw_quality
            
            quality_contribution = self.quality_weight * quality_score
            total_score += quality_contribution
            score_components['quality'] = pd.Series(quality_score, index=data.index)
        elif self.quality_weight > 0:
            logger.warning(f"未找到质量因子列: {self.quality_col}")
        
        # ===== 动量因子 =====
        momentum_contribution = pd.Series(0.0, index=data.index)
        if self.momentum_col in data.columns and self.momentum_weight > 0:
            momentum_contribution = self.momentum_weight * data[self.momentum_col].fillna(0)
            total_score += momentum_contribution
            score_components['momentum'] = data[self.momentum_col].fillna(0)
        elif self.momentum_weight > 0:
            logger.warning(f"未找到动量因子列: {self.momentum_col}")
        
        # ===== 市值因子 =====
        size_contribution = pd.Series(0.0, index=data.index)
        if self.size_col in data.columns and self.size_weight > 0:
            size_contribution = self.size_weight * data[self.size_col].fillna(0)
            total_score += size_contribution
            score_components['size'] = data[self.size_col].fillna(0)
        elif self.size_weight > 0:
            logger.warning(f"未找到市值因子列: {self.size_col}")
        
        # 存储分量供后续 Debug 使用
        if return_components:
            data['_value_score'] = value_contribution
            data['_quality_score'] = quality_contribution
            data['_momentum_score'] = momentum_contribution
            data['_size_score'] = size_contribution
        
        # ===== 情绪进攻型策略：加入情绪分数 =====
        # 情绪因子量纲对齐：情绪分数范围 [-1, 1]，Z-Score 通常在 [-3, 3]
        # 乘以放大系数 3.0 使其影响力与技术因子匹配
        SENTIMENT_SCALE_FACTOR = 3.0
        
        if sentiment_scores is not None and self.sentiment_weight > 0:
            # 确定股票代码列用于对齐
            stock_col = self.stock_col if self.stock_col in data.columns else 'symbol'
            
            if stock_col in data.columns:
                # 使用股票代码对齐情绪分数
                aligned_sentiment = data[stock_col].map(sentiment_scores).fillna(0)
                # 应用放大系数进行量纲对齐
                scaled_sentiment = aligned_sentiment * SENTIMENT_SCALE_FACTOR
                total_score += self.sentiment_weight * scaled_sentiment
                logger.debug(
                    f"情绪分数已加入综合得分: 权重={self.sentiment_weight}, "
                    f"放大系数={SENTIMENT_SCALE_FACTOR}, "
                    f"有效股票数={aligned_sentiment.notna().sum()}"
                )
            else:
                # 尝试使用索引对齐
                if isinstance(data.index, pd.MultiIndex):
                    stock_codes = data.index.get_level_values(-1)
                else:
                    stock_codes = data.index
                aligned_sentiment = stock_codes.to_series().map(sentiment_scores).fillna(0)
                aligned_sentiment.index = data.index
                # 应用放大系数进行量纲对齐
                scaled_sentiment = aligned_sentiment * SENTIMENT_SCALE_FACTOR
                total_score += self.sentiment_weight * scaled_sentiment
                logger.debug(
                    f"情绪分数已加入综合得分 (索引对齐): 权重={self.sentiment_weight}, "
                    f"放大系数={SENTIMENT_SCALE_FACTOR}"
                )
        
        return total_score
    
    def get_month_end_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        获取每月最后一个交易日
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            所有交易日期
        
        Returns
        -------
        pd.DatetimeIndex
            每月最后一个交易日
        """
        dates_series = pd.Series(dates, index=dates)
        # 按年月分组，取每组最后一个日期
        month_end_dates = dates_series.groupby(
            [dates_series.index.year, dates_series.index.month]
        ).last()
        
        return pd.DatetimeIndex(month_end_dates.values)
    
    def get_week_end_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        获取每周最后一个交易日
        
        用于周度调仓策略，更快捕捉动量变化。
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            所有交易日期
        
        Returns
        -------
        pd.DatetimeIndex
            每周最后一个交易日
        
        Notes
        -----
        使用 ISO 周数（周一开始，周日结束）进行分组。
        """
        dates_series = pd.Series(dates, index=dates)
        # 按年-周分组，取每组最后一个日期
        # isocalendar 返回 (year, week, weekday)
        week_end_dates = dates_series.groupby(
            [dates_series.index.isocalendar().year, 
             dates_series.index.isocalendar().week]
        ).last()
        
        return pd.DatetimeIndex(week_end_dates.values)
    
    def get_rebalance_dates(
        self, 
        dates: pd.DatetimeIndex, 
        frequency: str = "monthly"
    ) -> pd.DatetimeIndex:
        """
        根据频率获取调仓日期
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            所有交易日期
        frequency : str
            调仓频率，可选 'monthly' 或 'weekly'
        
        Returns
        -------
        pd.DatetimeIndex
            调仓日期
        
        Raises
        ------
        ValueError
            当 frequency 参数不合法时
        """
        if frequency == "monthly":
            return self.get_month_end_dates(dates)
        elif frequency == "weekly":
            return self.get_week_end_dates(dates)
        else:
            raise ValueError(
                f"不支持的调仓频率: {frequency}，可选 'monthly' 或 'weekly'"
            )
    
    # ===== 小资金实盘优化：流动性与可交易性过滤常量 =====
    # 最低日成交额（元），低于此值的股票流动性不足
    # 中证1000尾部股票流动性较差，提高到5000万以避免滑点陷阱
    MIN_DAILY_AMOUNT = 50_000_000  # 5000万（从2000万提高）
    # 涨停判断阈值（涨幅 >= 9.5% 视为涨停）
    LIMIT_UP_THRESHOLD = 0.095
    # 跌停判断阈值（跌幅 >= 9.5% 视为跌停）
    LIMIT_DOWN_THRESHOLD = -0.095
    # ST/退市股关键字
    ST_KEYWORDS = ('ST', '*ST', '退', 'S', 'PT')
    # 换手率过热熔断阈值
    # [牛市进攻型] 大幅放宽：Z-Score > 5.0 才视为极度过热
    # 牛市龙头股换手率通常很高，过于敏感会卖飞妖股
    TURNOVER_OVERHEAT_THRESHOLD = 5.0
    
    def filter_stocks(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        根据条件过滤股票（增强版：小资金实盘优化 + 风控熔断）
        
        针对激进型小市值策略的增强过滤，确保选出的股票：
        1. 可交易（非涨跌停、非ST）
        2. 有流动性（成交额足够）
        3. 价格适中（能买入足够手数）
        4. 上市足够久（非次新股）
        5. 不过热（换手率Z分数不超标）
        
        过滤条件（硬性风控，直接剔除）：
        1. 剔除涨跌停股票 (无法买入/卖出)
        2. 剔除一字涨停股票 (High == Low 且涨幅 >= 9.5%)
        3. 剔除流动性不足股票 (日成交额 < 5000万)
        4. 剔除 ST/*ST/退市股票 (高风险标的)
        5. 剔除高价股 (> 100元)
        6. 剔除上市不满 6 个月的股票
        7. 剔除创业板/科创板（可配置）
        8. **剔除过热股票 (turnover_5d_zscore > 2.5)**
        
        Parameters
        ----------
        data : pd.DataFrame
            因子数据，应包含以下列（部分可选）：
            - close, high, low: 价格数据
            - amount: 成交额
            - pct_change 或 pctChg: 涨跌幅
            - name 或 stock_name: 股票名称（用于ST过滤）
            - is_limit: 涨跌停标志
            - listing_days 或 list_date: 上市信息
            - turnover_5d_zscore: 5日换手率Z分数（用于过热熔断）
        date : pd.Timestamp
            当前日期
        
        Returns
        -------
        pd.DataFrame
            过滤后的数据
        
        Notes
        -----
        小资金实盘优化说明：
        - 30万资金持有5只股票，每只约6万
        - 日成交额 < 5000万的股票，6万资金可能产生滑点（中证1000尾部）
        - 一字涨停股票实盘无法买入，必须剔除
        - ST股票风险极高，不适合激进策略
        
        过热熔断说明：
        - turnover_5d_zscore > 2.5 表示换手率处于极端高位
        - 这类股票短期投机过热，容易在高位接盘
        - 直接剔除而非降低分数，属于硬性风控
        """
        # 获取当日数据
        if self.date_col in data.columns:
            day_data = data[data[self.date_col] == date].copy()
        elif isinstance(data.index, pd.DatetimeIndex):
            day_data = data.loc[data.index == date].copy()
        elif isinstance(data.index, pd.MultiIndex):
            # MultiIndex: (date, stock_code)
            if date in data.index.get_level_values(0):
                day_data = data.loc[date].copy()
            else:
                day_data = pd.DataFrame()
        else:
            logger.warning(f"无法获取日期 {date} 的数据")
            return pd.DataFrame()
        
        if day_data.empty:
            return day_data
        
        # [FIX] 定义股票代码列名
        stock_col = self.stock_col if self.stock_col in day_data.columns else 'symbol'
        
        initial_count = len(day_data)
        filter_stats = {}  # 记录各过滤条件剔除的数量
        
        # ==========================================
        # 过滤条件 1: 剔除涨跌停股票（通用 is_limit 标志）
        # ==========================================
        if 'is_limit' in day_data.columns:
            before = len(day_data)
            day_data = day_data[~day_data['is_limit'].fillna(False)]
            filter_stats['is_limit'] = before - len(day_data)
        
        # ==========================================
        # 过滤条件 2: 剔除一字涨停股票（买不进）
        # 判断条件：High == Low 且 涨幅 >= 9.5%
        # ==========================================
        if 'high' in day_data.columns and 'low' in day_data.columns:
            # 获取涨跌幅列（兼容多种列名）
            pct_col = None
            for col in ['pct_change', 'pctChg', 'change_pct', 'pct']:
                if col in day_data.columns:
                    pct_col = col
                    break
            
            if pct_col:
                # 一字涨停：最高价 == 最低价 且 涨幅 >= 9.5%
                is_one_word_limit_up = (
                    (day_data['high'] == day_data['low']) & 
                    (day_data[pct_col] >= self.LIMIT_UP_THRESHOLD * 100)  # 假设百分比格式
                )
                
                # 如果涨跌幅是小数格式（如 0.095）
                if day_data[pct_col].abs().max() < 1:
                    is_one_word_limit_up = (
                        (day_data['high'] == day_data['low']) & 
                        (day_data[pct_col] >= self.LIMIT_UP_THRESHOLD)
                    )
                
                before = len(day_data)
                day_data = day_data[~is_one_word_limit_up.fillna(False)]
                filter_stats['一字涨停'] = before - len(day_data)
        
        # ==========================================
        # 过滤条件 3: 剔除流动性黑洞（日成交额 < 5000万）
        # 中证1000尾部股票流动性较差，5000万阈值避免滑点陷阱
        # ==========================================
        if 'amount' in day_data.columns:
            before = len(day_data)
            # 成交额可能是万元单位，统一转换
            amount_values = day_data['amount'].copy()
            # 判断单位：如果最大值 < 100万，可能是万元单位
            if amount_values.max() < 1_000_000:
                # 万元单位，转换为元
                amount_in_yuan = amount_values * 10000
            else:
                # 元单位
                amount_in_yuan = amount_values
            
            low_liquidity_mask = amount_in_yuan < self.MIN_DAILY_AMOUNT
            low_liquidity_stocks = day_data[low_liquidity_mask]
            
            if len(low_liquidity_stocks) > 0:
                # 获取被剔除的股票代码和成交额
                if stock_col in low_liquidity_stocks.columns:
                    low_liq_codes = low_liquidity_stocks[stock_col].tolist()
                elif isinstance(low_liquidity_stocks.index, pd.Index):
                    low_liq_codes = low_liquidity_stocks.index.tolist()
                else:
                    low_liq_codes = []
                
                # 详细日志（显示成交额）
                if len(low_liq_codes) <= 5:
                    logger.debug(
                        f"💧 流动性不足 {date.strftime('%Y-%m-%d')}: "
                        f"剔除 {len(low_liq_codes)} 只 (成交额 < {self.MIN_DAILY_AMOUNT/1e8:.1f}亿): "
                        f"{low_liq_codes}"
                    )
            
            day_data = day_data[~low_liquidity_mask]
            filter_stats['流动性不足'] = before - len(day_data)
        else:
            logger.debug("数据中缺少 'amount' 列，跳过流动性过滤")
            
        # ==========================================
        # 过滤条件 4: 剔除 ST/*ST/退市股票
        # ==========================================
        name_col = None
        for col in ['name', 'stock_name', '股票名称', 'sec_name']:
            if col in day_data.columns:
                name_col = col
                break
        
        if name_col:
            before = len(day_data)
            # 构建 ST 过滤条件
            st_mask = day_data[name_col].astype(str).apply(
                lambda x: any(kw in x for kw in self.ST_KEYWORDS)
            )
            day_data = day_data[~st_mask]
            filter_stats['ST/退市'] = before - len(day_data)
        
        # ==========================================
        # 过滤条件 5: 剔除高价股 (> 100元)
        # 依据规则：确保每只股票能买入至少 2-3 手（200-300股）
        # 30万资金，3只股票，每只约10万，100元股票可买1000股
        # ==========================================
        MAX_PRICE_LIMIT = 100.0
        price_col = 'close'
        if price_col not in day_data.columns:
            price_col = next((col for col in ['price', 'close_price'] if col in day_data.columns), None)
        
        if price_col:
            before = len(day_data)
            high_price_mask = day_data[price_col] > MAX_PRICE_LIMIT
            day_data = day_data[~high_price_mask]
            filter_stats['高价股'] = before - len(day_data)
        else:
            logger.warning(f"数据中缺少价格列，无法执行高价股过滤")
        
        # ==========================================
        # 过滤条件 6: 剔除上市不满 6 个月的股票
        # ==========================================
        before = len(day_data)
        if 'days_listed' in day_data.columns:
            day_data = day_data[day_data['days_listed'] >= self.min_listing_days]
        elif 'listing_days' in day_data.columns:
            day_data = day_data[day_data['listing_days'] >= self.min_listing_days]
        elif 'list_date' in day_data.columns:
            list_dates = pd.to_datetime(day_data['list_date'])
            listing_days = (date - list_dates).dt.days
            day_data = day_data[listing_days >= self.min_listing_days]
        elif 'ipo_date' in day_data.columns:
            ipo_dates = pd.to_datetime(day_data['ipo_date'])
            listing_days = (date - ipo_dates).dt.days
            day_data = day_data[listing_days >= self.min_listing_days]
        filter_stats['次新股'] = before - len(day_data)
        
        # ==========================================
        # 过滤条件 7: 剔除创业板股票（可配置）
        # 创业板代码以 300xxx 或 301xxx 开头
        # ==========================================
        stock_col = self.stock_col if self.stock_col in day_data.columns else 'symbol'
        
        if self._exclude_chinext:
            if stock_col in day_data.columns:
                before = len(day_data)
                # 创业板股票代码以 300 或 301 开头
                chinext_mask = day_data[stock_col].astype(str).str[:3].isin(['300', '301'])
                day_data = day_data[~chinext_mask]
                filter_stats['创业板'] = before - len(day_data)
            elif isinstance(day_data.index, pd.Index):
                before = len(day_data)
                chinext_mask = day_data.index.astype(str).str[:3].isin(['300', '301'])
                day_data = day_data[~chinext_mask]
                filter_stats['创业板'] = before - len(day_data)
        
        # ==========================================
        # 过滤条件 8: 剔除科创板股票（可配置）
        # 科创板代码以 688xxx 开头
        # ==========================================
        if self._exclude_star:
            if stock_col in day_data.columns:
                before = len(day_data)
                star_mask = day_data[stock_col].astype(str).str[:3] == '688'
                day_data = day_data[~star_mask]
                filter_stats['科创板'] = before - len(day_data)
            elif isinstance(day_data.index, pd.Index):
                before = len(day_data)
                star_mask = day_data.index.astype(str).str[:3] == '688'
                day_data = day_data[~star_mask]
                filter_stats['科创板'] = before - len(day_data)
        
        # ==========================================
        # 过滤条件 10: RSI 过热保护 (RSI > 80)
        # 在震荡市中，纯动量策略容易死在最高点
        # ==========================================
        if 'rsi_20' in day_data.columns:
            before = len(day_data)
            rsi_mask = day_data['rsi_20'] > 80
            day_data = day_data[~rsi_mask]
            filter_stats['RSI过热'] = before - len(day_data)

        # ==========================================
        # 过滤条件 9: 过热熔断（Turnover Overheat Filter）
        # 换手率 Z-Score 超过阈值时剔除，不参与后续打分
        # 
        # 风控逻辑：
        # - 极高换手率往往意味着短期投机过热
        # - 这类股票波动剧烈，容易在高位接盘
        # - 直接剔除比降低分数更安全（硬性风控）
        # 
        # 情绪豁免逻辑（牛市进攻型）：
        # - 如果情绪分数极高（> 0.8），说明是市场合力
        # - 即使换手率超标也不熔断，保留热门股机会
        # ==========================================
        turnover_col = self.quality_col  # 默认 turnover_5d_zscore
        
        # 恢复过热熔断逻辑（尊重 Config 配置）
        check_col = turnover_col
        
        # 情绪豁免阈值：情绪分数高于此值时豁免熔断
        SENTIMENT_EXEMPT_THRESHOLD = 0.8
        
        if check_col in day_data.columns:
            before = len(day_data)
            
            # 检查是否有情绪分数列（支持 score 或 sentiment_score）
            sentiment_col = None
            if 'score' in day_data.columns:
                sentiment_col = 'score'
            elif 'sentiment_score' in day_data.columns:
                sentiment_col = 'sentiment_score'
            
            # 构建过热熔断 mask（使用配置中的阈值 turnover_threshold）
            if sentiment_col is not None:
                # 情绪豁免逻辑：换手率超标 且 情绪分不高时，才触发熔断
                overheat_mask = (
                    (day_data[check_col] > self.turnover_threshold) & 
                    (day_data[sentiment_col].fillna(0) < SENTIMENT_EXEMPT_THRESHOLD)
                )
                # 统计被情绪豁免的股票数量
                raw_overheat_count = (day_data[check_col] > self.turnover_threshold).sum()
                exempt_count = raw_overheat_count - overheat_mask.sum()
                if exempt_count > 0:
                    logger.info(
                        f"🛡️ 情绪豁免 {date.strftime('%Y-%m-%d')}: "
                        f"{exempt_count} 只股票换手率超标但情绪分 >= {SENTIMENT_EXEMPT_THRESHOLD}，保留"
                    )
            else:
                # 无情绪分数列，仅使用阈值判断
                overheat_mask = day_data[check_col] > self.turnover_threshold
            
                overheat_stocks = day_data[overheat_mask]
            
            if len(overheat_stocks) > 0:
                # 获取被剔除的股票代码列表
                if stock_col in overheat_stocks.columns:
                    overheat_codes = overheat_stocks[stock_col].tolist()
                elif isinstance(overheat_stocks.index, pd.Index):
                    overheat_codes = overheat_stocks.index.tolist()
                else:
                    overheat_codes = []
                
                # 获取具体的 Z-Score 值用于日志
                overheat_details = []
                for idx, row in overheat_stocks.iterrows():
                    code = row[stock_col] if stock_col in row.index else idx
                    zscore = row[check_col]
                    # 如果有情绪分数，也显示出来
                    if sentiment_col is not None and sentiment_col in row.index:
                        sent_score = row[sentiment_col]
                        overheat_details.append(f"{code}(turn={zscore:.2f},sent={sent_score:.2f})")
                    else:
                        overheat_details.append(f"{code}({zscore:.2f})")
                
                # 剔除过热股票
                day_data = day_data[~overheat_mask]
                filter_stats['过热熔断'] = before - len(day_data)
                
                # 输出详细日志
                logger.warning(
                    f"🔥 过热熔断 {date.strftime('%Y-%m-%d')}: "
                    f"剔除 {len(overheat_codes)} 只 (turnover_zscore > {self.turnover_threshold}): "
                    f"{overheat_details[:10]}"  # 最多显示10只
                    + (f"... 等共 {len(overheat_codes)} 只" if len(overheat_codes) > 10 else "")
                )
        else:
            logger.debug(f"数据中缺少 '{check_col}' 列，跳过过热熔断过滤")
        
        # ==========================================
        # 注意：情绪分析已移至 _apply_sentiment_filter 方法
        # 采用 "Filter-Then-Analyze" 模式：仅对 Top N * buffer 的候选股票进行情绪分析
        # 这显著减少了 LLM API 调用次数，降低成本并提高效率
        # ==========================================
        
        # 汇总日志
        total_filtered = initial_count - len(day_data)
        if total_filtered > 0:
            filter_detail = ", ".join(f"{k}:{v}" for k, v in filter_stats.items() if v > 0)
            logger.debug(
                f"日期 {date.strftime('%Y-%m-%d')}: "
                f"过滤 {total_filtered} 只 ({filter_detail}), "
                f"剩余 {len(day_data)}/{initial_count}"
            )
        else:
            logger.debug(f"日期 {date.strftime('%Y-%m-%d')}: 无股票被过滤, 剩余 {len(day_data)}")
        
        return day_data
    
    # ==================== 拥挤度板块轮动方法 ====================
    
    def calculate_sector_crowding(
        self,
        price_data: pd.DataFrame,
        stock_sector_map: Dict[str, str],
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算行业拥挤度因子（用于板块轮动决策）
        
        拥挤度 = 行业内股票收益率的平均相关系数
        高拥挤度表示行业抱团严重，有拥挤风险
        
        Parameters
        ----------
        price_data : pd.DataFrame
            价格数据，index=date, columns=stock_codes
        stock_sector_map : Dict[str, str]
            股票代码到行业的映射
        window : int
            滚动窗口大小，默认 20
        
        Returns
        -------
        pd.DataFrame
            行业拥挤度数据，index=date, columns=sector_names
        
        Notes
        -----
        - 拥挤度 > 95% 分位数时，建议分批止盈
        - 拥挤度 < 50% 分位数且动量起爆时，可切入
        """
        if not self._enable_crowding_rotation:
            logger.debug("拥挤度轮动未启用，跳过计算")
            return pd.DataFrame()
        
        try:
            from src.crowding_factor import CrowdingFactorCalculator
        except ImportError:
            try:
                from crowding_factor import CrowdingFactorCalculator
            except ImportError:
                logger.warning("无法导入 CrowdingFactorCalculator，拥挤度轮动不可用")
                return pd.DataFrame()
        
        # 初始化计算器（懒加载）
        if self._crowding_calculator is None:
            self._crowding_calculator = CrowdingFactorCalculator(
                window=window,
                min_periods=10,
                use_dask=False
            )
        
        try:
            crowding_df = self._crowding_calculator.calculate(
                price_data,
                stock_sector_map
            )
            logger.info(f"行业拥挤度计算完成: {crowding_df.shape}")
            return crowding_df
        except Exception as e:
            logger.warning(f"行业拥挤度计算失败: {e}")
            return pd.DataFrame()
    
    def apply_crowding_rotation(
        self,
        candidates: List[str],
        crowding_data: pd.DataFrame,
        stock_sector_map: Dict[str, str],
        date: pd.Timestamp,
        current_holdings: Optional[List[str]] = None
    ) -> List[str]:
        """
        应用拥挤度轮动策略调整候选股票
        
        策略逻辑：
        1. 当持仓股票所属行业拥挤度 > exit_threshold 时，从候选中移除
        2. 当行业拥挤度回落到 entry_threshold 以下且动量起爆时，优先选入
        
        Parameters
        ----------
        candidates : List[str]
            候选股票代码列表
        crowding_data : pd.DataFrame
            行业拥挤度数据
        stock_sector_map : Dict[str, str]
            股票代码到行业的映射
        date : pd.Timestamp
            当前日期
        current_holdings : Optional[List[str]]
            当前持仓股票列表
        
        Returns
        -------
        List[str]
            调整后的候选股票列表
        
        Notes
        -----
        - 拥挤度使用百分位排名，0-1 范围
        - exit_threshold: 0.95 表示 95% 分位以上触发止盈
        - entry_threshold: 0.50 表示 50% 分位以下可切入
        """
        if not self._enable_crowding_rotation:
            return candidates
        
        if crowding_data.empty or date not in crowding_data.index:
            logger.debug(f"日期 {date} 无拥挤度数据，跳过轮动")
            return candidates
        
        # 获取当日各行业拥挤度
        day_crowding = crowding_data.loc[date]
        
        # 计算拥挤度百分位（0-1）
        crowding_percentile = day_crowding.rank(pct=True)
        
        adjusted_candidates = []
        removed_by_crowding = []
        
        for stock in candidates:
            sector = stock_sector_map.get(stock, None)
            if sector is None:
                # 无行业信息，保留
                adjusted_candidates.append(stock)
                continue
            
            sector_percentile = crowding_percentile.get(sector, 0.5)
            
            # 拥挤度过高的行业：剔除（分批止盈逻辑）
            if sector_percentile > self._crowding_exit_threshold:
                removed_by_crowding.append(f"{stock}({sector}:{sector_percentile:.0%})")
            else:
                adjusted_candidates.append(stock)
        
        if removed_by_crowding:
            logger.info(
                f"📊 拥挤度轮动 {date.strftime('%Y-%m-%d')}: "
                f"因行业过热剔除 {len(removed_by_crowding)} 只: "
                f"{removed_by_crowding[:5]}"
                + (f"... 共{len(removed_by_crowding)}只" if len(removed_by_crowding) > 5 else "")
            )
        
        return adjusted_candidates
    
    def get_low_crowding_sectors(
        self,
        crowding_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> List[str]:
        """
        获取低拥挤度的行业列表（用于寻找切入机会）
        
        Parameters
        ----------
        crowding_data : pd.DataFrame
            行业拥挤度数据
        date : pd.Timestamp
            当前日期
        
        Returns
        -------
        List[str]
            低拥挤度行业名称列表
        """
        if crowding_data.empty or date not in crowding_data.index:
            return []
        
        day_crowding = crowding_data.loc[date]
        crowding_percentile = day_crowding.rank(pct=True)
        
        # 筛选低于入场阈值的行业
        low_crowding_sectors = crowding_percentile[
            crowding_percentile < self._crowding_entry_threshold
        ].index.tolist()
        
        if low_crowding_sectors:
            logger.debug(
                f"低拥挤度行业 ({date.strftime('%Y-%m-%d')}): "
                f"{low_crowding_sectors}"
            )
        
        return low_crowding_sectors
    
    def _apply_sentiment_filter(
        self,
        candidates: List[str],
        date: pd.Timestamp
    ) -> List[str]:
        """
        对预选候选股票应用 LLM 情绪分析过滤
        
        采用 "Filter-Then-Analyze" 模式：
        仅对 Top N * buffer 的候选股票进行情绪分析，而非全市场股票。
        这显著减少了 LLM API 调用次数，降低成本并提高效率。
        
        **安全策略 (Fail-Closed)**:
        当 LLM 服务不可用或发生错误时，系统采用安全优先策略：
        - 熔断器触发: 抛出 LLMCircuitBreakerError，完全停止交易
        - 其他 LLM 错误: 返回空列表，阻止生成买入信号
        这避免了在 LLM 风控不可用时仍产生买入信号的危险情况。
        
        Parameters
        ----------
        candidates : List[str]
            预选的股票代码列表（通常为 Top N * buffer）
        date : pd.Timestamp
            分析日期
        
        Returns
        -------
        List[str]
            通过情绪过滤的股票代码列表
        
        Raises
        ------
        LLMCircuitBreakerError
            当 LLM API 连续失败超过阈值时抛出，停止交易
        
        Notes
        -----
        过滤规则：
        1. score < sentiment_threshold: 剔除（负面情绪）
        2. confidence < min_confidence: 剔除并记录警告（不确定分析结果）
        
        异常处理（Fail-Closed 策略）：
        - LLMCircuitBreakerError: 记录 CRITICAL 日志并抛出异常，停止交易
        - 其他异常: 记录 CRITICAL 日志并返回空列表，阻止产生买入信号
        """
        if not candidates:
            return []
        
        if not self._enable_sentiment_filter or self._sentiment_engine is None:
            return candidates
        
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        try:
            # 调用情绪分析引擎（返回 DataFrame）
            sentiment_df = self._sentiment_engine.calculate_sentiment(candidates, date_str)
            
            if sentiment_df.empty:
                # 情绪分析返回空结果也视为异常情况
                # Fail-Closed: 返回空列表而非原候选列表
                logger.critical(
                    f"⛔ 情绪分析返回空结果 ({date_str}), "
                    f"Fail-Closed: 阻止所有 {len(candidates)} 只候选股票的买入信号"
                )
                return []
            
            # 过滤逻辑
            # 一票否决阈值：情绪分数 < -0.5 的股票直接剔除
            VETO_THRESHOLD = -0.5
            
            filtered_candidates: List[str] = []
            low_confidence_count = 0
            negative_sentiment_count = 0
            vetoed_stocks: List[str] = []
            
            for _, row in sentiment_df.iterrows():
                stock_code = row["stock_code"]
                score = row["score"]
                confidence = row["confidence"]
                
                # 规则1: 一票否决 - 情绪分数 < -0.5 直接剔除
                if score < VETO_THRESHOLD:
                    negative_sentiment_count += 1
                    vetoed_stocks.append(stock_code)
                    logger.warning(
                        f"风控剔除: {stock_code} 情绪分 {score:.2f} < {VETO_THRESHOLD}"
                    )
                    continue
                
                # 规则2: 检查置信度
                if confidence < self._min_confidence:
                    low_confidence_count += 1
                    logger.warning(
                        f"Skipping {stock_code} due to low confidence: "
                        f"confidence={confidence:.2f} < min={self._min_confidence}"
                    )
                    continue
                
                # 通过所有检查
                filtered_candidates.append(stock_code)
            
            # 汇总日志
            if vetoed_stocks:
                logger.warning(
                    f"🚨 情绪风控一票否决: {len(vetoed_stocks)} 只股票被剔除 "
                    f"(情绪分 < {VETO_THRESHOLD}): {vetoed_stocks}"
                )
            
            logger.info(
                f"情绪过滤完成 ({date_str}): "
                f"输入 {len(candidates)} 只, "
                f"通过 {len(filtered_candidates)} 只, "
                f"一票否决剔除 {negative_sentiment_count} 只, "
                f"低置信度剔除 {low_confidence_count} 只"
            )
            
            return filtered_candidates
        
        except LLMCircuitBreakerError as e:
            # ===== 熔断器触发: 停止交易 =====
            logger.critical(
                f"⛔ LLM Circuit Breaker Triggered! Risk control failed. "
                f"HALTING TRADING SIGNALS. Error: {e}"
            )
            # 直接抛出异常，完全停止交易流程
            raise
        
        except Exception as e:
            # ===== 其他异常: Fail-Closed 策略 =====
            # 返回空列表而非原候选列表，避免在 LLM 风控不可用时产生买入信号
            logger.critical(
                f"⛔ LLM Sentiment Analysis Failed ({date_str}): {e}. "
                f"Fail-Closed: 阻止所有 {len(candidates)} 只候选股票的买入信号. "
                f"原因: LLM 风控不可用时不应产生交易信号."
            )
            return []
    
    def select_top_stocks(
        self,
        data: pd.DataFrame,
        n: Optional[int] = None,
        date: Optional[pd.Timestamp] = None,
        use_sentiment_scoring: bool = True
    ) -> List[str]:
        """
        两阶段选股：技术面初筛 + 情绪面加成
        
        实现"情绪进攻型"策略的核心选股逻辑：
        
        **第一阶段（技术面初筛）**：
        仅根据技术因子（Momentum + Turnover + Size 等）计算基础得分，
        选出前 N * buffer_multiplier 只候选股票。
        
        **第二阶段（情绪面加成）**：
        对候选股调用 LLM 情绪分析引擎，获取情绪分数（-1 到 1）。
        
        **第三阶段（最终排名）**：
        Final_Score = Base_Score + Sentiment_Weight * Sentiment_Score
        根据最终得分选出 Top N。
        
        Parameters
        ----------
        data : pd.DataFrame
            包含因子数据的 DataFrame
        n : Optional[int]
            选取数量，默认使用 self.top_n
        date : Optional[pd.Timestamp]
            分析日期，用于情绪分析。如果启用情绪分析但未传入日期，
            则使用当前日期或从数据中推断。
        use_sentiment_scoring : bool
            是否启用情绪进攻型加分。默认 True。
            设置为 False 可强制使用纯技术面选股。
        
        Returns
        -------
        List[str]
            选中的股票代码列表
        
        Notes
        -----
        - 如果 LLM 调用失败，自动降级为仅使用技术面得分排序
        - 情绪分数范围为 -1 到 1，会乘以 sentiment_weight 加到基础分上
        """
        n = n or self.top_n
        
        if data.empty:
            return []
        
        # 确定股票代码列
        stock_col = self.stock_col if self.stock_col in data.columns else 'symbol'
        
        # ==========================================
        # 数据预处理：确保每只股票只有一条记录（使用最新日期）
        # ==========================================
        data = data.copy()
        
        # 确定日期列
        date_col_name = None
        if self.date_col in data.columns:
            date_col_name = self.date_col
        elif 'trade_date' in data.columns:
            date_col_name = 'trade_date'
        elif isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            date_col_name = 'index' if 'index' in data.columns else data.columns[0]
        elif isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()
            date_col_name = data.columns[0]
        
        # 去重：每只股票只保留最新日期的记录
        if stock_col in data.columns and date_col_name is not None:
            data = data.sort_values(date_col_name, ascending=False)
            data = data.drop_duplicates(subset=[stock_col], keep='first')
            logger.debug(f"数据去重完成：保留 {len(data)} 只股票的最新记录")
        
        # ==========================================
        # 第一阶段：技术面初筛（基础得分）
        # ==========================================
        # 使用 return_components=True 获取各因子得分分量
        data['base_score'] = self.calculate_total_score(data, sentiment_scores=None, return_components=True)
        
        # [NEW] 应用持股惯性加分 - 当前持仓股票获得额外得分
        if self.holding_bonus > 0 and 'is_holding' in data.columns:
            holding_mask = data['is_holding'] == True
            n_holdings = holding_mask.sum()
            if n_holdings > 0:
                data.loc[holding_mask, 'base_score'] += self.holding_bonus
                logger.debug(f"持股惯性加分: {n_holdings} 只持仓股票获得 +{self.holding_bonus:.2f} 加成")
        
        # 剔除得分为 NaN 的股票
        valid_data = data.dropna(subset=['base_score'])
        
        if valid_data.empty:
            return []
        
        # 判断是否需要进行情绪面加成
        should_use_sentiment = (
            use_sentiment_scoring
            and self._enable_sentiment_filter
            and self._sentiment_engine is not None
            and self.sentiment_weight > 0
        )
        
        if not should_use_sentiment:
            # 纯技术面选股：直接返回 Top N
            if stock_col not in valid_data.columns:
                if isinstance(valid_data.index, pd.MultiIndex):
                    top_df = valid_data.nlargest(n, 'base_score')
                    top_stocks = top_df.index.get_level_values(-1).tolist()
                else:
                    top_df = valid_data.nlargest(n, 'base_score')
                    top_stocks = top_df.index.tolist()
            else:
                top_df = valid_data.nlargest(n, 'base_score')
                top_stocks = top_df[stock_col].tolist()
            
            # 确保去重并保持顺序
            top_stocks = list(dict.fromkeys(top_stocks))[:n]
            
            # ===== 输出选中股票的详细得分构成 =====
            self._log_selected_stocks_scores(top_df, stock_col, top_stocks)
            
            return top_stocks
        
        # ==========================================
        # 第一阶段：选出扩展候选列表（Top N * buffer）
        # ==========================================
        buffer_n = n * self._sentiment_buffer_multiplier
        
        if stock_col not in valid_data.columns:
            if isinstance(valid_data.index, pd.MultiIndex):
                pre_selected = valid_data.nlargest(buffer_n, 'base_score')
                pre_candidates = pre_selected.index.get_level_values(-1).tolist()
            else:
                pre_selected = valid_data.nlargest(buffer_n, 'base_score')
                pre_candidates = pre_selected.index.tolist()
        else:
            pre_selected = valid_data.nlargest(buffer_n, 'base_score')
            pre_candidates = pre_selected[stock_col].tolist()
        
        logger.debug(
            f"第一阶段技术面初筛: 共 {len(valid_data)} 只股票 -> "
            f"选出 {len(pre_candidates)} 只候选股 (buffer={self._sentiment_buffer_multiplier}x)"
        )
        
        # ==========================================
        # 第二阶段：情绪面加成（LLM 分析）
        # ==========================================
        sentiment_scores: Optional[pd.Series] = None
        
        # 确定分析日期
        if date is None:
            if self.date_col in data.columns:
                date = pd.to_datetime(data[self.date_col]).max()
            elif isinstance(data.index, pd.DatetimeIndex):
                date = data.index.max()
            elif isinstance(data.index, pd.MultiIndex):
                date = data.index.get_level_values(0).max()
            else:
                date = pd.Timestamp.now()
        
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        # 一票否决阈值：情绪分数 < -0.5 的股票直接剔除
        VETO_THRESHOLD = -0.5
        vetoed_stocks: List[str] = []
        
        try:
            # 调用情绪分析引擎
            sentiment_df = self._sentiment_engine.calculate_sentiment(pre_candidates, date_str)
            
            if sentiment_df.empty:
                logger.warning(
                    f"情绪分析返回空结果 ({date_str}), "
                    f"降级为纯技术面选股"
                )
            else:
                # ========== 一票否决逻辑 ==========
                # 情绪分数 < -0.5 的股票直接从候选列表中剔除
                for _, row in sentiment_df.iterrows():
                    stock_code = row["stock_code"]
                    score = row["score"]
                    
                    if score < VETO_THRESHOLD:
                        vetoed_stocks.append(stock_code)
                        logger.warning(
                            f"风控剔除: {stock_code} 情绪分 {score:.2f} < {VETO_THRESHOLD}"
                        )
                
                # 汇总一票否决日志
                if vetoed_stocks:
                    logger.warning(
                        f"🚨 情绪风控一票否决: {len(vetoed_stocks)} 只股票被剔除 "
                        f"(情绪分 < {VETO_THRESHOLD}): {vetoed_stocks}"
                    )
                    # 从候选列表中移除被否决的股票
                    pre_candidates = [s for s in pre_candidates if s not in vetoed_stocks]
                
                # ========== 构建情绪分数 Series ==========
                # 仅对 score > 0 的股票进行加分，score <= 0 时不加分（设为 0）
                raw_scores = pd.Series(
                    sentiment_df['score'].values,
                    index=sentiment_df['stock_code'].values
                )
                
                # 只保留正分用于加分，负分和零分不加分（设为0）
                sentiment_scores = raw_scores.clip(lower=0)
                
                positive_count = (raw_scores > 0).sum()
                neutral_count = ((raw_scores <= 0) & (raw_scores >= VETO_THRESHOLD)).sum()
                
                logger.info(
                    f"情绪分析完成 ({date_str}): "
                    f"原始 {len(raw_scores)} 只, "
                    f"正面加分 {positive_count} 只, "
                    f"中性不加分 {neutral_count} 只, "
                    f"一票否决 {len(vetoed_stocks)} 只"
                )
        
        except LLMCircuitBreakerError:
            # 熔断器触发：直接抛出，由上层处理
            raise
        
        except Exception as e:
            # ===== LLM 异常处理：降级为纯技术面 =====
            logger.warning(
                f"⚠️ 情绪分析失败 ({date_str}): {e}. "
                f"降级为纯技术面选股，不阻断交易流程。"
            )
            sentiment_scores = None
        
        # ==========================================
        # 第三阶段：最终排名
        # Final_Score = Base_Score + Sentiment_Weight * Sentiment_Score
        # 注意：sentiment_scores 已处理，仅正分会被加入
        # ==========================================
        # 筛选出候选股的数据子集（排除被一票否决的股票）
        if stock_col in valid_data.columns:
            candidate_mask = valid_data[stock_col].isin(pre_candidates)
        else:
            if isinstance(valid_data.index, pd.MultiIndex):
                candidate_mask = valid_data.index.get_level_values(-1).isin(pre_candidates)
            else:
                candidate_mask = valid_data.index.isin(pre_candidates)
        
        candidate_data = valid_data[candidate_mask].copy()
        
        # 计算最终得分（包含情绪分数，仅正分加分）
        candidate_data['final_score'] = self.calculate_total_score(
            candidate_data,
            sentiment_scores=sentiment_scores
        )
        
        # 选取最终 Top N
        if stock_col not in candidate_data.columns:
            if isinstance(candidate_data.index, pd.MultiIndex):
                top_stocks = candidate_data.nlargest(n, 'final_score').index.get_level_values(-1).tolist()
            else:
                top_stocks = candidate_data.nlargest(n, 'final_score').index.tolist()
        else:
            top_stocks = candidate_data.nlargest(n, 'final_score')[stock_col].tolist()
        
        # 确保去重并保持顺序
        top_stocks = list(dict.fromkeys(top_stocks))[:n]
        
        logger.debug(
            f"第三阶段最终排名: {len(pre_candidates)} 只候选股 -> "
            f"选出 {len(top_stocks)} 只目标股票 "
            f"(一票否决剔除 {len(vetoed_stocks)} 只)"
        )
        
        return top_stocks
    
    def _log_selected_stocks_scores(
        self,
        top_df: pd.DataFrame,
        stock_col: str,
        top_stocks: List[str]
    ) -> None:
        """
        输出选中股票的详细得分构成（用于人工复核）
        
        Parameters
        ----------
        top_df : pd.DataFrame
            包含选中股票数据的 DataFrame
        stock_col : str
            股票代码列名
        top_stocks : List[str]
            选中的股票代码列表
        """
        # 定义因子列映射
        factor_cols = {
            'value': (self.value_col, self.value_weight),
            'quality': (self.quality_col, self.quality_weight),
            'momentum': (self.momentum_col, self.momentum_weight),
            'size': (self.size_col, self.size_weight),
        }
        
        logger.info("=" * 70)
        logger.info(f"📊 选中股票详细得分构成 (Top {len(top_stocks)})")
        logger.info("=" * 70)
        logger.info(
            f"因子权重: Value={self.value_weight:.2f}, Quality={self.quality_weight:.2f}, "
            f"Momentum={self.momentum_weight:.2f}, Size={self.size_weight:.2f}"
        )
        logger.info("-" * 70)
        
        for rank, stock in enumerate(top_stocks, 1):
            # 获取该股票的数据行
            if stock_col in top_df.columns:
                stock_row = top_df[top_df[stock_col] == stock]
            else:
                stock_row = top_df.loc[[stock]] if stock in top_df.index else pd.DataFrame()
            
            if stock_row.empty:
                logger.info(f"  {rank}. {stock}: 数据缺失")
                continue
            
            stock_row = stock_row.iloc[0]
            
            # 获取各因子原始 Z-Score 和贡献分
            score_parts = []
            total = stock_row.get('base_score', 0)
            
            for factor_name, (col_name, weight) in factor_cols.items():
                if weight > 0 and col_name in top_df.columns:
                    raw_z = stock_row.get(col_name, 0)
                    contribution = weight * raw_z if pd.notna(raw_z) else 0
                    score_parts.append(f"{factor_name.capitalize()}({raw_z:.2f}×{weight:.1f}={contribution:.2f})")
                elif weight > 0:
                    score_parts.append(f"{factor_name.capitalize()}(N/A)")
            
            # 检查是否有持股惯性加分
            holding_bonus_str = ""
            if '_value_score' in top_df.columns:
                # 使用分解的得分
                v_score = stock_row.get('_value_score', 0)
                q_score = stock_row.get('_quality_score', 0)
                m_score = stock_row.get('_momentum_score', 0)
                s_score = stock_row.get('_size_score', 0)
                component_total = v_score + q_score + m_score + s_score
                if abs(total - component_total) > 0.01:
                    # 有额外加分（如持股惯性）
                    holding_bonus_str = f" +惯性={total - component_total:.2f}"
            
            logger.info(
                f"  {rank}. {stock} | 总分={total:.3f}{holding_bonus_str} | "
                f"{' + '.join(score_parts)}"
            )
        
        logger.info("=" * 70)
    
    def generate_target_positions(
        self,
        data: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        生成目标持仓矩阵
        
        每月最后一个交易日进行调仓，选取 Total_Score 最高的 Top N 只股票。
        
        Parameters
        ----------
        data : pd.DataFrame
            因子数据，必须包含：
            - 日期列（date 或 DatetimeIndex）
            - 股票代码列（stock_code 或 symbol）
            - 价值因子 Z-Score 列
            - 质量因子 Z-Score 列
            - 动量因子 Z-Score 列
            - is_limit: 涨跌停标志（可选）
            - listing_days 或 list_date: 上市天数/日期（可选）
        start_date : Optional[pd.Timestamp]
            开始日期
        end_date : Optional[pd.Timestamp]
            结束日期
        benchmark_data : Optional[pd.DataFrame]
            基准指数数据（如沪深300），用于大盘风控。
            需包含 'close' 列，索引为 DatetimeIndex。
            如果为 None，则跳过大盘风控逻辑。
        
        Returns
        -------
        pd.DataFrame
            布尔型 DataFrame，Index=Date, Columns=Symbol
            True 代表持有该股票
        
        Examples
        --------
        >>> strategy = MultiFactorStrategy("MF", {"top_n": 30})
        >>> # 不启用大盘风控
        >>> positions = strategy.generate_target_positions(factor_data)
        >>> 
        >>> # 启用大盘风控
        >>> hs300_data = data_loader.fetch_index_price("000300", "2020-01-01", "2024-12-31")
        >>> positions = strategy.generate_target_positions(factor_data, benchmark_data=hs300_data)
        >>> print(positions.sum(axis=1))  # 每日持仓数量
        """
        # 确定日期列
        if self.date_col in data.columns:
            dates_array = pd.to_datetime(data[self.date_col].unique())
        elif isinstance(data.index, pd.DatetimeIndex):
            dates_array = data.index.unique()
        elif isinstance(data.index, pd.MultiIndex):
            dates_array = data.index.get_level_values(0).unique()
        else:
            raise ValueError("无法确定日期列，请检查数据格式或配置 date_col")
        
        # 确定股票列
        stock_col = self.stock_col if self.stock_col in data.columns else 'symbol'
        if stock_col in data.columns:
            all_stocks = data[stock_col].unique()
        elif isinstance(data.index, pd.MultiIndex):
            all_stocks = data.index.get_level_values(-1).unique()
        else:
            raise ValueError("无法确定股票代码列，请检查数据格式或配置 stock_col")
        
        # 排序日期
        all_dates = pd.DatetimeIndex(sorted(dates_array))
        
        # 应用日期过滤
        if start_date is not None:
            all_dates = all_dates[all_dates >= start_date]
        if end_date is not None:
            all_dates = all_dates[all_dates <= end_date]
        
        if all_dates.empty:
            logger.warning("过滤后无有效日期")
            return pd.DataFrame()
        
        # 根据配置的频率获取调仓日期
        rebalance_dates = set(self.get_rebalance_dates(all_dates, self.rebalance_frequency))
        
        logger.info(
            f"调仓日期数量: {len(rebalance_dates)} ({self.rebalance_frequency})"
        )

        # === 大盘风控准备（激进版：MA60 + 20日跌幅判断）===
        # 激进策略使用更宽松的风控条件，避免踏空行情
        # 默认为 False (无风险)
        market_risk_series = pd.Series(False, index=all_dates)
        
        if benchmark_data is not None and not benchmark_data.empty:
            try:
                index_df = benchmark_data.copy()
                
                # 确保索引是 DatetimeIndex
                if not isinstance(index_df.index, pd.DatetimeIndex):
                    if 'date' in index_df.columns:
                        index_df['date'] = pd.to_datetime(index_df['date'])
                        index_df = index_df.set_index('date')
                    else:
                        index_df.index = pd.to_datetime(index_df.index)
                
                index_df = index_df.sort_index()
                
                # ===== 激进版风控：MA60（牛熊线）+ 20日跌幅 =====
                # 计算60日均线（牛熊线）
                index_df['ma60'] = index_df['close'].rolling(window=60).mean()
                
                # 计算20天前跌幅
                # drop_20d = (close_today - close_20d_ago) / close_20d_ago
                drop_lookback = 20
                index_df['drop_20d'] = (
                    index_df['close'] - index_df['close'].shift(drop_lookback)
                ) / index_df['close'].shift(drop_lookback)
                
                # 对齐到策略日期范围
                # 使用 ffill 填充非交易日的空缺 (如果有的话)
                aligned_index = index_df.reindex(all_dates, method='ffill')
                
                # ===== 趋势风控条件（OR 逻辑）=====
                # 原规则（AND）: (Close < MA60) AND (20日跌幅 > 5%)
                #   问题：缓慢阴跌的熊市中不会触发，非常危险
                # 新规则（OR）: (Close < MA60) OR (20日跌幅 > 5%)
                # 
                # 逻辑说明：
                # 1. 只要跌破 MA60（牛熊线），就触发风控
                # 2. 或者发生暴跌（20日跌幅超5%），也触发风控
                # 3. 两者满足其一即空仓，更加安全
                drop_threshold = -0.05  # 20日跌幅阈值（-5%）
                
                condition_below_ma60 = aligned_index['close'] < aligned_index['ma60']
                condition_crash = aligned_index['drop_20d'] < drop_threshold
                
                # 使用 OR 逻辑：只要满足其一即触发风控
                market_risk_series = (condition_below_ma60 | condition_crash).fillna(False)
                
                logger.info(
                    f"已使用趋势风控 (OR 逻辑): (Close < MA60) OR (20日跌幅 < {drop_threshold:.0%})"
                )
                
            except Exception as e:
                logger.warning(f"计算大盘风控指标失败: {e}，风控功能暂时失效")
        else:
            logger.debug("未传入基准数据，大盘风控未启用")
        # ===========================
        
        # 初始化目标持仓矩阵
        target_positions = pd.DataFrame(
            False,
            index=all_dates,
            columns=sorted(all_stocks),
            dtype=bool
        )
        target_positions.index.name = 'date'
        target_positions.columns.name = 'symbol'
        
        # 当前持仓
        current_holdings: List[str] = []
        
        for date in all_dates:
            # 1. 每日风控检查
            is_risk_triggered = market_risk_series.loc[date]
            
            # 2. 调仓逻辑（采用两阶段选股模式：技术面初筛 + 情绪面加成）
            if date in rebalance_dates:
                # 调仓日: 重新选股 (无论是否有风控，都更新选股列表以备后用)
                filtered_data = self.filter_stocks(data, date)
                
                if not filtered_data.empty:
                    try:
                        # 使用新的两阶段选股方法
                        # select_top_stocks 内部已实现：
                        # 1. 技术面初筛 -> Top N * buffer 候选股
                        # 2. 情绪分析获取情绪分数
                        # 3. 最终排名 = 基础分 + 情绪权重 * 情绪分
                        current_holdings = self.select_top_stocks(
                            filtered_data,
                            n=self.top_n,
                            date=date,
                            use_sentiment_scoring=True
                        )
                        
                        logger.debug(
                            f"调仓日 {date.strftime('%Y-%m-%d')}: "
                            f"两阶段选股完成, 选中 {len(current_holdings)} 只股票"
                        )
                    
                    except LLMCircuitBreakerError:
                        # ===== 熔断器触发: 停止交易 =====
                        # 直接抛出，由调用方处理
                        raise
                else:
                    logger.warning(f"调仓日 {date.strftime('%Y-%m-%d')}: 无可选股票")
                    current_holdings = []
            
            # 3. 风控处理
            if is_risk_triggered:
                # 仅在调仓日记录日志，避免日志爆炸
                if date in rebalance_dates:
                    logger.warning(f"日期 {date.strftime('%Y-%m-%d')}: 大盘跌破20日均线，系统强制空仓")
                # 触发风控时，直接跳过持仓设置 (保持为 False)
                continue
            
            # 4. 设置当日持仓 (如果无风控)
            for stock in current_holdings:
                if stock in target_positions.columns:
                    target_positions.loc[date, stock] = True
        
        # 统计信息
        
        # 统计信息
        total_trades = (target_positions.astype(int).diff().abs().sum().sum()) // 2
        avg_holdings = target_positions.sum(axis=1).mean()
        logger.info(
            f"目标持仓矩阵生成完成: "
            f"日期范围 {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}, "
            f"平均持仓 {avg_holdings:.1f} 只, 预计换手次数 {total_trades:.0f}"
        )
        
        return target_positions
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号（兼容基类接口）
        
        对于多因子策略，主要使用 generate_target_positions 方法。
        此方法提供简化的单股票信号生成，用于兼容基类接口。
        
        Parameters
        ----------
        data : pd.DataFrame
            价格和因子数据
        
        Returns
        -------
        pd.Series
            交易信号序列，1=买入，-1=卖出，0=持有
        """
        # 检查是否有所需因子列
        has_factors = all(
            col in data.columns
            for col in [self.value_col, self.quality_col, self.momentum_col]
        )
        
        if not has_factors:
            logger.warning("数据中缺少因子列，返回空信号")
            return pd.Series(0, index=data.index)
        
        # 计算综合得分
        total_score = self.calculate_total_score(data)
        
        # 基于分位数生成信号
        signals = pd.Series(0, index=data.index)
        
        # 高分（Top 10%）买入
        high_threshold = total_score.quantile(0.9)
        signals[total_score >= high_threshold] = 1
        
        # 低分（Bottom 10%）卖出
        low_threshold = total_score.quantile(0.1)
        signals[total_score <= low_threshold] = -1
        
        return signals
    
    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float
    ) -> float:
        """
        计算仓位大小
        
        使用等权重分配策略，每只股票分配相等的资金。
        
        Parameters
        ----------
        signal : TradeSignal
            交易信号
        portfolio_value : float
            组合总价值
        
        Returns
        -------
        float
            建议仓位金额
        """
        # 等权重分配
        base_size = portfolio_value / self.top_n
        adjusted_size = base_size * signal.strength
        
        return adjusted_size
    
    def get_rebalance_summary(
        self,
        target_positions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        获取调仓汇总信息
        
        Parameters
        ----------
        target_positions : pd.DataFrame
            目标持仓矩阵
        
        Returns
        -------
        pd.DataFrame
            调仓汇总，包含每个调仓日的买入/卖出股票数量
        """
        if target_positions.empty:
            return pd.DataFrame()
        
        # 计算每日变化
        position_change = target_positions.astype(int).diff()
        
        # 获取调仓日
        rebalance_dates = self.get_month_end_dates(target_positions.index)
        
        summary_records = []
        
        for date in rebalance_dates:
            if date not in position_change.index:
                continue
            
            day_change = position_change.loc[date]
            
            # 买入股票（0 -> 1）
            buy_stocks = day_change[day_change == 1].index.tolist()
            
            # 卖出股票（1 -> 0）
            sell_stocks = day_change[day_change == -1].index.tolist()
            
            # 持有股票
            hold_stocks = target_positions.loc[date][target_positions.loc[date]].index.tolist()
            
            summary_records.append({
                'date': date,
                'buy_count': len(buy_stocks),
                'sell_count': len(sell_stocks),
                'hold_count': len(hold_stocks),
                'buy_stocks': buy_stocks,
                'sell_stocks': sell_stocks,
            })
        
        return pd.DataFrame(summary_records)
    
    # ==================== 权重优化 ====================
    
    def optimize_weights(
        self,
        prices: pd.DataFrame,
        selected_stocks: List[str],
        objective: str = "max_sharpe",
        risk_free_rate: float = 0.02,
        max_weight: Optional[float] = None,
        min_weight: float = 0.0,
        lookback_days: int = 252
    ) -> Dict[str, float]:
        """
        使用 PyPortfolioOpt 优化投资组合权重
        
        基于选中股票的历史价格，计算最优权重配置。
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据，索引为日期，列为股票代码
        selected_stocks : List[str]
            选中的股票列表
        objective : str, optional
            优化目标，可选：
            - 'max_sharpe': 最大夏普比率（默认）
            - 'min_volatility': 最小波动率
        risk_free_rate : float, optional
            无风险利率，默认0.02（2%）
        max_weight : Optional[float]
            单只股票最大权重，默认0.05（5%）
            如果为None，使用 1/top_n 或 0.05 中的较小值
        min_weight : float, optional
            单只股票最小权重，默认0.0
        lookback_days : int, optional
            回溯天数用于计算协方差，默认252
        
        Returns
        -------
        Dict[str, float]
            优化后的权重字典，股票代码 -> 权重
        
        Examples
        --------
        >>> strategy = MultiFactorStrategy("MF", {"top_n": 30})
        >>> weights = strategy.optimize_weights(
        ...     prices_df, selected_stocks, objective='max_sharpe'
        ... )
        >>> print(weights)
        
        Notes
        -----
        - 使用 Ledoit-Wolf 压缩协方差矩阵以提高稳定性
        - 单只股票权重默认不超过 5%
        - 优化失败时返回等权重分配
        """
        try:
            from pypfopt import EfficientFrontier, risk_models, expected_returns
        except ImportError:
            logger.warning(
                "未安装 pypfopt，使用等权重分配。"
                "安装命令: pip install pyportfolioopt"
            )
            return self._equal_weights(selected_stocks)
        
        # 设置默认最大权重
        if max_weight is None:
            max_weight = min(0.05, 1.0 / len(selected_stocks))
        
        # 过滤价格数据
        available_stocks = [s for s in selected_stocks if s in prices.columns]
        
        if len(available_stocks) < 2:
            logger.warning("可用股票数少于2，使用等权重分配")
            return self._equal_weights(selected_stocks)
        
        # 获取回溯期价格数据
        stock_prices = prices[available_stocks].tail(lookback_days).dropna(axis=1)
        
        if stock_prices.shape[1] < 2:
            logger.warning("有效价格数据的股票数少于2，使用等权重分配")
            return self._equal_weights(selected_stocks)
        
        try:
            # 如果优化目标是等权重，直接返回
            if objective == "equal_weight":
                logger.info("使用等权重分配策略")
                return self._equal_weights(selected_stocks)
            
            # 计算预期收益率
            mu = expected_returns.mean_historical_return(stock_prices)
            
            # 使用 Ledoit-Wolf 压缩协方差矩阵
            S = risk_models.CovarianceShrinkage(stock_prices).ledoit_wolf()
            
            # 创建有效边界优化器
            ef = EfficientFrontier(
                mu, S,
                weight_bounds=(min_weight, max_weight)
            )
            
            # 执行优化
            if objective == "max_sharpe":
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == "min_volatility":
                weights = ef.min_volatility()
            else:
                raise ValueError(f"不支持的优化目标: {objective}")
            
            # 清理权重
            clean_weights = ef.clean_weights(cutoff=1e-4, rounding=4)
            
            # 获取绩效指标
            performance = ef.portfolio_performance(
                verbose=False,
                risk_free_rate=risk_free_rate
            )
            
            logger.info(
                f"权重优化完成 [{objective}]: "
                f"预期收益 {performance[0]:.2%}, "
                f"波动率 {performance[1]:.2%}, "
                f"夏普比率 {performance[2]:.2f}"
            )
            
            return clean_weights
            
        except Exception as e:
            logger.warning(f"权重优化失败: {e}，使用等权重分配")
            return self._equal_weights(available_stocks)
    
    def _equal_weights(self, stocks: List[str]) -> Dict[str, float]:
        """
        生成等权重分配
        
        Parameters
        ----------
        stocks : List[str]
            股票列表
        
        Returns
        -------
        Dict[str, float]
            等权重字典
        """
        if not stocks:
            return {}
        weight = 1.0 / len(stocks)
        return {stock: weight for stock in stocks}
    
    def generate_target_weights(
        self,
        factor_data: pd.DataFrame,
        prices: pd.DataFrame,
        objective: str = "equal_weight",
        risk_free_rate: float = 0.02,
        max_weight: Optional[float] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        current_holdings_weights: Optional[Dict[str, float]] = None,
        rebalance_threshold: Optional[float] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        生成带权重的目标持仓矩阵（含再平衡缓冲区 + 大盘风控）
        
        每月最后一个交易日进行调仓，使用优化权重或等权重。
        为避免小资金账户支付最低5元佣金的成本，仅当权重变化超过阈值时才调整。
        
        **关键特性**：当大盘（如沪深300）跌破20日均线且均线向下倾斜时，
        强制清仓以规避系统性风险。
        
        Parameters
        ----------
        factor_data : pd.DataFrame
            因子数据
        prices : pd.DataFrame
            价格数据，索引为日期，列为股票代码
        objective : str, optional
            优化目标，可选:
            - 'equal_weight': 等权重分配（推荐小资金账户，默认）
            - 'max_sharpe': 最大夏普比率
            - 'min_volatility': 最小波动率
        risk_free_rate : float, optional
            无风险利率，默认0.02
        max_weight : Optional[float]
            单只股票最大权重，默认根据 top_n 计算
        start_date : Optional[pd.Timestamp]
            开始日期
        end_date : Optional[pd.Timestamp]
            结束日期
        current_holdings_weights : Optional[Dict[str, float]]
            当前持仓权重字典，用于懒惰再平衡计算。
            如果为 None，则使用前一日的目标权重。
            
            **⚠️ 重要警告 - Drifted Weights 要求**：
            
            此参数必须传入 **漂移后权重（Drifted Weights）**，即：
            
            ``weight[stock] = (最新收盘价 × 持仓股数) / 当前总资产``
            
            **错误用法**：直接传入上期目标权重（Target Weights）
            
            **原因**：懒惰再平衡的核心思想是"Let profits run"（让盈利奔跑）。
            如果某只股票大涨，其市值权重会自然漂移增大，我们应该保持这个
            增大后的权重，而不是强行卖出盈利部分回归到上期目标权重。
            
            **示例**：
            - 上期目标权重：A股 20%, B股 20%, C股 20%, D股 20%, E股 20%
            - A股大涨50%后漂移权重：A股 26.7%, B股 18.3%, C股 18.3%, D股 18.3%, E股 18.3%
            - 正确：传入漂移权重 → A股保持26.7%，不产生交易
            - 错误：传入目标权重 → 系统会卖出A股6.7%的仓位，违背动量原则
            
        rebalance_threshold : Optional[float]
            再平衡阈值，默认使用 self.rebalance_buffer (5%)。
            仅当 |new_weight - current_weight| > 阈值时才调整该股票仓位。
            用于避免小资金账户频繁交易产生最低5元佣金。
        benchmark_data : Optional[pd.DataFrame]
            基准指数数据（如沪深300），用于大盘风控。
            需包含 'close' 列，索引为 DatetimeIndex。
            如果为 None，则跳过大盘风控逻辑。
        
        Returns
        -------
        pd.DataFrame
            权重 DataFrame，Index=Date, Columns=Symbol
            值为权重（0-1之间），0表示不持有
        
        Notes
        -----
        懒惰再平衡逻辑（Lazy Rebalance，适用于30万小资金账户）：
        
        **背景**：最低5元佣金导致小资金账户的交易摩擦成本极高。
        例如：30万资金，5%权重 = 1.5万，按万三计算佣金仅4.5元，不足最低5元。
        
        **核心规则（只换股，不调仓）**：
        
        1. 继续持有：如果股票仍在选中列表中，**直接沿用当前权重**，不产生任何交易
           - 效果：避免了因权重微调产生的无效买卖单
        
        2. 卖出：如果股票不再在选中列表中，权重设为0
           - 释放的仓位用于买入新股票
        
        3. 买入：新进入选中列表的股票，用卖出释放的仓位进行等权分配
           - 只有换股才会产生交易
        
        **效果示例（5只股票组合）**：
        - 全部继续入选 → 本期零交易
        - 换1只股票 → 仅1买1卖共2笔交易
        - 换2只股票 → 仅2买2卖共4笔交易
        
        **大盘风控规则（Market Risk Control）**：
        - 条件：(Close < MA60) AND (20日跌幅 < -5%)
        - 触发时：强制清空所有仓位
        - 目的：规避系统性下跌风险
        
        **⚠️ Drifted Weights 与动量策略的关系**：
        
        懒惰再平衡的核心哲学是 "Let profits run, cut losses short"。
        当传入正确的漂移后权重时：
        
        - 盈利股票的权重自然增大 → 保持增大后的权重 → 让利润奔跑
        - 亏损股票的权重自然缩小 → 保持缩小后的权重 → 减少损失敞口
        
        这与动量策略的理念高度一致：强者恒强，弱者恒弱。
        
        **计算 Drifted Weights 的公式**::
        
            for stock, shares in current_positions.items():
                market_value = latest_price[stock] * shares
                drifted_weight[stock] = market_value / total_portfolio_value
        
        Examples
        --------
        >>> strategy = MultiFactorStrategy("MF", {"top_n": 5, "rebalance_buffer": 0.05})
        >>> # 启用大盘风控
        >>> hs300_data = data_loader.fetch_index_price("000300", "2020-01-01", "2024-12-31")
        >>> weights = strategy.generate_target_weights(
        ...     factor_data, prices_df, objective='equal_weight',
        ...     benchmark_data=hs300_data
        ... )
        >>> print(weights.sum(axis=1))  # 每日权重之和（应接近1，风控时为0）
        """
        # 使用配置的再平衡阈值，或使用参数覆盖
        buffer_threshold = rebalance_threshold if rebalance_threshold is not None else self.rebalance_buffer
        
        # 设置默认最大权重
        if max_weight is None:
            max_weight = min(0.25, 1.0 / self.top_n)  # 对于5只股票，最大权重0.25
        
        # 确定日期列
        if self.date_col in factor_data.columns:
            dates_array = pd.to_datetime(factor_data[self.date_col].unique())
        elif isinstance(factor_data.index, pd.DatetimeIndex):
            dates_array = factor_data.index.unique()
        elif isinstance(factor_data.index, pd.MultiIndex):
            dates_array = factor_data.index.get_level_values(0).unique()
        else:
            raise ValueError("无法确定日期列")
        
        # 确定股票列
        stock_col = self.stock_col if self.stock_col in factor_data.columns else 'symbol'
        if stock_col in factor_data.columns:
            all_stocks = factor_data[stock_col].unique()
        elif isinstance(factor_data.index, pd.MultiIndex):
            all_stocks = factor_data.index.get_level_values(-1).unique()
        else:
            raise ValueError("无法确定股票代码列")
        
        # 排序日期
        all_dates = pd.DatetimeIndex(sorted(dates_array))
        
        # 应用日期过滤
        if start_date is not None:
            all_dates = all_dates[all_dates >= start_date]
        if end_date is not None:
            all_dates = all_dates[all_dates <= end_date]
        
        if all_dates.empty:
            logger.warning("过滤后无有效日期")
            return pd.DataFrame()
        
        # 根据配置的频率获取调仓日期
        rebalance_dates = set(self.get_rebalance_dates(all_dates, self.rebalance_frequency))
        
        logger.info(
            f"调仓日期数量: {len(rebalance_dates)} ({self.rebalance_frequency}), "
            f"再平衡缓冲区: {buffer_threshold:.1%}, 优化目标: {objective}"
        )
        
        # ========================================
        # 大盘风控准备（激进版 Market Risk Control）
        # ========================================
        # 激进策略使用 MA60 + 20日跌幅判断，避免踏空行情
        # 默认为 False (无风险)
        market_risk_series = pd.Series(False, index=all_dates)
        risk_triggered_days = 0
        
        # [FIX] 检查风控是否启用
        if self._market_risk_enabled and benchmark_data is not None and not benchmark_data.empty:
            try:
                index_df = benchmark_data.copy()
                
                # 确保索引是 DatetimeIndex
                if not isinstance(index_df.index, pd.DatetimeIndex):
                    if 'date' in index_df.columns:
                        index_df['date'] = pd.to_datetime(index_df['date'])
                        index_df = index_df.set_index('date')
                    else:
                        index_df.index = pd.to_datetime(index_df.index)
                
                index_df = index_df.sort_index()
                
                # ===== 激进版风控：MA60（牛熊线）+ 20日跌幅 =====
                # 计算60日均线（牛熊线）- 从配置读取周期
                ma_period = self._market_risk_ma_period
                index_df['ma60'] = index_df['close'].rolling(window=ma_period).mean()
                
                # 计算跌幅 - 从配置读取回溯天数
                drop_lookback = self._market_risk_drop_lookback
                index_df['drop_20d'] = (
                    index_df['close'] - index_df['close'].shift(drop_lookback)
                ) / index_df['close'].shift(drop_lookback)
                
                # 对齐到策略日期范围（使用 ffill 填充非交易日）
                aligned_index = index_df.reindex(all_dates, method='ffill')
                
                # ===== 激进版风控条件 =====
                # (Close < MA60) AND (20日跌幅 > threshold%)
                # 只有确认暴跌趋势时才触发熔断 - 从配置读取阈值
                drop_threshold = -self._market_risk_drop_threshold
                
                condition_below_ma60 = aligned_index['close'] < aligned_index['ma60']
                condition_crash = aligned_index['drop_20d'] < drop_threshold
                
                market_risk_series = (condition_below_ma60 & condition_crash).fillna(False)
                risk_triggered_days = market_risk_series.sum()
                
                logger.info(
                    f"大盘风控已启用（激进版）: (Close < MA{ma_period}) AND "
                    f"({drop_lookback}日跌幅 < {drop_threshold:.0%}), "
                    f"预计触发 {risk_triggered_days} 天"
                )
                
            except Exception as e:
                logger.warning(f"计算大盘风控指标失败: {e}，风控功能暂时失效")
        else:
            logger.debug("未传入基准数据，大盘风控未启用")
        
        # ========================================
        # 初始化权重矩阵
        # ========================================
        target_weights = pd.DataFrame(
            0.0,
            index=all_dates,
            columns=sorted(all_stocks),
            dtype=float
        )
        target_weights.index.name = 'date'
        target_weights.columns.name = 'symbol'
        
        # 当前权重（用于再平衡缓冲区）
        current_weights: Dict[str, float] = current_holdings_weights.copy() if current_holdings_weights else {}
        
        # 统计
        skipped_adjustments = 0
        forced_executions = 0
        risk_clear_count = 0
        
        for date in all_dates:
            # ========================================
            # Step 1: 每日风控检查
            # ========================================
            is_risk_triggered = market_risk_series.loc[date]
            
            if is_risk_triggered:
                # 风控触发：强制清仓
                if current_weights:
                    # 仅在有持仓时记录日志，避免日志爆炸
                    logger.warning(
                        f"Market Risk Triggered on {date.strftime('%Y-%m-%d')}, "
                        f"clearing positions (持有 {len(current_weights)} 只股票)"
                    )
                    risk_clear_count += 1
                
                # 清空当前权重
                current_weights = {}
                
                # 跳过后续选股逻辑，直接进入下一天
                # target_weights 保持为 0（已初始化）
                continue
            
            # ========================================
            # Step 2: 调仓日逻辑（仅在无风控时执行）
            # 采用 Filter-Then-Analyze 模式
            # ========================================
            if date in rebalance_dates:
                # 调仓日: 重新选股并优化权重
                filtered_data = self.filter_stocks(factor_data, date)
                
                if not filtered_data.empty:
                    # [NEW] 添加持股惯性标记 - 当前持仓股票获得选股加分
                    current_holding_set = set(current_weights.keys()) if current_weights else set()
                    filtered_data = filtered_data.copy()
                    if stock_col in filtered_data.columns:
                        filtered_data['is_holding'] = filtered_data[stock_col].isin(current_holding_set)
                    else:
                        filtered_data['is_holding'] = False
                    
                    # Step 2.1: 获取扩展候选列表（Top N * buffer）
                    buffer_n = self.top_n * self._sentiment_buffer_multiplier
                    pre_candidates = self.select_top_stocks(filtered_data, n=buffer_n)
                    
                    # Step 2.2: 应用情绪过滤（Just-in-Time 分析）
                    # 注意: _apply_sentiment_filter 内部已实现 Fail-Closed 策略
                    if self._enable_sentiment_filter and self._sentiment_engine is not None:
                        try:
                            final_candidates = self._apply_sentiment_filter(pre_candidates, date)
                        except LLMCircuitBreakerError:
                            # ===== 熔断器触发: 停止交易 =====
                            # 直接抛出，由调用方处理
                            raise
                        # 注意: 其他异常已在 _apply_sentiment_filter 中处理，
                        # 会返回空列表（Fail-Closed），不会抛出
                    else:
                        final_candidates = pre_candidates
                    
                    # Step 2.3: 最终选取 Top N
                    selected_stocks = final_candidates[:self.top_n]
                    
                    logger.debug(
                        f"调仓日 {date.strftime('%Y-%m-%d')}: "
                        f"预选 {len(pre_candidates)} 只 -> "
                        f"情绪过滤后 {len(final_candidates)} 只 -> "
                        f"最终选中 {len(selected_stocks)} 只股票"
                    )
                    
                    if selected_stocks:
                        # =====================================================
                        # 懒惰再平衡逻辑（Lazy Rebalance for Small Capital）
                        # =====================================================
                        # 
                        # 背景：30万资金实盘，最低5元佣金导致摩擦成本极高
                        # 目的：只做必要的换股（Swap），避免对持仓进行微小的权重调整（Rebalance）
                        # 
                        # 核心规则：
                        # 1. 继续持有：股票仍在选中列表中 → 保持当前权重不变，不产生交易
                        # 2. 卖出：股票不再在选中列表中 → 权重设为0，释放仓位
                        # 3. 买入：新进入选中列表 → 用卖出释放的仓位进行等权分配
                        # 
                        # 效果：
                        # - 如果5只股票全部继续入选，则本期零交易
                        # - 如果换1只股票，则只产生1买1卖两笔交易
                        # - 大幅降低交易频率和佣金成本
                        # =====================================================
                        
                        current_holding_set = set(current_weights.keys()) if current_weights else set()
                        selected_set = set(selected_stocks)
                        
                        # 分类股票：继续持有 / 卖出 / 买入
                        continuing_stocks = current_holding_set & selected_set  # 交集：继续持有
                        stocks_to_sell = current_holding_set - selected_set     # 差集：需要卖出
                        stocks_to_buy = selected_set - current_holding_set      # 差集：需要买入
                        
                        if not current_holding_set:
                            # ===== 首次建仓：全部等权分配 =====
                            final_weights = self._equal_weights(selected_stocks)
                            forced_executions += len(selected_stocks)
                            logger.info(
                                f"🚀 首次建仓 {date.strftime('%Y-%m-%d')}: "
                                f"等权分配 {len(selected_stocks)} 只股票"
                            )
                        else:
                            # ===== 懒惰再平衡：只换股，不调仓 =====
                            final_weights: Dict[str, float] = {}
                            
                            # Step 1: 继续持有的股票 - 沿用当前权重，不产生任何交易
                            # 
                            # ⚠️ 关键假设：current_weights 必须是 Drifted Weights（漂移后权重）
                            # 
                            # Drifted Weights = (当前股价 × 持仓股数) / 当前总资产
                            # 
                            # 为什么需要 Drifted Weights？
                            # ────────────────────────────
                            # 动量策略核心：Let profits run（让盈利奔跑）
                            # 
                            # 场景：A股大涨50%
                            # - 上期目标权重：20%
                            # - 漂移后权重：26.7%（因涨幅自然增大）
                            # 
                            # 如果传入上期目标权重（20%）：
                            #   → 系统会卖出6.7%仓位，把盈利部分变现
                            #   → 违背"让盈利奔跑"的动量原则 ❌
                            # 
                            # 如果传入漂移后权重（26.7%）：
                            #   → 保持26.7%权重不变，不产生交易
                            #   → 符合动量策略，强者恒强 ✅
                            # 
                            # 调用方责任：确保传入的是基于最新价格计算的真实权重！
                            # 
                            for stock in continuing_stocks:
                                final_weights[stock] = current_weights[stock]
                                skipped_adjustments += 1  # 记录跳过的调整次数
                            
                            # Step 2: 计算卖出释放的仓位
                            released_weight = sum(current_weights.get(s, 0.0) for s in stocks_to_sell)
                            
                            # 记录卖出
                            if stocks_to_sell:
                                forced_executions += len(stocks_to_sell)
                                logger.debug(
                                    f"📤 卖出 {len(stocks_to_sell)} 只: {list(stocks_to_sell)}, "
                                    f"释放权重: {released_weight:.2%}"
                                )
                            
                            # Step 3: 新买入的股票 - 分配释放的仓位
                            if stocks_to_buy:
                                forced_executions += len(stocks_to_buy)
                                
                                if released_weight > 0:
                                    # 有释放的仓位：等权分配给新股票
                                    weight_per_new = released_weight / len(stocks_to_buy)
                                    for stock in stocks_to_buy:
                                        final_weights[stock] = weight_per_new
                                    
                                    logger.debug(
                                        f"📥 买入 {len(stocks_to_buy)} 只: {list(stocks_to_buy)}, "
                                        f"每只权重: {weight_per_new:.2%}"
                                    )
                                else:
                                    # 极端情况：没有释放的仓位但有新股票要买入
                                    # 这种情况说明 top_n 发生了变化，需要重新分配
                                    # 策略：从现有持仓中按比例腾出空间（重新等权分配）
                                    n_total = len(continuing_stocks) + len(stocks_to_buy)
                                    target_weight = 1.0 / n_total
                                    
                                    # 重新分配所有股票权重
                                    final_weights = {}
                                    for stock in continuing_stocks:
                                        final_weights[stock] = target_weight
                                    for stock in stocks_to_buy:
                                        final_weights[stock] = target_weight
                                    
                                    logger.warning(
                                        f"⚠️ 无释放仓位但需买入新股票，重新等权分配 {n_total} 只"
                                    )
                            
                            # Step 4: 归一化权重（确保总和为1）
                            weight_sum = sum(final_weights.values())
                            if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
                                final_weights = {k: v / weight_sum for k, v in final_weights.items()}
                            
                            # 懒惰再平衡日志
                            if stocks_to_sell or stocks_to_buy:
                                logger.info(
                                    f"🔄 懒惰再平衡 {date.strftime('%Y-%m-%d')}: "
                                    f"继续持有 {len(continuing_stocks)} 只 (权重不变), "
                                    f"卖出 {len(stocks_to_sell)} 只, "
                                    f"买入 {len(stocks_to_buy)} 只"
                                )
                            else:
                                logger.debug(
                                    f"✅ 无换股 {date.strftime('%Y-%m-%d')}: "
                                    f"全部 {len(continuing_stocks)} 只股票继续持有，本期零交易"
                                )
                        
                        current_weights = final_weights
                else:
                    logger.warning(f"调仓日 {date.strftime('%Y-%m-%d')}: 无可选股票")
                    current_weights = {}
            
            # ========================================
            # Step 3: 设置当日权重
            # ========================================
            for stock, weight in current_weights.items():
                if stock in target_weights.columns:
                    target_weights.loc[date, stock] = weight
        
        # ========================================
        # 统计信息
        # ========================================
        avg_weight_sum = target_weights.sum(axis=1).mean()
        n_holdings = (target_weights > 0).sum(axis=1).mean()
        
        log_msg = (
            f"目标权重矩阵生成完成: "
            f"平均持仓 {n_holdings:.1f} 只, "
            f"平均权重和 {avg_weight_sum:.4f}, "
            f"跳过微调 {skipped_adjustments} 次, "
            f"强制执行 {forced_executions} 次 (缓冲区: {buffer_threshold:.1%})"
        )
        
        if risk_clear_count > 0:
            log_msg += f", 风控清仓 {risk_clear_count} 次"
        
        logger.info(log_msg)
        
        return target_weights