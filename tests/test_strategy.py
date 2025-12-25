"""
策略模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '..')

from src.strategy import (
    BaseStrategy,
    MACrossStrategy,
    RSIStrategy,
    CompositeStrategy,
    SignalType,
    TradeSignal
)


class TestMACrossStrategy:
    """双均线策略测试类"""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """创建测试数据"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        # 生成带趋势的价格数据
        trend = np.linspace(100, 120, 100)
        noise = np.random.randn(100) * 2
        close = trend + noise
        
        return pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_strategy_initialization(self):
        """测试策略初始化"""
        config = {"short_window": 5, "long_window": 20}
        strategy = MACrossStrategy(name="Test MA", config=config)
        
        assert strategy.name == "Test MA"
        assert strategy.short_window == 5
        assert strategy.long_window == 20
    
    def test_generate_signals(self, sample_data):
        """测试信号生成"""
        strategy = MACrossStrategy(config={"short_window": 5, "long_window": 10})
        signals = strategy.generate_signals(sample_data)
        
        assert len(signals) == len(sample_data)
        assert signals.dtype in [np.int64, np.int32, int]
        assert set(signals.unique()).issubset({-1, 0, 1})
    
    def test_position_size_calculation(self):
        """测试仓位计算"""
        strategy = MACrossStrategy(config={"position_size": 0.1})
        
        signal = TradeSignal(
            timestamp=pd.Timestamp.now(),
            symbol="000001",
            signal_type=SignalType.BUY,
            price=10.0,
            strength=1.0
        )
        
        position = strategy.calculate_position_size(signal, 1000000)
        assert position == 100000  # 10% of 1000000
    
    def test_strategy_reset(self, sample_data):
        """测试策略重置"""
        strategy = MACrossStrategy()
        
        # 生成一些信号
        strategy.on_data(sample_data)
        
        # 重置
        strategy.reset()
        
        assert len(strategy._signals) == 0
        assert len(strategy._positions) == 0


class TestRSIStrategy:
    """RSI策略测试类"""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """创建测试数据"""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.randn(50))
        
        return pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000000, 5000000, 50)
        }, index=dates)
    
    def test_rsi_strategy_initialization(self):
        """测试RSI策略初始化"""
        config = {"rsi_period": 14, "oversold": 25, "overbought": 75}
        strategy = RSIStrategy(name="Test RSI", config=config)
        
        assert strategy.rsi_period == 14
        assert strategy.oversold == 25
        assert strategy.overbought == 75
    
    def test_rsi_signals(self, sample_data):
        """测试RSI信号生成"""
        strategy = RSIStrategy(config={"rsi_period": 7})
        signals = strategy.generate_signals(sample_data)
        
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestCompositeStrategy:
    """组合策略测试类"""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """创建测试数据"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.randn(100))
        
        return pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_composite_strategy(self, sample_data):
        """测试组合策略"""
        composite = CompositeStrategy(name="Composite Test")
        
        ma_strategy = MACrossStrategy(config={"short_window": 5, "long_window": 20})
        rsi_strategy = RSIStrategy(config={"rsi_period": 14})
        
        composite.add_strategy(ma_strategy, weight=0.6)
        composite.add_strategy(rsi_strategy, weight=0.4)
        
        signals = composite.generate_signals(sample_data)
        
        assert len(signals) == len(sample_data)
        assert len(composite.strategies) == 2


class TestTradeSignal:
    """交易信号测试类"""
    
    def test_trade_signal_creation(self):
        """测试交易信号创建"""
        signal = TradeSignal(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="000001",
            signal_type=SignalType.BUY,
            price=10.5,
            strength=0.8,
            reason="Golden cross"
        )
        
        assert signal.symbol == "000001"
        assert signal.signal_type == SignalType.BUY
        assert signal.price == 10.5
        assert signal.strength == 0.8
        assert signal.reason == "Golden cross"

