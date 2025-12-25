"""
因子计算模块单元测试
"""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '..')

from src.features import TechnicalFeatures, AlphaFeatures


class TestTechnicalFeatures:
    """技术指标测试类"""
    
    @pytest.fixture
    def sample_ohlcv(self) -> pd.DataFrame:
        """创建OHLCV测试数据"""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.randn(100))
        
        return pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000000, 5000000, 100),
            "amount": np.random.randint(10000000, 50000000, 100)
        }, index=dates)
    
    def test_feature_engine_initialization(self):
        """测试因子引擎初始化"""
        engine = TechnicalFeatures()
        
        feature_names = engine.get_feature_names()
        assert len(feature_names) > 0
        assert "sma_5" in feature_names
        assert "rsi_14" in feature_names
    
    def test_calculate_all_features(self, sample_ohlcv):
        """测试计算所有因子"""
        engine = TechnicalFeatures()
        result = engine.calculate(sample_ohlcv)
        
        # 验证所有因子列都存在
        for feature_name in engine.get_feature_names():
            assert feature_name in result.columns
        
        # 验证行数不变
        assert len(result) == len(sample_ohlcv)
    
    def test_sma_calculation(self, sample_ohlcv):
        """测试SMA计算"""
        sma = TechnicalFeatures.sma(sample_ohlcv["close"], 5)
        
        assert len(sma) == len(sample_ohlcv)
        assert not sma.isna().all()
        
        # 验证第5个值是前5个收盘价的平均
        expected = sample_ohlcv["close"].iloc[:5].mean()
        assert np.isclose(sma.iloc[4], expected, rtol=1e-5)
    
    def test_ema_calculation(self, sample_ohlcv):
        """测试EMA计算"""
        ema = TechnicalFeatures.ema(sample_ohlcv["close"], 12)
        
        assert len(ema) == len(sample_ohlcv)
        assert not ema.isna().all()
    
    def test_rsi_calculation(self, sample_ohlcv):
        """测试RSI计算"""
        rsi = TechnicalFeatures.rsi(sample_ohlcv["close"], 14)
        
        assert len(rsi) == len(sample_ohlcv)
        
        # RSI应该在0-100之间
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    def test_macd_calculation(self, sample_ohlcv):
        """测试MACD计算"""
        macd_line, signal_line, histogram = TechnicalFeatures.macd(
            sample_ohlcv["close"]
        )
        
        assert len(macd_line) == len(sample_ohlcv)
        assert len(signal_line) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)
        
        # 验证histogram = macd - signal
        diff = macd_line - signal_line
        assert np.allclose(histogram.dropna(), diff.dropna(), rtol=1e-5)
    
    def test_bollinger_bands(self, sample_ohlcv):
        """测试布林带计算"""
        upper, middle, lower = TechnicalFeatures.bollinger_bands(
            sample_ohlcv["close"], 20, 2.0
        )
        
        assert len(upper) == len(sample_ohlcv)
        
        # 验证 upper > middle > lower
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
    
    def test_atr_calculation(self, sample_ohlcv):
        """测试ATR计算"""
        atr = TechnicalFeatures.atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            14
        )
        
        assert len(atr) == len(sample_ohlcv)
        
        # ATR应该为正
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_kdj_calculation(self, sample_ohlcv):
        """测试KDJ计算"""
        k, d, j = TechnicalFeatures.kdj(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"]
        )
        
        assert len(k) == len(sample_ohlcv)
        assert len(d) == len(sample_ohlcv)
        assert len(j) == len(sample_ohlcv)
    
    def test_add_custom_feature(self, sample_ohlcv):
        """测试添加自定义因子"""
        engine = TechnicalFeatures()
        
        # 添加自定义因子
        engine.add_feature(
            "custom_factor",
            lambda df: df["close"] / df["open"] - 1
        )
        
        assert "custom_factor" in engine.get_feature_names()
        
        result = engine.calculate(sample_ohlcv)
        assert "custom_factor" in result.columns


class TestAlphaFeatures:
    """Alpha因子测试类"""
    
    @pytest.fixture
    def sample_ohlcv(self) -> pd.DataFrame:
        """创建OHLCV测试数据"""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.randn(50))
        volume = np.random.randint(1000000, 5000000, 50).astype(float)
        
        return pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": volume,
            "amount": volume * close
        }, index=dates)
    
    def test_alpha_features_calculation(self, sample_ohlcv):
        """测试Alpha因子计算"""
        engine = AlphaFeatures()
        result = engine.calculate(sample_ohlcv)
        
        # 验证Alpha因子列存在
        for feature_name in engine.get_feature_names():
            assert feature_name in result.columns

