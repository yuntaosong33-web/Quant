"""
因子计算性能测试模块

测试 numba 优化后的因子计算函数性能，对比优化前后的计算时间。
"""

import time
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from src.features import (
    TechnicalFeatures,
    FactorCalculator,
    _rolling_std_1d,
    _calculate_rsi_numba,
    _grouped_rolling_std,
    _grouped_rsi,
    _calculate_atr_numba,
    _calculate_volatility_numba,
    _calculate_kdj_numba,
)


def generate_mock_ohlcv(n_rows: int, n_stocks: int = 1) -> pd.DataFrame:
    """
    生成模拟 OHLCV 数据
    
    Parameters
    ----------
    n_rows : int
        每只股票的数据行数
    n_stocks : int
        股票数量
    
    Returns
    -------
    pd.DataFrame
        模拟 OHLCV 数据
    """
    np.random.seed(42)
    
    data_list = []
    for stock_idx in range(n_stocks):
        stock_code = f"{600000 + stock_idx:06d}"
        
        # 生成随机价格序列（随机游走）
        close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        close = np.maximum(close, 1)  # 确保价格为正
        
        high = close * (1 + np.abs(np.random.randn(n_rows) * 0.01))
        low = close * (1 - np.abs(np.random.randn(n_rows) * 0.01))
        open_price = (high + low) / 2 + np.random.randn(n_rows) * 0.1
        volume = np.random.randint(1000000, 10000000, n_rows)
        amount = volume * close
        
        dates = pd.date_range('2020-01-01', periods=n_rows, freq='D')
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'amount': amount,
            'stock_code': stock_code,
        }, index=dates)
        
        data_list.append(df)
    
    result = pd.concat(data_list, ignore_index=False)
    result.index = pd.to_datetime(result.index)
    
    return result


def generate_mock_financial(n_rows: int, n_stocks: int = 1) -> pd.DataFrame:
    """
    生成模拟财务数据
    
    Parameters
    ----------
    n_rows : int
        每只股票的数据行数
    n_stocks : int
        股票数量
    
    Returns
    -------
    pd.DataFrame
        模拟财务数据
    """
    np.random.seed(42)
    
    data_list = []
    for stock_idx in range(n_stocks):
        stock_code = f"{600000 + stock_idx:06d}"
        
        pe_ttm = np.random.uniform(5, 50, n_rows)
        roe = np.random.uniform(0.05, 0.3, n_rows)
        dividend_yield = np.random.uniform(0, 0.05, n_rows)
        
        dates = pd.date_range('2020-01-01', periods=n_rows, freq='D')
        
        df = pd.DataFrame({
            'pe_ttm': pe_ttm,
            'roe': roe,
            'dividend_yield': dividend_yield,
            'stock_code': stock_code,
        }, index=dates)
        
        data_list.append(df)
    
    result = pd.concat(data_list, ignore_index=False)
    result.index = pd.to_datetime(result.index)
    
    return result


def benchmark(func: Callable, name: str, *args, **kwargs) -> Tuple[float, any]:
    """
    对函数进行性能测试
    
    Parameters
    ----------
    func : Callable
        要测试的函数
    name : str
        测试名称
    *args, **kwargs
        函数参数
    
    Returns
    -------
    Tuple[float, any]
        (执行时间, 函数返回值)
    """
    # 预热（对于 numba 编译）
    try:
        _ = func(*args, **kwargs)
    except Exception:
        pass
    
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    
    print(f"{name}: {elapsed:.4f} 秒")
    return elapsed, result


def test_rsi_numba_performance() -> None:
    """测试 RSI numba 加速性能"""
    print("\n" + "=" * 60)
    print("测试 RSI Numba 加速性能")
    print("=" * 60)
    
    # 测试数据规模
    n_rows = 100000
    close = np.random.randn(n_rows).cumsum() + 100
    close = np.maximum(close, 1).astype(np.float64)
    
    print(f"数据规模: {n_rows:,} 行")
    
    # Numba 版本
    time_numba, _ = benchmark(_calculate_rsi_numba, "Numba RSI", close, 14)
    
    # Pandas 版本（原始实现）
    def pandas_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    close_series = pd.Series(close)
    time_pandas, _ = benchmark(pandas_rsi, "Pandas RSI", close_series, 14)
    
    speedup = time_pandas / time_numba if time_numba > 0 else float('inf')
    print(f"加速比: {speedup:.2f}x")


def test_rolling_std_numba_performance() -> None:
    """测试滚动标准差 numba 加速性能"""
    print("\n" + "=" * 60)
    print("测试滚动标准差 Numba 加速性能")
    print("=" * 60)
    
    n_rows = 100000
    data = np.random.randn(n_rows).astype(np.float64)
    
    print(f"数据规模: {n_rows:,} 行")
    
    # Numba 版本
    time_numba, _ = benchmark(_rolling_std_1d, "Numba Rolling Std", data, 20, 10)
    
    # Pandas 版本
    def pandas_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(arr).rolling(window=window, min_periods=10).std().values
    
    time_pandas, _ = benchmark(pandas_rolling_std, "Pandas Rolling Std", data, 20)
    
    speedup = time_pandas / time_numba if time_numba > 0 else float('inf')
    print(f"加速比: {speedup:.2f}x")


def test_grouped_ivol_performance() -> None:
    """测试分组 IVOL 计算性能"""
    print("\n" + "=" * 60)
    print("测试分组 IVOL 计算性能 (全市场规模)")
    print("=" * 60)
    
    # 模拟全市场 A 股规模：约 5000 只股票，每只 250 天数据
    n_stocks = 500  # 减少测试规模
    n_days = 250
    
    ohlcv = generate_mock_ohlcv(n_days, n_stocks)
    financial = generate_mock_financial(n_days, n_stocks)
    
    print(f"数据规模: {n_stocks} 只股票 x {n_days} 天 = {len(ohlcv):,} 行")
    
    # 测试 FactorCalculator 的 IVOL 计算
    calculator = FactorCalculator(ohlcv, financial)
    
    time_ivol, _ = benchmark(calculator.calculate_ivol, "Numba IVOL (分组)", period=20)
    print(f"每只股票平均耗时: {time_ivol / n_stocks * 1000:.2f} 毫秒")


def test_grouped_rsi_performance() -> None:
    """测试分组 RSI 计算性能"""
    print("\n" + "=" * 60)
    print("测试分组 RSI 计算性能 (全市场规模)")
    print("=" * 60)
    
    n_stocks = 500
    n_days = 250
    
    ohlcv = generate_mock_ohlcv(n_days, n_stocks)
    financial = generate_mock_financial(n_days, n_stocks)
    
    print(f"数据规模: {n_stocks} 只股票 x {n_days} 天 = {len(ohlcv):,} 行")
    
    calculator = FactorCalculator(ohlcv, financial)
    
    time_rsi, _ = benchmark(calculator.calculate_rsi, "Numba RSI (分组)", period=20)
    print(f"每只股票平均耗时: {time_rsi / n_stocks * 1000:.2f} 毫秒")


def test_technical_features_performance() -> None:
    """测试 TechnicalFeatures 全量计算性能"""
    print("\n" + "=" * 60)
    print("测试 TechnicalFeatures 全量计算性能")
    print("=" * 60)
    
    n_rows = 50000
    ohlcv = generate_mock_ohlcv(n_rows, 1)
    # 移除 stock_code 列以测试单股票场景
    ohlcv = ohlcv.drop(columns=['stock_code'])
    
    print(f"数据规模: {n_rows:,} 行")
    
    engine = TechnicalFeatures()
    
    time_total, _ = benchmark(engine.calculate, "TechnicalFeatures.calculate()", ohlcv)
    
    n_features = len(engine.get_feature_names())
    print(f"计算因子数量: {n_features}")
    print(f"每个因子平均耗时: {time_total / n_features * 1000:.2f} 毫秒")


def test_atr_kdj_volatility_performance() -> None:
    """测试 ATR、KDJ、Volatility 的 numba 加速性能"""
    print("\n" + "=" * 60)
    print("测试 ATR/KDJ/Volatility Numba 加速性能")
    print("=" * 60)
    
    n_rows = 100000
    np.random.seed(42)
    
    close = (100 + np.cumsum(np.random.randn(n_rows) * 0.5)).astype(np.float64)
    close = np.maximum(close, 1)
    high = (close * (1 + np.abs(np.random.randn(n_rows) * 0.01))).astype(np.float64)
    low = (close * (1 - np.abs(np.random.randn(n_rows) * 0.01))).astype(np.float64)
    
    print(f"数据规模: {n_rows:,} 行")
    
    # ATR
    time_atr, _ = benchmark(_calculate_atr_numba, "Numba ATR", high, low, close, 14)
    
    # Volatility
    time_vol, _ = benchmark(_calculate_volatility_numba, "Numba Volatility", close, 20)
    
    # KDJ
    time_kdj, _ = benchmark(_calculate_kdj_numba, "Numba KDJ", high, low, close, 9)


def run_all_benchmarks() -> None:
    """运行所有性能测试"""
    print("\n" + "#" * 70)
    print("# 因子计算 Numba 优化性能测试")
    print("#" * 70)
    
    test_rsi_numba_performance()
    test_rolling_std_numba_performance()
    test_atr_kdj_volatility_performance()
    test_grouped_ivol_performance()
    test_grouped_rsi_performance()
    test_technical_features_performance()
    
    print("\n" + "=" * 60)
    print("所有性能测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()

