"""
Numba 优化的底层计算函数

该模块提供使用 Numba JIT 编译加速的滚动窗口计算函数。
这些函数被技术指标和因子计算模块调用。

Performance Notes
-----------------
- 所有函数使用 @jit(nopython=True, cache=True) 装饰器
- 并行函数使用 @jit(nopython=True, parallel=True, cache=True)
- 缓存编译结果以加速后续调用
- 如果 numba 未安装，将使用纯 Python 实现（较慢）
"""

import numpy as np

# Numba 可选依赖
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 创建无操作装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, cache=True)
def rolling_std_1d(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Numba 加速的一维滚动标准差计算
    
    Parameters
    ----------
    arr : np.ndarray
        输入数组
    window : int
        滚动窗口大小
    min_periods : int
        最小有效数据点数
    
    Returns
    -------
    np.ndarray
        滚动标准差结果
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        
        valid_count = 0
        sum_val = 0.0
        sum_sq = 0.0
        
        for j in range(start, end):
            if not np.isnan(arr[j]):
                valid_count += 1
                sum_val += arr[j]
                sum_sq += arr[j] * arr[j]
        
        if valid_count >= min_periods:
            mean_val = sum_val / valid_count
            variance = (sum_sq / valid_count) - (mean_val * mean_val)
            if valid_count > 1:
                variance = variance * valid_count / (valid_count - 1)
            if variance > 0:
                result[i] = np.sqrt(variance)
            else:
                result[i] = 0.0
    
    return result


@jit(nopython=True, cache=True)
def rolling_mean_1d(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Numba 加速的一维滚动均值计算
    
    Parameters
    ----------
    arr : np.ndarray
        输入数组
    window : int
        滚动窗口大小
    min_periods : int
        最小有效数据点数
    
    Returns
    -------
    np.ndarray
        滚动均值结果
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        
        valid_count = 0
        sum_val = 0.0
        
        for j in range(start, end):
            if not np.isnan(arr[j]):
                valid_count += 1
                sum_val += arr[j]
        
        if valid_count >= min_periods:
            result[i] = sum_val / valid_count
    
    return result


@jit(nopython=True, cache=True)
def calculate_rsi_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    Numba 加速的 RSI 计算
    
    Parameters
    ----------
    close : np.ndarray
        收盘价数组
    period : int
        计算周期
    
    Returns
    -------
    np.ndarray
        RSI 值数组 (0-100)
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n < period + 1:
        return result
    
    delta = np.empty(n, dtype=np.float64)
    delta[0] = np.nan
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if delta[i] > 0:
            gain[i] = delta[i]
        elif delta[i] < 0:
            loss[i] = -delta[i]
    
    alpha = 1.0 / period
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    sum_gain = 0.0
    sum_loss = 0.0
    for i in range(1, period + 1):
        sum_gain += gain[i]
        sum_loss += loss[i]
    avg_gain[period] = sum_gain / period
    avg_loss[period] = sum_loss / period
    
    for i in range(period + 1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            if avg_gain[i] == 0:
                result[i] = 50.0
            else:
                result[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


@jit(nopython=True, cache=True)
def calculate_returns_1d(close: np.ndarray) -> np.ndarray:
    """
    Numba 加速的收益率计算
    
    Parameters
    ----------
    close : np.ndarray
        收盘价数组
    
    Returns
    -------
    np.ndarray
        日收益率数组
    """
    n = len(close)
    returns = np.empty(n, dtype=np.float64)
    returns[0] = np.nan
    
    for i in range(1, n):
        if close[i - 1] != 0 and not np.isnan(close[i - 1]):
            returns[i] = (close[i] - close[i - 1]) / close[i - 1]
        else:
            returns[i] = np.nan
    
    return returns


@jit(nopython=True, parallel=True, cache=True)
def grouped_rolling_std(
    values: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
    window: int,
    min_periods: int
) -> np.ndarray:
    """
    Numba 加速的分组滚动标准差计算（并行版本）
    
    Parameters
    ----------
    values : np.ndarray
        值数组
    group_ids : np.ndarray
        分组 ID 数组（从0开始的整数）
    n_groups : int
        分组数量
    window : int
        滚动窗口大小
    min_periods : int
        最小有效数据点数
    
    Returns
    -------
    np.ndarray
        分组滚动标准差结果
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        gid = group_ids[i]
        if gid >= 0:
            group_sizes[gid] += 1
    
    group_starts = np.zeros(n_groups + 1, dtype=np.int64)
    for i in range(n_groups):
        group_starts[i + 1] = group_starts[i] + group_sizes[i]
    
    group_indices = np.empty(n, dtype=np.int64)
    group_positions = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        gid = group_ids[i]
        if gid >= 0:
            pos = group_starts[gid] + group_positions[gid]
            group_indices[pos] = i
            group_positions[gid] += 1
    
    for g in prange(n_groups):
        start_pos = group_starts[g]
        end_pos = group_starts[g + 1]
        group_size = end_pos - start_pos
        
        if group_size == 0:
            continue
        
        for k in range(group_size):
            original_idx = group_indices[start_pos + k]
            win_start = max(0, k - window + 1)
            win_end = k + 1
            
            valid_count = 0
            sum_val = 0.0
            sum_sq = 0.0
            
            for w in range(win_start, win_end):
                val_idx = group_indices[start_pos + w]
                val = values[val_idx]
                if not np.isnan(val):
                    valid_count += 1
                    sum_val += val
                    sum_sq += val * val
            
            if valid_count >= min_periods:
                mean_val = sum_val / valid_count
                variance = (sum_sq / valid_count) - (mean_val * mean_val)
                if valid_count > 1:
                    variance = variance * valid_count / (valid_count - 1)
                if variance > 0:
                    result[original_idx] = np.sqrt(variance)
                else:
                    result[original_idx] = 0.0
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def grouped_rsi(
    close: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
    period: int
) -> np.ndarray:
    """
    Numba 加速的分组 RSI 计算（并行版本）
    
    Parameters
    ----------
    close : np.ndarray
        收盘价数组
    group_ids : np.ndarray
        分组 ID 数组
    n_groups : int
        分组数量
    period : int
        RSI 周期
    
    Returns
    -------
    np.ndarray
        RSI 结果数组
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        gid = group_ids[i]
        if gid >= 0:
            group_sizes[gid] += 1
    
    group_starts = np.zeros(n_groups + 1, dtype=np.int64)
    for i in range(n_groups):
        group_starts[i + 1] = group_starts[i] + group_sizes[i]
    
    group_indices = np.empty(n, dtype=np.int64)
    group_positions = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        gid = group_ids[i]
        if gid >= 0:
            pos = group_starts[gid] + group_positions[gid]
            group_indices[pos] = i
            group_positions[gid] += 1
    
    for g in prange(n_groups):
        start_pos = group_starts[g]
        end_pos = group_starts[g + 1]
        group_size = end_pos - start_pos
        
        if group_size < period + 1:
            continue
        
        group_close = np.empty(group_size, dtype=np.float64)
        for k in range(group_size):
            group_close[k] = close[group_indices[start_pos + k]]
        
        group_rsi = calculate_rsi_numba(group_close, period)
        
        for k in range(group_size):
            original_idx = group_indices[start_pos + k]
            result[original_idx] = group_rsi[k]
    
    return result


@jit(nopython=True, cache=True)
def calculate_atr_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Numba 加速的 ATR 计算
    
    Parameters
    ----------
    high : np.ndarray
        最高价数组
    low : np.ndarray
        最低价数组
    close : np.ndarray
        收盘价数组
    period : int
        计算周期
    
    Returns
    -------
    np.ndarray
        ATR 值数组
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n < 2:
        return result
    
    true_range = np.empty(n, dtype=np.float64)
    true_range[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        true_range[i] = max(tr1, tr2, tr3)
    
    result = rolling_mean_1d(true_range, period, 1)
    
    return result


@jit(nopython=True, cache=True)
def calculate_volatility_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    Numba 加速的年化波动率计算
    
    Parameters
    ----------
    close : np.ndarray
        收盘价数组
    period : int
        计算周期
    
    Returns
    -------
    np.ndarray
        年化波动率数组
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n < 2:
        return result
    
    log_returns = np.empty(n, dtype=np.float64)
    log_returns[0] = np.nan
    
    for i in range(1, n):
        if close[i - 1] > 0 and close[i] > 0:
            log_returns[i] = np.log(close[i] / close[i - 1])
        else:
            log_returns[i] = np.nan
    
    volatility = rolling_std_1d(log_returns, period, 1)
    
    annualization_factor = np.sqrt(252.0)
    for i in range(n):
        if not np.isnan(volatility[i]):
            result[i] = volatility[i] * annualization_factor
    
    return result


@jit(nopython=True, cache=True)
def calculate_kdj_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 9
) -> tuple:
    """
    Numba 加速的 KDJ 指标计算
    
    Parameters
    ----------
    high : np.ndarray
        最高价数组
    low : np.ndarray
        最低价数组
    close : np.ndarray
        收盘价数组
    period : int
        计算周期，默认9
    
    Returns
    -------
    tuple
        (K值数组, D值数组, J值数组)
    """
    n = len(close)
    k_values = np.empty(n, dtype=np.float64)
    d_values = np.empty(n, dtype=np.float64)
    j_values = np.empty(n, dtype=np.float64)
    k_values[:] = 50.0
    d_values[:] = 50.0
    j_values[:] = 50.0
    
    rsv = np.empty(n, dtype=np.float64)
    rsv[:] = 50.0
    
    for i in range(period - 1, n):
        highest = high[i]
        lowest = low[i]
        for j in range(i - period + 1, i):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]
        
        if highest != lowest:
            rsv[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
        else:
            rsv[i] = 50.0
    
    alpha = 1.0 / 3.0
    k_values[0] = rsv[0]
    d_values[0] = k_values[0]
    
    for i in range(1, n):
        k_values[i] = alpha * rsv[i] + (1 - alpha) * k_values[i - 1]
        d_values[i] = alpha * k_values[i] + (1 - alpha) * d_values[i - 1]
        j_values[i] = 3 * k_values[i] - 2 * d_values[i]
    
    return k_values, d_values, j_values


# 导出所有公共函数
__all__ = [
    'rolling_std_1d',
    'rolling_mean_1d',
    'calculate_rsi_numba',
    'calculate_returns_1d',
    'grouped_rolling_std',
    'grouped_rsi',
    'calculate_atr_numba',
    'calculate_volatility_numba',
    'calculate_kdj_numba',
]

