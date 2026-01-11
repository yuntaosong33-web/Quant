"""
回测结果数据类

提供回测结果的数据结构定义。
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    """
    回测结果数据类
    
    Attributes
    ----------
    total_return : float
        总收益率
    annual_return : float
        年化收益率
    sharpe_ratio : float
        夏普比率
    max_drawdown : float
        最大回撤
    win_rate : float
        胜率
    profit_factor : float
        盈亏比
    total_trades : int
        总交易次数
    portfolio_values : pd.Series
        组合净值曲线
    trade_records : pd.DataFrame
        交易记录
    """
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    portfolio_values: pd.Series
    trade_records: pd.DataFrame


__all__ = ['BacktestResult']

