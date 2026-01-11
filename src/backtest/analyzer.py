"""
绩效分析器模块

提供详细的策略绩效分析和可视化功能。
"""

from typing import Dict, Any

import pandas as pd

from .result import BacktestResult


class PerformanceAnalyzer:
    """
    绩效分析器
    
    提供详细的策略绩效分析和可视化功能。
    """
    
    def __init__(self, result: BacktestResult) -> None:
        """
        初始化绩效分析器
        
        Parameters
        ----------
        result : BacktestResult
            回测结果
        """
        self.result = result
    
    def summary(self) -> Dict[str, Any]:
        """
        生成绩效摘要
        
        Returns
        -------
        Dict[str, Any]
            绩效指标字典
        """
        return {
            "总收益率": f"{self.result.total_return:.2%}",
            "年化收益率": f"{self.result.annual_return:.2%}",
            "夏普比率": f"{self.result.sharpe_ratio:.2f}",
            "最大回撤": f"{self.result.max_drawdown:.2%}",
            "胜率": f"{self.result.win_rate:.2%}",
            "盈亏比": f"{self.result.profit_factor:.2f}",
            "总交易次数": self.result.total_trades,
        }
    
    def monthly_returns(self) -> pd.DataFrame:
        """
        计算月度收益
        
        Returns
        -------
        pd.DataFrame
            月度收益表
        """
        portfolio_values = self.result.portfolio_values
        monthly = portfolio_values.resample("M").last()
        monthly_returns = monthly.pct_change()
        
        # 转换为透视表格式
        monthly_returns.index = pd.MultiIndex.from_arrays([
            monthly_returns.index.year,
            monthly_returns.index.month
        ])
        
        return monthly_returns
    
    def drawdown_analysis(self) -> Dict[str, Any]:
        """
        回撤分析
        
        Returns
        -------
        Dict[str, Any]
            回撤分析结果
        """
        portfolio_values = self.result.portfolio_values
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        
        # 找到最大回撤的起止时间
        max_dd_end = drawdown.idxmin()
        max_dd_start = portfolio_values[:max_dd_end].idxmax()
        
        return {
            "最大回撤": f"{abs(drawdown.min()):.2%}",
            "最大回撤开始": max_dd_start,
            "最大回撤结束": max_dd_end,
            "平均回撤": f"{abs(drawdown.mean()):.2%}",
            "回撤序列": drawdown,
        }
    
    def trade_analysis(self) -> Dict[str, Any]:
        """
        交易分析
        
        Returns
        -------
        Dict[str, Any]
            交易统计信息
        """
        trades = self.result.trade_records
        
        if trades.empty:
            return {
                "总交易次数": 0,
                "买入次数": 0,
                "卖出次数": 0,
                "平均佣金": 0,
                "总佣金": 0
            }
        
        buys = trades[trades['action'] == 'BUY']
        sells = trades[trades['action'] == 'SELL']
        
        return {
            "总交易次数": len(trades),
            "买入次数": len(buys),
            "卖出次数": len(sells),
            "平均佣金": trades['commission'].mean() if 'commission' in trades.columns else 0,
            "总佣金": trades['commission'].sum() if 'commission' in trades.columns else 0,
            "涉及股票数": trades['symbol'].nunique() if 'symbol' in trades.columns else 0,
        }

