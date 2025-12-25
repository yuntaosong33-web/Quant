"""
VectorBT回测流程模块

该模块提供基于VectorBT的回测功能，支持策略绩效评估和分析。
采用向量化回测引擎确保高性能。
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import logging

import pandas as pd
import numpy as np

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logging.warning("vectorbt未安装，部分回测功能不可用")

from .strategy import BaseStrategy

logger = logging.getLogger(__name__)


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


class BacktestEngine:
    """
    回测引擎
    
    基于VectorBT实现高性能向量化回测。
    对于小资金账户（< 50万），强制使用 Simple 模式以确保佣金计算准确（最低5元规则）。
    
    Attributes
    ----------
    config : Dict[str, Any]
        回测配置
    
    Examples
    --------
    >>> engine = BacktestEngine(config)
    >>> result = engine.run(strategy, price_data, signals)
    >>> print(f"夏普比率: {result.sharpe_ratio:.2f}")
    """
    
    # 小资金阈值：低于此值强制使用 Simple 模式
    SMALL_CAPITAL_THRESHOLD = 500000  # 50万
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化回测引擎
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            回测配置，包含初始资金、手续费等参数
        """
        self.config = config or {}
        
        # 回测参数
        self._initial_capital = self.config.get("initial_capital", 1000000)
        self._commission = self.config.get("commission", 0.0003)
        self._slippage = self.config.get("slippage", 0.001)
        self._risk_free_rate = self.config.get("risk_free_rate", 0.03)
        
        # 强制使用 Simple 模式的标志
        self._force_simple_mode = self.config.get("force_simple_mode", False)
        
        logger.info(
            f"回测引擎初始化: 初始资金={self._initial_capital}, "
            f"佣金={self._commission}, 滑点={self._slippage}"
        )
    
    def run(
        self,
        strategy: BaseStrategy,
        price_data: pd.DataFrame,
        signals: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        执行回测
        
        对于小资金账户（< 50万），强制使用 Simple 模式以确保佣金计算准确。
        VectorBT 无法精确模拟 A股"最低5元佣金"规则，对小资金影响显著。
        
        Parameters
        ----------
        strategy : BaseStrategy
            交易策略实例
        price_data : pd.DataFrame
            价格数据，必须包含 'close' 列
        signals : Optional[pd.Series]
            预计算的信号序列，如果为None则调用策略生成
        
        Returns
        -------
        BacktestResult
            回测结果
        """
        logger.info(f"开始回测策略: {strategy.name}")
        
        # 生成信号
        if signals is None:
            signals = strategy.generate_signals(price_data)
        
        # === 小资金账户强制使用 Simple 模式 ===
        # 原因：VectorBT 无法精确模拟 A股"最低5元佣金"规则
        # 对于30万资金，每笔交易约3-6万，按万三计算只有9-18元
        # 但实际最低5元的限制会导致费率实际高达万1.5左右
        use_simple_mode = (
            self._force_simple_mode or 
            self._initial_capital < self.SMALL_CAPITAL_THRESHOLD
        )
        
        if use_simple_mode:
            if self._initial_capital < self.SMALL_CAPITAL_THRESHOLD:
                logger.warning(
                    f"小资金模式（{self._initial_capital/10000:.0f}万 < {self.SMALL_CAPITAL_THRESHOLD/10000:.0f}万）："
                    f"强制使用简单回测引擎，以精确计算最低5元佣金"
                )
            result = self._run_simple(price_data, signals)
        elif VBT_AVAILABLE:
            logger.info("使用 VectorBT 回测引擎（大资金模式）")
            result = self._run_vectorbt(price_data, signals)
        else:
            result = self._run_simple(price_data, signals)
        
        logger.info(
            f"回测完成: 总收益={result.total_return:.2%}, "
            f"夏普比率={result.sharpe_ratio:.2f}, "
            f"最大回撤={result.max_drawdown:.2%}"
        )
        
        return result
    
    def _run_vectorbt(
        self,
        price_data: pd.DataFrame,
        signals: pd.Series
    ) -> BacktestResult:
        """
        使用VectorBT执行回测
        
        Parameters
        ----------
        price_data : pd.DataFrame
            价格数据
        signals : pd.Series
            交易信号
        
        Returns
        -------
        BacktestResult
            回测结果
        """
        close = price_data["close"]
        
        # 创建入场和出场信号
        entries = signals == 1
        exits = signals == -1
        
        # 构建投资组合
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self._initial_capital,
            fees=self._commission,
            slippage=self._slippage,
            freq="1D"
        )
        
        # 提取回测指标
        stats = portfolio.stats()
        trades = portfolio.trades.records_readable
        
        return BacktestResult(
            total_return=stats.get("Total Return [%]", 0) / 100,
            annual_return=stats.get("Annualized Return [%]", 0) / 100,
            sharpe_ratio=stats.get("Sharpe Ratio", 0),
            max_drawdown=stats.get("Max Drawdown [%]", 0) / 100,
            win_rate=stats.get("Win Rate [%]", 0) / 100,
            profit_factor=stats.get("Profit Factor", 0),
            total_trades=stats.get("Total Trades", 0),
            portfolio_values=portfolio.value(),
            trade_records=trades if len(trades) > 0 else pd.DataFrame()
        )
    
    def _run_simple(
        self,
        price_data: pd.DataFrame,
        signals: pd.Series
    ) -> BacktestResult:
        """
        简单回测实现（不依赖VectorBT）
        
        Parameters
        ----------
        price_data : pd.DataFrame
            价格数据
        signals : pd.Series
            交易信号
        
        Returns
        -------
        BacktestResult
            回测结果
        """
        close = price_data["close"]
        
        # 计算持仓
        position = signals.replace(0, np.nan).ffill().fillna(0)
        position = position.clip(-1, 1)
        
        # 计算毛收益 (不含成本)
        returns = close.pct_change().fillna(0)
        strategy_returns_gross = position.shift(1) * returns
        
        # 估算资金曲线 (用于计算交易金额)
        # 注意：使用无成本资金曲线近似计算交易金额，以保持向量化性能
        gross_portfolio_value = (1 + strategy_returns_gross).cumprod() * self._initial_capital
        current_capital_series = gross_portfolio_value.shift(1).fillna(self._initial_capital)
        
        # 计算交易动作和金额
        trades_pct = position.diff().abs().fillna(0)  # 仓位变化比例
        trade_amounts = trades_pct * current_capital_series  # 估算交易金额
        
        # ========================================
        # 计算交易成本 (含A股最低佣金限制)
        # ========================================
        # 
        # A股佣金规则：
        # - 费率：通常万分之三 (0.0003)
        # - 最低限制：单笔最低5元
        # 
        # 对于30万小资金账户：
        # - 持仓5只股票，每只约6万
        # - 按万三计算：6万 * 0.0003 = 18元
        # - 但如果单笔交易金额 < 1.67万，则佣金不足5元，需按5元收取
        # - 这导致实际费率可能高达 5元/1.67万 ≈ 万3
        #
        # 1. 佣金：max(5元, 交易额 * 费率)
        commission_costs = np.maximum(5.0, trade_amounts * self._commission)
        # 修正：只有发生交易(trades_pct > 0)时才计算佣金，否则为0
        commission_costs = np.where(trades_pct > 0, commission_costs, 0.0)
        
        # 2. 滑点：交易额 * 滑点率
        slippage_costs = trade_amounts * self._slippage
        
        # 3. 转换为收益率扣减
        total_costs = commission_costs + slippage_costs
        cost_penalties = total_costs / current_capital_series
        
        # 计算净收益率
        strategy_returns = strategy_returns_gross - cost_penalties
        
        # 计算净值曲线
        portfolio_values = (1 + strategy_returns).cumprod() * self._initial_capital
        
        # 计算指标
        total_return = portfolio_values.iloc[-1] / self._initial_capital - 1
        
        trading_days = len(portfolio_values)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 夏普比率
        excess_returns = strategy_returns - self._risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() \
            if excess_returns.std() > 0 else 0
        
        # 最大回撤
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        # 胜率和盈亏比
        trade_returns = strategy_returns[trades_pct > 0]
        wins = (trade_returns > 0).sum()
        total_trades = len(trade_returns)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=abs(max_drawdown),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            portfolio_values=portfolio_values,
            trade_records=pd.DataFrame()
        )
    
    def run_optimization(
        self,
        strategy_class: type,
        price_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """
        参数优化
        
        对策略参数进行网格搜索优化。
        
        Parameters
        ----------
        strategy_class : type
            策略类
        price_data : pd.DataFrame
            价格数据
        param_grid : Dict[str, List[Any]]
            参数网格，如 {"short_window": [5, 10], "long_window": [20, 30]}
        
        Returns
        -------
        pd.DataFrame
            优化结果，包含参数组合和对应绩效
        """
        from itertools import product
        
        logger.info(f"开始参数优化，参数网格: {param_grid}")
        
        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        results = []
        total = len(param_combinations)
        
        for i, values in enumerate(param_combinations, 1):
            params = dict(zip(param_names, values))
            logger.debug(f"测试参数组合 ({i}/{total}): {params}")
            
            try:
                strategy = strategy_class(config=params)
                result = self.run(strategy, price_data)
                
                results.append({
                    **params,
                    "total_return": result.total_return,
                    "annual_return": result.annual_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                })
            except Exception as e:
                logger.warning(f"参数组合 {params} 回测失败: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("sharpe_ratio", ascending=False)
        
        logger.info(f"参数优化完成，测试了 {len(results)} 个参数组合")
        return results_df
    
    def compare_strategies(
        self,
        strategies: List[BaseStrategy],
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        策略对比
        
        Parameters
        ----------
        strategies : List[BaseStrategy]
            策略列表
        price_data : pd.DataFrame
            价格数据
        
        Returns
        -------
        pd.DataFrame
            策略对比结果
        """
        logger.info(f"开始策略对比，共 {len(strategies)} 个策略")
        
        results = []
        for strategy in strategies:
            try:
                result = self.run(strategy, price_data)
                results.append({
                    "strategy": strategy.name,
                    "total_return": result.total_return,
                    "annual_return": result.annual_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                })
            except Exception as e:
                logger.warning(f"策略 {strategy.name} 回测失败: {e}")
                continue
        
        return pd.DataFrame(results)


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


class VBTProBacktester:
    """
    VectorBT Pro 全市场回测器
    
    专为A股全市场多标的回测设计，支持固定金额买入策略。
    
    注意：VBT 无法精确模拟 A股"最低5元佣金"规则。
    对于小资金账户（< 50万），建议使用 BacktestEngine 的 Simple 模式。
    
    Attributes
    ----------
    init_cash : float
        初始资金
    fees : float
        交易费率（双边）
    fixed_amount : float
        每次交易固定金额
    
    Examples
    --------
    >>> backtester = VBTProBacktester(init_cash=10_000_000)
    >>> result = backtester.run(close_df, entries_df)
    >>> backtester.plot_cumulative_returns()
    """
    
    # A股交易常量
    DEFAULT_INIT_CASH = 10_000_000  # 1000万初始资金
    DEFAULT_FEES = 0.0003           # 双边万分之三
    DEFAULT_FIXED_AMOUNT = 100_000  # 默认每次买入10万
    TRADING_DAYS_PER_YEAR = 252     # 年交易日
    RISK_FREE_RATE = 0.03           # 无风险利率
    SMALL_CAPITAL_THRESHOLD = 500000  # 小资金阈值：50万
    
    def __init__(
        self,
        init_cash: float = DEFAULT_INIT_CASH,
        fees: float = DEFAULT_FEES,
        fixed_amount: float = DEFAULT_FIXED_AMOUNT,
        slippage: float = 0.0,
        risk_free_rate: float = RISK_FREE_RATE
    ) -> None:
        """
        初始化VectorBT Pro回测器
        
        Parameters
        ----------
        init_cash : float, optional
            初始资金，默认1000万
        fees : float, optional
            交易费率（双边），默认万分之三 (0.0003)
            模拟佣金+印花税均摊
        fixed_amount : float, optional
            每次交易买入的固定金额，默认10万
        slippage : float, optional
            滑点，默认为0
        risk_free_rate : float, optional
            无风险利率，用于计算夏普比率，默认3%
        """
        self.init_cash = init_cash
        self.fees = fees
        self.fixed_amount = fixed_amount
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        # 回测结果缓存
        self._portfolio = None
        self._stats = None
        self._returns = None
        
        # 小资金警告
        if init_cash < self.SMALL_CAPITAL_THRESHOLD:
            logger.warning(
                f"⚠️ 小资金警告：初始资金 {init_cash/10000:.0f}万 < {self.SMALL_CAPITAL_THRESHOLD/10000:.0f}万\n"
                f"   VBT 无法精确模拟 A股最低5元佣金规则，回测结果可能偏乐观\n"
                f"   建议使用 BacktestEngine(config={{'force_simple_mode': True}}) 获取更准确的结果"
            )
        
        logger.info(
            f"VBT Pro回测器初始化: "
            f"初始资金={init_cash/1e6:.0f}M, "
            f"费率={fees*10000:.1f}‱, "
            f"固定金额={fixed_amount/1e4:.0f}万"
        )
    
    def run(
        self,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        exits: Optional[pd.DataFrame] = None,
        freq: str = "1D"
    ) -> Dict[str, Any]:
        """
        执行全市场回测
        
        Parameters
        ----------
        close : pd.DataFrame
            收盘价DataFrame，行索引为日期，列为股票代码
        entries : pd.DataFrame
            买入信号DataFrame (布尔型)，与close相同的shape
        exits : Optional[pd.DataFrame]
            卖出信号DataFrame，如果为None则使用反向entries
        freq : str, optional
            数据频率，默认 '1D' 日线
        
        Returns
        -------
        Dict[str, Any]
            回测结果字典，包含：
            - sharpe_ratio: 夏普比率
            - max_drawdown: 最大回撤
            - annual_return: 年化收益率
            - total_return: 总收益率
            - portfolio: VectorBT Portfolio对象
        
        Examples
        --------
        >>> close = pd.DataFrame(...)  # 全市场收盘价
        >>> entries = signal_generator(close)  # 布尔型买入信号
        >>> result = backtester.run(close, entries)
        >>> print(f"夏普比率: {result['sharpe_ratio']:.2f}")
        """
        if not VBT_AVAILABLE:
            logger.warning("VectorBT未安装，使用内置回测引擎")
            return self._run_fallback(close, entries, exits)
        
        logger.info(
            f"开始VBT Pro回测: "
            f"{len(close.columns)}只股票, "
            f"{len(close)}个交易日"
        )
        
        # 确保entries是布尔型
        entries = entries.astype(bool)
        
        # 生成exits信号（如果未提供）
        if exits is None:
            # 默认策略：下一个买入信号前卖出，或持有到期末
            exits = self._generate_default_exits(entries)
        else:
            exits = exits.astype(bool)
        
        # 计算每只股票的买入数量（固定金额 / 价格）
        size = self._calculate_fixed_amount_size(close, entries)
        
        try:
            # 使用VectorBT构建投资组合
            self._portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                size=size,
                size_type="amount",  # 按股数买入
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                freq=freq,
                cash_sharing=True,  # 全市场共享资金池
                call_seq="auto",    # 自动决定执行顺序
                group_by=True,      # 将所有股票作为一个组合
            )
            
            # 提取统计指标
            self._stats = self._extract_stats()
            self._returns = self._portfolio.returns()
            
            logger.info(
                f"回测完成: "
                f"夏普比率={self._stats['sharpe_ratio']:.2f}, "
                f"年化收益={self._stats['annual_return']:.2%}, "
                f"最大回撤={self._stats['max_drawdown']:.2%}"
            )
            
            return self._stats
            
        except Exception as e:
            logger.error(f"VectorBT回测失败: {e}")
            return self._run_fallback(close, entries, exits)
    
    def _calculate_fixed_amount_size(
        self,
        close: pd.DataFrame,
        entries: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算固定金额对应的买入股数
        
        Parameters
        ----------
        close : pd.DataFrame
            收盘价
        entries : pd.DataFrame
            买入信号
        
        Returns
        -------
        pd.DataFrame
            每次交易的买入股数
        """
        # 计算股数 = 固定金额 / 价格，向下取整到100的倍数（A股最小交易单位）
        size = (self.fixed_amount / close).apply(
            lambda x: (x // 100) * 100
        )
        
        # 只在有买入信号时有效
        size = size.where(entries, 0)
        
        return size
    
    def _generate_default_exits(self, entries: pd.DataFrame) -> pd.DataFrame:
        """
        生成默认的卖出信号
        
        策略：在持有N天后卖出，或遇到反向信号时卖出
        
        Parameters
        ----------
        entries : pd.DataFrame
            买入信号
        
        Returns
        -------
        pd.DataFrame
            卖出信号
        """
        # 简单策略：买入信号shift后作为卖出信号
        # 即每次新的买入信号触发时，卖出之前的持仓
        exits = entries.shift(1, fill_value=False)
        return exits
    
    def _extract_stats(self) -> Dict[str, Any]:
        """
        提取回测统计指标
        
        Returns
        -------
        Dict[str, Any]
            统计指标字典
        """
        pf = self._portfolio
        
        # 基础指标
        total_return = pf.total_return()
        
        # 计算年化收益率
        total_days = len(pf.value())
        years = total_days / self.TRADING_DAYS_PER_YEAR
        
        if years > 0 and (1 + total_return) > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0.0
        
        # 计算夏普比率
        returns = pf.returns()
        excess_returns = returns - self.risk_free_rate / self.TRADING_DAYS_PER_YEAR
        
        if excess_returns.std() > 0:
            sharpe_ratio = (
                np.sqrt(self.TRADING_DAYS_PER_YEAR) * 
                excess_returns.mean() / 
                excess_returns.std()
            )
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        max_drawdown = pf.max_drawdown()
        
        # 其他指标
        try:
            stats_df = pf.stats()
            win_rate = stats_df.get("Win Rate [%]", 0) / 100
            profit_factor = stats_df.get("Profit Factor", 0)
            total_trades = stats_df.get("Total Trades", 0)
        except Exception:
            win_rate = 0.0
            profit_factor = 0.0
            total_trades = 0
        
        return {
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "annual_return": float(annual_return),
            "total_return": float(total_return),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_trades": int(total_trades),
            "portfolio": pf,
            "final_value": float(pf.value().iloc[-1]),
            "init_cash": self.init_cash,
        }
    
    def _run_fallback(
        self,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        exits: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        不依赖VectorBT的回测实现
        
        Parameters
        ----------
        close : pd.DataFrame
            收盘价
        entries : pd.DataFrame
            买入信号
        exits : Optional[pd.DataFrame]
            卖出信号
        
        Returns
        -------
        Dict[str, Any]
            回测结果
        """
        logger.info("使用内置回测引擎...")
        
        entries = entries.astype(bool)
        if exits is None:
            exits = entries.shift(1, fill_value=False)
        else:
            exits = exits.astype(bool)
        
        # 计算每只股票的持仓状态 (1=持有, 0=空仓)
        position = pd.DataFrame(0, index=close.index, columns=close.columns)
        
        for col in close.columns:
            pos = 0
            positions = []
            for i in range(len(close)):
                if entries.iloc[i][col] and pos == 0:
                    pos = 1
                elif exits.iloc[i][col] and pos == 1:
                    pos = 0
                positions.append(pos)
            position[col] = positions
        
        # 计算收益率
        returns = close.pct_change().fillna(0)
        
        # 计算加权收益（固定金额权重）
        weights = position * self.fixed_amount / self.init_cash
        portfolio_returns = (returns * weights.shift(1)).sum(axis=1)
        
        # 扣除交易成本
        trades = position.diff().abs()
        trade_costs = (trades * self.fees).sum(axis=1)
        portfolio_returns = portfolio_returns - trade_costs
        
        # 计算净值曲线
        portfolio_value = (1 + portfolio_returns).cumprod() * self.init_cash
        
        # 计算指标
        total_return = portfolio_value.iloc[-1] / self.init_cash - 1
        
        years = len(portfolio_value) / self.TRADING_DAYS_PER_YEAR
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        excess_returns = portfolio_returns - self.risk_free_rate / self.TRADING_DAYS_PER_YEAR
        sharpe_ratio = (
            np.sqrt(self.TRADING_DAYS_PER_YEAR) * 
            excess_returns.mean() / 
            excess_returns.std()
        ) if excess_returns.std() > 0 else 0
        
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # 存储结果用于绘图
        self._returns = portfolio_returns
        self._portfolio_value = portfolio_value
        
        return {
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "annual_return": float(annual_return),
            "total_return": float(total_return),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": int(trades.sum().sum()),
            "portfolio": None,
            "final_value": float(portfolio_value.iloc[-1]),
            "init_cash": self.init_cash,
        }
    
    def get_portfolio_value(self) -> pd.Series:
        """
        获取组合净值曲线
        
        Returns
        -------
        pd.Series
            净值时间序列
        """
        if self._portfolio is not None:
            return self._portfolio.value()
        elif hasattr(self, "_portfolio_value"):
            return self._portfolio_value
        else:
            raise ValueError("请先调用run()方法执行回测")
    
    def get_returns(self) -> pd.Series:
        """
        获取收益率序列
        
        Returns
        -------
        pd.Series
            日收益率序列
        """
        if self._returns is not None:
            return self._returns
        else:
            raise ValueError("请先调用run()方法执行回测")
    
    def plot_cumulative_returns(
        self,
        benchmark: Optional[pd.Series] = None,
        title: str = "策略累计收益曲线",
        figsize: tuple = (14, 7),
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制累计收益曲线图
        
        Parameters
        ----------
        benchmark : Optional[pd.Series]
            基准收益率序列（如沪深300）
        title : str, optional
            图表标题
        figsize : tuple, optional
            图表大小
        save_path : Optional[str]
            保存路径，如果为None则显示图表
        
        Examples
        --------
        >>> backtester.run(close, entries)
        >>> backtester.plot_cumulative_returns(
        ...     benchmark=hs300_returns,
        ...     title="量化策略 vs 沪深300"
        ... )
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.error("请安装matplotlib: pip install matplotlib")
            return
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 获取净值数据
        portfolio_value = self.get_portfolio_value()
        cumulative_returns = (portfolio_value / self.init_cash - 1) * 100  # 转为百分比
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # ===== 上图：累计收益曲线 =====
        ax1 = axes[0]
        
        # 绘制策略收益
        ax1.plot(
            cumulative_returns.index, 
            cumulative_returns.values,
            label=f'策略 (年化: {self._stats["annual_return"]:.1%})',
            color='#2E86AB',
            linewidth=2
        )
        
        # 绘制基准（如果提供）
        if benchmark is not None:
            benchmark_cum = (1 + benchmark).cumprod()
            benchmark_cum = (benchmark_cum / benchmark_cum.iloc[0] - 1) * 100
            ax1.plot(
                benchmark_cum.index,
                benchmark_cum.values,
                label='基准',
                color='#A23B72',
                linewidth=1.5,
                linestyle='--'
            )
        
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.fill_between(
            cumulative_returns.index,
            0,
            cumulative_returns.values,
            where=cumulative_returns.values >= 0,
            alpha=0.3,
            color='#2E86AB'
        )
        ax1.fill_between(
            cumulative_returns.index,
            0,
            cumulative_returns.values,
            where=cumulative_returns.values < 0,
            alpha=0.3,
            color='#E94F37'
        )
        
        ax1.set_ylabel('累计收益率 (%)', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加关键指标文本框
        stats_text = (
            f"夏普比率: {self._stats['sharpe_ratio']:.2f}\n"
            f"最大回撤: {self._stats['max_drawdown']:.1%}\n"
            f"年化收益: {self._stats['annual_return']:.1%}\n"
            f"总收益率: {self._stats['total_return']:.1%}\n"
            f"期末净值: ¥{self._stats['final_value']/1e6:.2f}M"
        )
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(
            0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props
        )
        
        # ===== 下图：回撤曲线 =====
        ax2 = axes[1]
        
        # 计算回撤
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak * 100
        
        ax2.fill_between(
            drawdown.index,
            0,
            drawdown.values,
            color='#E94F37',
            alpha=0.5
        )
        ax2.plot(drawdown.index, drawdown.values, color='#E94F37', linewidth=1)
        
        ax2.set_ylabel('回撤 (%)', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 标注最大回撤点
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax2.annotate(
            f'最大回撤: {max_dd_val:.1f}%',
            xy=(max_dd_idx, max_dd_val),
            xytext=(max_dd_idx, max_dd_val - 5),
            fontsize=9,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.5)
        )
        
        # 格式化日期轴
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        else:
            plt.show()
    
    def print_summary(self) -> None:
        """
        打印回测结果摘要
        """
        if self._stats is None:
            raise ValueError("请先调用run()方法执行回测")
        
        print("\n" + "=" * 60)
        print("                    回测结果摘要")
        print("=" * 60)
        print(f"  初始资金:        ¥{self.init_cash:,.0f}")
        print(f"  期末净值:        ¥{self._stats['final_value']:,.0f}")
        print(f"  总收益率:        {self._stats['total_return']:.2%}")
        print(f"  年化收益率:      {self._stats['annual_return']:.2%}")
        print("-" * 60)
        print(f"  夏普比率:        {self._stats['sharpe_ratio']:.2f}")
        print(f"  最大回撤:        {self._stats['max_drawdown']:.2%}")
        print(f"  胜率:            {self._stats['win_rate']:.2%}")
        print(f"  盈亏比:          {self._stats['profit_factor']:.2f}")
        print("-" * 60)
        print(f"  总交易次数:      {self._stats['total_trades']}")
        print(f"  交易费率:        {self.fees * 10000:.1f}‱ (双边)")
        print(f"  固定买入金额:    ¥{self.fixed_amount:,.0f}")
        print("=" * 60 + "\n")

