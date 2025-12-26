"""
VectorBT回测流程模块

该模块提供基于VectorBT的回测功能，支持策略绩效评估和分析。
采用向量化回测引擎确保高性能。

重构说明：
- 支持权重驱动的多资产组合交易 (Weight-Driven Multi-Asset Portfolio)
- 符合A股交易规则：整手交易(100股)、涨跌停限制、最低5元佣金
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
    回测引擎 - 权重驱动多资产组合版
    
    支持Weight-Driven执行模式，通过调用策略的generate_target_weights方法
    获取每日目标权重，模拟真实的组合调仓过程。
    
    关键特性：
    - 多资产组合管理（适配ZZ500成分股等）
    - A股最低5元佣金规则
    - 涨跌停板交易限制
    - 整手交易（100股为1手）
    - 先卖后买执行顺序
    
    Attributes
    ----------
    config : Dict[str, Any]
        回测配置
    
    Examples
    --------
    >>> engine = BacktestEngine(config)
    >>> result = engine.run(strategy, price_data, factor_data)
    >>> print(f"夏普比率: {result.sharpe_ratio:.2f}")
    """
    
    # 小资金阈值：低于此值强制使用 Simple 模式
    SMALL_CAPITAL_THRESHOLD = 500000  # 50万
    
    # A股交易常量
    MIN_COMMISSION_CNY = 5.0  # 最低佣金（元）
    ROUND_LOT = 100  # 最小交易单位（股）
    TRADING_DAYS_PER_YEAR = 252
    
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
        self._initial_capital = self.config.get("initial_capital", 300000)
        self._commission = self.config.get("commission", 0.0003)  # 标准A股费率（万分之三）
        self._stamp_duty = self.config.get("stamp_duty", 0.001)  # 印花税（千分之一，仅卖出时收取）
        self._slippage = self.config.get("slippage", 0.001)
        self._risk_free_rate = self.config.get("risk_free_rate", 0.03)
        
        # 强制使用 Simple 模式的标志
        self._force_simple_mode = self.config.get("force_simple_mode", False)
        
        logger.info(
            f"回测引擎初始化: 初始资金={self._initial_capital/10000:.1f}万, "
            f"佣金={self._commission*10000:.1f}‱ (最低{self.MIN_COMMISSION_CNY}元), "
            f"印花税={self._stamp_duty*1000:.1f}‰ (仅卖出), "
            f"滑点={self._slippage*100:.2f}%"
        )
    
    def run(
        self,
        strategy: BaseStrategy,
        price_data: pd.DataFrame,
        factor_data: Optional[pd.DataFrame] = None,
        target_weights: Optional[pd.DataFrame] = None,
        objective: str = "equal_weight",
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> BacktestResult:
        """
        执行回测 - 权重驱动模式
        
        使用策略的generate_target_weights获取目标权重，驱动多资产组合交易。
        
        Parameters
        ----------
        strategy : BaseStrategy
            交易策略实例（需实现generate_target_weights方法）
        price_data : pd.DataFrame
            价格数据，DataFrame格式：
            - Index: DatetimeIndex（交易日期）
            - Columns: 股票代码
            - Values: 收盘价（必须有'close'列或直接以股票代码为列）
        factor_data : Optional[pd.DataFrame]
            因子数据，用于调用generate_target_weights
            如果为None，需要提供target_weights
        target_weights : Optional[pd.DataFrame]
            预计算的目标权重矩阵
            如果提供，则跳过generate_target_weights调用
        objective : str
            权重优化目标，可选：'equal_weight', 'max_sharpe', 'min_volatility'
        benchmark_data : Optional[pd.DataFrame]
            基准指数数据（如沪深300），用于大盘风控。
            需包含 'close' 列，索引为 DatetimeIndex。
            如果为 None，则跳过大盘风控逻辑。
        
        Returns
        -------
        BacktestResult
            回测结果
        
        Raises
        ------
        ValueError
            如果无法生成目标权重
        """
        logger.info(f"开始回测策略: {strategy.name}")
        
        # ========================================
        # Step 1: 准备价格数据
        # ========================================
        # 处理价格数据格式：如果包含'close'列，提取收盘价；否则假设已是股票价格矩阵
        if isinstance(price_data.columns, pd.MultiIndex):
            # 多级列索引，提取收盘价
            close_prices = price_data.xs('close', axis=1, level=1) if 'close' in price_data.columns.get_level_values(1) else price_data
        elif 'close' in price_data.columns and len(price_data.columns) < 10:
            # 单资产数据（包含OHLCV）
            raise ValueError(
                "价格数据需要为多资产DataFrame格式 (Index=日期, Columns=股票代码). "
                "当前数据看起来是单资产OHLCV格式。"
            )
        else:
            # 假设已经是价格矩阵（Index=日期, Columns=股票代码）
            close_prices = price_data.copy()
        
        logger.info(f"价格数据: {len(close_prices)} 天, {len(close_prices.columns)} 只股票")
        
        # ========================================
        # Step 2: 生成目标权重
        # ========================================
        if target_weights is not None:
            # 使用预计算的权重
            weights = target_weights.copy()
            logger.info(f"使用预计算目标权重: {weights.shape}")
        elif hasattr(strategy, 'generate_target_weights') and factor_data is not None:
            # 调用策略的generate_target_weights方法
            logger.info("调用策略 generate_target_weights 生成目标权重...")
            weights = strategy.generate_target_weights(
                factor_data=factor_data,
                prices=close_prices,
                objective=objective,
                risk_free_rate=self._risk_free_rate,
                benchmark_data=benchmark_data
            )
            logger.info(f"目标权重生成完成: {weights.shape}")
        else:
            raise ValueError(
                "无法生成目标权重：需要提供 (factor_data + strategy.generate_target_weights) "
                "或预计算的 target_weights 参数"
            )
        
        # ========================================
        # Step 3: 执行回测
        # ========================================
        # 对齐日期：取权重和价格的交集
        common_dates = weights.index.intersection(close_prices.index)
        if len(common_dates) == 0:
            raise ValueError("权重数据和价格数据没有共同日期")
        
        weights_aligned = weights.loc[common_dates]
        prices_aligned = close_prices.loc[common_dates]
        
        # 确保列对齐
        common_stocks = weights_aligned.columns.intersection(prices_aligned.columns)
        if len(common_stocks) == 0:
            raise ValueError("权重数据和价格数据没有共同股票")
        
        weights_aligned = weights_aligned[common_stocks]
        prices_aligned = prices_aligned[common_stocks]
        
        logger.info(
            f"对齐后数据: {len(common_dates)} 天, {len(common_stocks)} 只股票"
        )
        
        # 执行权重驱动回测
        result = self._run_simple(prices_aligned, weights_aligned)
        
        logger.info(
            f"回测完成: 总收益={result.total_return:.2%}, "
            f"夏普比率={result.sharpe_ratio:.2f}, "
            f"最大回撤={result.max_drawdown:.2%}, "
            f"总交易={result.total_trades}笔"
        )
        
        return result
    
    def _run_simple(
        self,
        price_data: pd.DataFrame,
        target_weights: pd.DataFrame
    ) -> BacktestResult:
        """
        权重驱动的多资产组合回测（Simple模式）
        
        核心特性：
        1. 涨跌停板限制（Limit Down/Up）
        2. 整手交易（Round-Lot，100股为1手）
        3. A股最低5元佣金规则
        4. 先卖后买执行顺序
        
        Parameters
        ----------
        price_data : pd.DataFrame
            价格数据，Index=日期，Columns=股票代码
        target_weights : pd.DataFrame
            目标权重，Index=日期，Columns=股票代码，值为0-1
        
        Returns
        -------
        BacktestResult
            回测结果
        """
        # ========================================
        # Step A: 初始化状态
        # ========================================
        cash = float(self._initial_capital)
        # 持仓：股票代码 -> 持有股数
        shares: Dict[str, int] = {stock: 0 for stock in price_data.columns}
        
        # 记录
        portfolio_values_list: List[float] = []
        trade_records_list: List[Dict[str, Any]] = []
        daily_returns_list: List[float] = []
        
        # 日期和数据
        dates = price_data.index.tolist()
        stocks = list(price_data.columns)
        n_days = len(dates)
        
        # 转换为numpy加速
        prices_arr = price_data.values  # shape: (n_days, n_stocks)
        weights_arr = target_weights.values  # shape: (n_days, n_stocks)
        
        # 股票代码到索引映射
        stock_to_idx = {stock: i for i, stock in enumerate(stocks)}
        
        # ========================================
        # Step B: 涨跌停检测
        # ========================================
        # 计算日收益率用于涨跌停判断
        returns = price_data.pct_change().fillna(0).values
        
        # 涨停：涨幅 >= 9.5%（主板10%，考虑精度误差）
        is_limit_up = returns >= 0.095
        # 跌停：跌幅 <= -9.5%
        is_limit_down = returns <= -0.095
        
        # 统计
        total_limit_up = is_limit_up.sum()
        total_limit_down = is_limit_down.sum()
        if total_limit_up > 0 or total_limit_down > 0:
            logger.info(f"检测到涨停 {total_limit_up} 次, 跌停 {total_limit_down} 次")
        
        # ========================================
        # Step C: 每日交易模拟
        # ========================================
        blocked_sells = 0
        blocked_buys = 0
        insufficient_capital_total = 0  # 小资金不足买入1手的累计次数
        prev_portfolio_value = self._initial_capital
        
        for day_idx in range(n_days):
            date = dates[day_idx]
            today_prices = prices_arr[day_idx]
            today_weights = weights_arr[day_idx]
            today_limit_up = is_limit_up[day_idx]
            today_limit_down = is_limit_down[day_idx]
            
            # ----------------------------------------
            # 1. 计算当前总资产价值（开盘时估算）
            # ----------------------------------------
            position_value = sum(
                shares[stock] * today_prices[stock_to_idx[stock]]
                for stock in stocks
                if not np.isnan(today_prices[stock_to_idx[stock]])
            )
            total_equity = cash + position_value
            
            # ----------------------------------------
            # 2. 计算目标持仓（基于目标权重）
            # ----------------------------------------
            target_shares_dict: Dict[str, int] = {}
            insufficient_capital_count = 0  # 记录资金不足无法买1手的情况
            
            for stock_idx, stock in enumerate(stocks):
                price = today_prices[stock_idx]
                weight = today_weights[stock_idx]
                
                # 跳过无效价格
                if np.isnan(price) or price <= 0:
                    target_shares_dict[stock] = 0
                    continue
                
                # 计算目标市值
                target_value = total_equity * weight
                
                # 计算目标股数（向下取整到整手）
                target_shares = int(target_value / price / self.ROUND_LOT) * self.ROUND_LOT
                
                # ===== 小资金警告：目标金额不足买入1手 =====
                if target_value > 0 and target_shares == 0:
                    insufficient_capital_count += 1
                    min_required = price * self.ROUND_LOT  # 买1手需要的最小资金
                    logger.warning(
                        f"{stock} 目标金额 {target_value:.0f} 不足买入1手 "
                        f"(股价 {price:.2f}, 需要 {min_required:.0f})，已放弃买入"
                    )
                
                target_shares_dict[stock] = max(0, target_shares)
            
            # 累计小资金不足的次数
            insufficient_capital_total += insufficient_capital_count
            
            # ----------------------------------------
            # 3. 执行交易（先卖后买）
            # ----------------------------------------
            # 3.1 卖出操作（释放现金）
            for stock_idx, stock in enumerate(stocks):
                current_shares = shares[stock]
                target = target_shares_dict[stock]
                price = today_prices[stock_idx]
                
                if np.isnan(price) or price <= 0:
                    continue
                
                if current_shares > target:
                    # 需要卖出
                    sell_shares = current_shares - target
                    
                    # 检查跌停限制
                    if today_limit_down[stock_idx]:
                        blocked_sells += 1
                        continue  # 跌停无法卖出
                    
                    # 执行卖出
                    sell_value = sell_shares * price
                    
                    # ===== 小资金实盘优化：精确计算卖出成本 =====
                    # 1. 佣金：max(5元最低佣金, 交易额 * 万分之三)
                    commission = max(self.MIN_COMMISSION_CNY, sell_value * self._commission)
                    # 2. 印花税：仅卖出时收取（千分之一）
                    stamp_duty = sell_value * self._stamp_duty
                    # 3. 滑点成本
                    slippage_cost = sell_value * self._slippage
                    
                    # 更新现金和持仓（卖出所得 = 成交额 - 佣金 - 印花税 - 滑点）
                    net_proceeds = sell_value - commission - stamp_duty - slippage_cost
                    cash += net_proceeds
                    shares[stock] = target
                    
                    # 记录交易
                    trade_records_list.append({
                        'date': date,
                        'symbol': stock,
                        'action': 'SELL',
                        'shares': sell_shares,
                        'price': price,
                        'value': sell_value,
                        'commission': commission,
                        'stamp_duty': stamp_duty,  # 新增：印花税记录
                        'slippage': slippage_cost,
                        'net_proceeds': net_proceeds
                    })
            
            # 3.2 买入操作（使用可用现金）
            # 按权重排序，优先买入高权重股票
            buy_orders: List[tuple] = []
            for stock_idx, stock in enumerate(stocks):
                current_shares = shares[stock]
                target = target_shares_dict[stock]
                
                if target > current_shares:
                    weight = today_weights[stock_idx]
                    buy_orders.append((stock, stock_idx, target - current_shares, weight))
            
            # 按权重降序排列
            buy_orders.sort(key=lambda x: x[3], reverse=True)
            
            for stock, stock_idx, buy_shares, _ in buy_orders:
                price = today_prices[stock_idx]
                
                if np.isnan(price) or price <= 0:
                    continue
                
                # 检查涨停限制
                if today_limit_up[stock_idx]:
                    blocked_buys += 1
                    continue  # 涨停无法买入
                
                # ===== 小资金实盘优化：精确计算买入成本 =====
                # 注意：印花税仅卖出时收取，买入不收
                buy_value = buy_shares * price
                # 佣金：max(5元最低佣金, 交易额 * 万分之三)
                commission = max(self.MIN_COMMISSION_CNY, buy_value * self._commission)
                slippage_cost = buy_value * self._slippage
                # 买入总成本 = 成交额 + 佣金 + 滑点（无印花税）
                total_cost = buy_value + commission + slippage_cost
                
                # 检查现金是否充足
                if total_cost > cash:
                    # 调整买入数量到可负担的最大整手数
                    affordable_value = cash / (1 + self._commission + self._slippage)
                    buy_shares = int(affordable_value / price / self.ROUND_LOT) * self.ROUND_LOT
                    
                    if buy_shares < self.ROUND_LOT:
                        # ===== 小资金警告：现金不足买入1手 =====
                        min_required = price * self.ROUND_LOT * (1 + self._commission + self._slippage)
                        logger.warning(
                            f"{stock} 剩余现金 {cash:.0f} 不足买入1手 "
                            f"(股价 {price:.2f}, 需要 {min_required:.0f})，已放弃买入"
                        )
                        continue  # 资金不足，跳过（现金保留，流转到下一只股票）
                    
                    buy_value = buy_shares * price
                    commission = max(self.MIN_COMMISSION_CNY, buy_value * self._commission)
                    slippage_cost = buy_value * self._slippage
                    total_cost = buy_value + commission + slippage_cost
                
                # 执行买入
                cash -= total_cost
                shares[stock] += buy_shares
                
                # 记录交易
                trade_records_list.append({
                    'date': date,
                    'symbol': stock,
                    'action': 'BUY',
                    'shares': buy_shares,
                    'price': price,
                    'value': buy_value,
                    'commission': commission,
                    'stamp_duty': 0.0,  # 买入不收印花税
                    'slippage': slippage_cost,
                    'total_cost': total_cost
                })
            
            # ----------------------------------------
            # 4. 计算收盘时组合价值
            # ----------------------------------------
            position_value_eod = sum(
                shares[stock] * today_prices[stock_to_idx[stock]]
                for stock in stocks
                if not np.isnan(today_prices[stock_to_idx[stock]])
            )
            current_portfolio_value = cash + position_value_eod
            portfolio_values_list.append(current_portfolio_value)
            
            # 计算日收益率
            daily_return = (current_portfolio_value / prev_portfolio_value) - 1 if prev_portfolio_value > 0 else 0
            daily_returns_list.append(daily_return)
            prev_portfolio_value = current_portfolio_value
        
        # 统计受限交易
        if blocked_sells > 0 or blocked_buys > 0:
            logger.info(f"受涨跌停限制: {blocked_sells} 次卖出被阻止, {blocked_buys} 次买入被阻止")
        
        # 统计小资金不足情况
        if insufficient_capital_total > 0:
            logger.warning(
                f"⚠️ 小资金警告: 共 {insufficient_capital_total} 次因资金不足无法买入1手，"
                f"建议增加本金或减少持仓数量 (top_n)"
            )
        
        # ========================================
        # Step D: 计算绩效指标
        # ========================================
        portfolio_values = pd.Series(portfolio_values_list, index=dates)
        strategy_returns = pd.Series(daily_returns_list, index=dates)
        
        # 总收益率
        total_return = portfolio_values.iloc[-1] / self._initial_capital - 1
        
        # 年化收益率
        trading_days = len(portfolio_values)
        years = trading_days / self.TRADING_DAYS_PER_YEAR
        if years > 0 and (1 + total_return) > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0.0
        
        # 夏普比率
        excess_returns = strategy_returns - self._risk_free_rate / self.TRADING_DAYS_PER_YEAR
        if excess_returns.std() > 0:
            sharpe_ratio = np.sqrt(self.TRADING_DAYS_PER_YEAR) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # 交易统计
        trade_records = pd.DataFrame(trade_records_list)
        total_trades = len(trade_records_list)
        
        # 胜率和盈亏比计算
        win_rate = 0.0
        profit_factor = 0.0
        
        if total_trades > 0 and not trade_records.empty:
            # 按股票分组计算完整交易的盈亏
            completed_trades = self._calculate_trade_pnl(trade_records)
            
            if len(completed_trades) > 0:
                wins = completed_trades[completed_trades['pnl'] > 0]
                losses = completed_trades[completed_trades['pnl'] <= 0]
                
                win_rate = len(wins) / len(completed_trades) if len(completed_trades) > 0 else 0
                
                gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
                gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor if not np.isinf(profit_factor) else 0.0,
            total_trades=total_trades,
            portfolio_values=portfolio_values,
            trade_records=trade_records
        )
    
    def _calculate_trade_pnl(self, trade_records: pd.DataFrame) -> pd.DataFrame:
        """
        计算每笔完整交易的盈亏
        
        将买入和卖出配对，计算实际盈亏。
        
        Parameters
        ----------
        trade_records : pd.DataFrame
            交易记录
        
        Returns
        -------
        pd.DataFrame
            完整交易的盈亏记录
        """
        if trade_records.empty:
            return pd.DataFrame()
        
        completed = []
        
        # 按股票分组
        for symbol in trade_records['symbol'].unique():
            symbol_trades = trade_records[trade_records['symbol'] == symbol].sort_values('date')
            
            # 追踪持仓成本
            position = 0
            avg_cost = 0.0
            
            for _, trade in symbol_trades.iterrows():
                if trade['action'] == 'BUY':
                    # 更新平均成本
                    new_shares = trade['shares']
                    buy_cost = trade.get('total_cost', trade['value'] + trade['commission'])
                    
                    if position + new_shares > 0:
                        avg_cost = (avg_cost * position + buy_cost) / (position + new_shares)
                    position += new_shares
                    
                elif trade['action'] == 'SELL' and position > 0:
                    # 计算卖出盈亏
                    sell_shares = min(trade['shares'], position)
                    sell_proceeds = trade.get('net_proceeds', trade['value'] - trade['commission'])
                    cost_basis = avg_cost * sell_shares / position * position if position > 0 else 0
                    
                    # 按比例计算成本
                    cost_basis = (sell_shares / position) * (avg_cost * position) if position > 0 else 0
                    pnl = (sell_proceeds / sell_shares - avg_cost) * sell_shares if sell_shares > 0 else 0
                    
                    completed.append({
                        'symbol': symbol,
                        'entry_date': symbol_trades[symbol_trades['action'] == 'BUY']['date'].iloc[0] if len(symbol_trades[symbol_trades['action'] == 'BUY']) > 0 else trade['date'],
                        'exit_date': trade['date'],
                        'shares': sell_shares,
                        'pnl': pnl
                    })
                    
                    position -= sell_shares
        
        return pd.DataFrame(completed)
    
    def run_optimization(
        self,
        strategy_class: type,
        price_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        factor_data: Optional[pd.DataFrame] = None
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
        factor_data : Optional[pd.DataFrame]
            因子数据
        
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
                result = self.run(strategy, price_data, factor_data=factor_data)
                
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
        price_data: pd.DataFrame,
        factor_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        策略对比
        
        Parameters
        ----------
        strategies : List[BaseStrategy]
            策略列表
        price_data : pd.DataFrame
            价格数据
        factor_data : Optional[pd.DataFrame]
            因子数据
        
        Returns
        -------
        pd.DataFrame
            策略对比结果
        """
        logger.info(f"开始策略对比，共 {len(strategies)} 个策略")
        
        results = []
        for strategy in strategies:
            try:
                result = self.run(strategy, price_data, factor_data=factor_data)
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
        stamp_duty: float = 0.001,
        risk_free_rate: float = RISK_FREE_RATE
    ) -> None:
        """
        初始化VectorBT Pro回测器
        
        Parameters
        ----------
        init_cash : float, optional
            初始资金，默认1000万
        fees : float, optional
            交易费率（佣金），默认万分之三 (0.0003)
        fixed_amount : float, optional
            每次交易买入的固定金额，默认10万
        slippage : float, optional
            滑点，默认为0
        stamp_duty : float, optional
            印花税率（仅卖出时收取），默认千分之一 (0.001)
        risk_free_rate : float, optional
            无风险利率，用于计算夏普比率，默认3%
        
        Notes
        -----
        小资金实盘优化：
        - 佣金执行 max(5元, 交易额 * fees) 规则
        - 印花税仅在卖出时收取
        """
        self.init_cash = init_cash
        self.fees = fees
        self.fixed_amount = fixed_amount
        self.slippage = slippage
        self.stamp_duty = stamp_duty  # 印花税（仅卖出）
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
    
    # A股最低佣金（元）
    MIN_COMMISSION_CNY = 5.0
    
    def _run_fallback(
        self,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        exits: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        不依赖VectorBT的回测实现
        
        增强版功能：
        1. A股"最低5元佣金"规则
        2. 跌停板卖出限制（Limit Down Lock-up）
        3. 整手交易（Round-Lot，100股为1手）
        
        Parameters
        ----------
        close : pd.DataFrame
            收盘价
        entries : pd.DataFrame
            买入信号
        exits : Optional[pd.DataFrame]
            卖出信号
        low : Optional[pd.DataFrame]
            最低价（用于跌停判断）
        
        Returns
        -------
        Dict[str, Any]
            回测结果
        """
        logger.info("使用内置回测引擎（含最低5元佣金规则、跌停限制、整手交易）...")
        
        entries = entries.astype(bool)
        if exits is None:
            exits = entries.shift(1, fill_value=False)
        else:
            exits = exits.astype(bool)
        
        # ========================================
        # 跌停板检测（Limit Down Detection）
        # ========================================
        is_limit_down = pd.DataFrame(False, index=close.index, columns=close.columns)
        
        if low is not None:
            pct_change = close.pct_change()
            # 跌停判断：最低价 == 收盘价 且 跌幅 >= 9.5%
            is_limit_down = (low == close) & (pct_change <= -0.095)
            is_limit_down = is_limit_down.fillna(False)
            
            limit_down_count = is_limit_down.sum().sum()
            if limit_down_count > 0:
                logger.info(f"检测到 {limit_down_count} 次跌停，卖出信号将被延迟处理")
        
        # ========================================
        # 整手交易模拟（Round-Lot Simulation）
        # ========================================
        # 记录每只股票的实际持仓股数（整手）
        shares_held = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=int)
        blocked_exits_count = 0
        total_trades = 0
        
        for col in close.columns:
            current_shares = 0
            shares_list = []
            pending_exit = False
            
            for i in range(len(close)):
                price = close.iloc[i][col]
                
                # 跳过价格为空的情况
                if pd.isna(price) or price <= 0:
                    shares_list.append(current_shares)
                    continue
                
                # 检查待执行卖出
                if pending_exit and current_shares > 0:
                    if not is_limit_down.iloc[i][col]:
                        current_shares = 0
                        pending_exit = False
                        total_trades += 1
                
                # 处理买入信号
                if entries.iloc[i][col] and current_shares == 0:
                    # 计算整手股数: (固定金额 / 价格) 向下取整到100
                    target_shares = int((self.fixed_amount / price) // 100) * 100
                    if target_shares >= 100:
                        current_shares = target_shares
                        pending_exit = False
                        total_trades += 1
                
                # 处理卖出信号
                elif exits.iloc[i][col] and current_shares > 0:
                    if is_limit_down.iloc[i][col]:
                        pending_exit = True
                        blocked_exits_count += 1
                    else:
                        current_shares = 0
                        total_trades += 1
                
                shares_list.append(current_shares)
            
            shares_held[col] = shares_list
        
        if blocked_exits_count > 0:
            logger.info(f"因跌停限制，{blocked_exits_count} 次卖出被延迟")
        
        # ========================================
        # 计算持仓市值和收益
        # ========================================
        # 持仓市值 = 股数 * 价格
        position_values = shares_held * close
        
        # 计算每日持仓市值变化（用于收益率）
        returns = close.pct_change().fillna(0)
        
        # 加权收益率：按前一日持仓占比计算
        prev_position_values = position_values.shift(1).fillna(0)
        prev_total_value = prev_position_values.sum(axis=1)
        
        # 初始化组合价值
        cash = self.init_cash
        portfolio_values = []
        portfolio_returns = []
        
        for i in range(len(close)):
            if i == 0:
                # 首日：初始资金
                total_value = self.init_cash
                portfolio_values.append(total_value)
                portfolio_returns.append(0.0)
                continue
            
            # 计算前一日持仓的当日收益
            prev_shares = shares_held.iloc[i - 1]
            curr_prices = close.iloc[i]
            prev_prices = close.iloc[i - 1]
            
            # 持仓收益 = sum(股数 * (今日价格 - 昨日价格))
            position_pnl = (prev_shares * (curr_prices - prev_prices)).sum()
            
            # 检测交易（股数变化）
            curr_shares = shares_held.iloc[i]
            share_changes = curr_shares - prev_shares
            
            trade_cost = 0.0
            for col in close.columns:
                change = share_changes[col]
                if change != 0:
                    trade_value = abs(change) * curr_prices[col]
                    # ===== 小资金实盘优化：精确计算交易成本 =====
                    # 1. 佣金：max(5元最低佣金, 交易额 * 万分之三)
                    commission = max(self.MIN_COMMISSION_CNY, trade_value * self.fees)
                    # 2. 印花税：仅卖出时收取（千分之一）
                    stamp_duty = trade_value * self.stamp_duty if change < 0 else 0.0
                    # 3. 滑点
                    slippage = trade_value * self.slippage
                    trade_cost += commission + stamp_duty + slippage
            
            # 更新组合价值
            prev_total = portfolio_values[-1]
            curr_total = prev_total + position_pnl - trade_cost
            portfolio_values.append(curr_total)
            
            # 日收益率
            daily_return = (curr_total / prev_total) - 1 if prev_total > 0 else 0
            portfolio_returns.append(daily_return)
        
        # 转换为Series
        portfolio_value = pd.Series(portfolio_values, index=close.index)
        portfolio_returns_series = pd.Series(portfolio_returns, index=close.index)
        
        # ========================================
        # 计算绩效指标
        # ========================================
        total_return = portfolio_value.iloc[-1] / self.init_cash - 1
        
        years = len(portfolio_value) / self.TRADING_DAYS_PER_YEAR
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        excess_returns = portfolio_returns_series - self.risk_free_rate / self.TRADING_DAYS_PER_YEAR
        sharpe_ratio = (
            np.sqrt(self.TRADING_DAYS_PER_YEAR) * 
            excess_returns.mean() / 
            excess_returns.std()
        ) if excess_returns.std() > 0 else 0
        
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # 存储结果用于绘图
        self._returns = portfolio_returns_series
        self._portfolio_value = portfolio_value
        
        return {
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "annual_return": float(annual_return),
            "total_return": float(total_return),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": total_trades,
            "portfolio": None,
            "final_value": float(portfolio_value.iloc[-1]),
            "init_cash": self.init_cash,
            "blocked_exits": blocked_exits_count,
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
