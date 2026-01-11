"""
回测引擎模块

提供权重驱动的多资产组合回测功能。
"""

from typing import Optional, Dict, Any, List
import logging

import pandas as pd
import numpy as np

from ..strategy import BaseStrategy
from .result import BacktestResult

logger = logging.getLogger(__name__)


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
        
        # =======================================================
        # [CRITICAL FIX] 防止前视偏差 (Look-ahead Bias)
        # =======================================================
        # 策略在 Day T 收盘后生成信号/权重，只能在 Day T+1 执行
        # 因此必须将权重滞后一天
        logger.info("应用 T+1 交易规则：将目标权重滞后一天执行")
        weights = weights.shift(1).fillna(0.0)

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
            
            # [FIX] Lazy Rebalancing: 检查当日权重是否全为 NaN
            # 如果全为 NaN，说明是非调仓日，直接沿用当前持仓（Buy and Hold）
            is_lazy_day = np.isnan(today_weights).all()
            
            if is_lazy_day:
                # 非调仓日：目标持仓 = 当前持仓
                target_shares_dict = shares.copy()
            else:
                # 调仓日：正常计算目标权重
                for stock_idx, stock in enumerate(stocks):
                    price = today_prices[stock_idx]
                    weight = today_weights[stock_idx]
                    
                    # 处理 NaN 权重：视为 0
                    if np.isnan(weight):
                        weight = 0.0
                    
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

