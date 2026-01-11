"""
VectorBT Pro 全市场回测器

专为A股全市场多标的回测设计，支持固定金额买入策略。
"""

from typing import Optional, Dict, Any
import logging

import pandas as pd
import numpy as np

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    MIN_COMMISSION_CNY = 5.0  # A股最低佣金（元）
    
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
        self._portfolio_value = None
        
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
        # [WARNING] 这是一个极简的退出逻辑，仅用于演示或基准测试
        # 实际策略应基于止损/止盈或反向信号
        logger.warning("使用默认极简退出策略 (Next Entry Rebalance)，可能不符合预期")
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
        elif self._portfolio_value is not None:
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

