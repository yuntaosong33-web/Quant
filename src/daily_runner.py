"""
每日更新运行器模块

本模块提供每日数据更新、因子计算、调仓信号生成和报告输出的核心逻辑。
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json
import os

import pandas as pd
import numpy as np

from .strategy import MultiFactorStrategy
from .report_generator import ReportGenerator

# 延迟导入，避免循环依赖
TushareDataLoader = None
AShareDataCleaner = None
DataLoader = None

logger = logging.getLogger(__name__)

# 默认路径
DATA_RAW_PATH = Path("data/raw")
DATA_PROCESSED_PATH = Path("data/processed")
REPORTS_PATH = Path("reports")


def _lazy_import():
    """延迟导入重型模块"""
    global TushareDataLoader, AShareDataCleaner, DataLoader
    
    if TushareDataLoader is None:
        try:
            from .tushare import TushareDataLoader
        except ImportError:
            TushareDataLoader = None
    
    if AShareDataCleaner is None:
        try:
            from .data_loader import AShareDataCleaner
        except ImportError:
            AShareDataCleaner = None


class DailyUpdateRunner:
    """
    每日更新运行器
    
    负责执行每日数据更新、因子计算、调仓信号生成和报告输出。
    
    Parameters
    ----------
    config : Optional[Dict[str, Any]]
        配置参数
    
    Attributes
    ----------
    config : Dict[str, Any]
        配置参数
    tushare_loader : TushareDataLoader
        Tushare 数据加载器
    strategy : MultiFactorStrategy
        多因子策略
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化每日更新运行器"""
        _lazy_import()
        
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 确保目录存在
        DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
        
        # 状态变量
        self.today = pd.Timestamp.now().normalize()
        self.ohlcv_data: Optional[pd.DataFrame] = None
        self.financial_data: Optional[pd.DataFrame] = None
        self.industry_data: Optional[pd.DataFrame] = None
        self.factor_data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.current_positions: Dict[str, float] = {}
        self.target_positions: Dict[str, float] = {}
        
        # 报告生成器
        self.report_generator = ReportGenerator(self.config, REPORTS_PATH)
        
        # 加载当前持仓
        self.load_current_holdings()
        
        self.logger.info("DailyUpdateRunner 初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "data": {
                "stock_pool": "hs300",
                "start_date": "2020-01-01",
                "update_days": 5,
            },
            "strategy": {
                "name": "Multi-Factor Strategy",
                "value_weight": 0.0,
                "quality_weight": 0.3,
                "momentum_weight": 0.7,
                "top_n": 5,
                "min_listing_days": 126,
            },
            "portfolio": {
                "total_capital": 300000,
                "max_weight": 0.25,
                "risk_free_rate": 0.02,
            },
            "report": {
                "format": "markdown",
                "output_dir": "reports",
            },
        }
    
    def _init_components(self) -> None:
        """初始化各组件"""
        # Tushare 数据加载器
        tushare_config = self.config.get("tushare", {})
        api_token = tushare_config.get("api_token") or os.environ.get("TUSHARE_TOKEN", "")
        
        if not api_token:
            self.logger.warning("Tushare API Token 未配置，部分功能可能不可用")
        
        self.tushare_loader = TushareDataLoader(
            api_token=api_token,
            cache_dir=tushare_config.get("cache_dir", "data/tushare_cache")
        )
        
        self.data_cleaner = AShareDataCleaner()
        
        # 策略
        strategy_config = self.config.get("strategy", {})
        llm_config = self.config.get("llm", {})
        
        self.strategy = MultiFactorStrategy(
            name=strategy_config.get("name", "Multi-Factor Strategy"),
            config={
                "value_weight": strategy_config.get("value_weight", 0.0),
                "quality_weight": strategy_config.get("quality_weight", 0.3),
                "momentum_weight": strategy_config.get("momentum_weight", 0.7),
                "size_weight": strategy_config.get("size_weight", 0.0),
                "sentiment_weight": strategy_config.get("sentiment_weight", 0.0),
                "top_n": strategy_config.get("top_n", 5),
                "min_listing_days": strategy_config.get("min_listing_days", 126),
                "exclude_chinext": strategy_config.get("exclude_chinext", False),
                "exclude_star": strategy_config.get("exclude_star", False),
                "value_col": strategy_config.get("value_col", "value_zscore"),
                "quality_col": strategy_config.get("quality_col", "turnover_5d_zscore"),
                "momentum_col": strategy_config.get("momentum_col", "sharpe_20_zscore"),
                "size_col": strategy_config.get("size_col", "small_cap_zscore"),
                "rebalance_frequency": strategy_config.get("rebalance_frequency", "monthly"),
                "rebalance_buffer": strategy_config.get("rebalance_buffer", 0.05),
                "holding_bonus": strategy_config.get("holding_bonus", 0.0),
                "market_risk": self.config.get("risk", {}).get("market_risk", {}),
                "llm": llm_config,
            }
        )
    
    def _validate_data_units(self, df: pd.DataFrame, data_type: str) -> None:
        """
        对数据进行单位一致性检查，并记录警告。
        
        Tushare 数据的标准单位:
        - volume: 股 (手需要 * 100)
        - amount: 千元 (元需要 * 1000)
        - total_mv/circ_mv: 万元
        
        Parameters
        ----------
        df : pd.DataFrame
            待检查的数据
        data_type : str
            数据类型: "ohlcv" 或 "financial"
        """
        if df.empty:
            return
        
        if data_type == "ohlcv":
            # 检查成交量单位 (预期为股，正常股票单日成交量应 > 10万股)
            if 'volume' in df.columns:
                median_vol = df['volume'].median()
                if median_vol < 1000:
                    self.logger.warning(
                        f"⚠️ OHLCV 'volume' 单位可能错误 (中位数 {median_vol:.0f})，"
                        f"预期为股，当前可能为手"
                    )
            
            # 检查成交额单位 (Tushare 原始为千元)
            if 'amount' in df.columns:
                median_amt = df['amount'].median()
                if median_amt > 1e9:
                    self.logger.warning(
                        f"⚠️ OHLCV 'amount' 可能已转换为元 (中位数 {median_amt:.0f})，"
                        f"请确认单位一致性"
                    )
                    
        elif data_type == "financial":
            # 检查市值单位 (Tushare 原始为万元)
            for col in ['total_mv', 'circ_mv']:
                if col in df.columns:
                    max_val = df[col].max()
                    # 万元单位下，千亿市值 = 1e7 万元
                    if max_val > 1e12:
                        self.logger.warning(
                            f"⚠️ 财务数据 '{col}' 可能已转换为元 "
                            f"(最大值 {max_val:.2e})，请确认单位一致性"
                        )
                    elif max_val < 1e6:
                        self.logger.warning(
                            f"⚠️ 财务数据 '{col}' 可能单位错误 "
                            f"(最大值 {max_val:.2e})，预期为万元"
                        )
    
    def load_current_holdings(self) -> None:
        """加载当前持仓"""
        holdings_path = DATA_PROCESSED_PATH / "real_holdings.json"
        
        if holdings_path.exists():
            try:
                with open(holdings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.current_positions = data.get("positions", {})
                self.current_positions = {str(k): float(v) for k, v in self.current_positions.items()}
                
                self.logger.info(
                    f"已加载持仓数据: {len(self.current_positions)} 只股票, "
                    f"总市值 ¥{sum(self.current_positions.values()):,.0f}"
                )
            except Exception as e:
                self.logger.warning(f"加载持仓文件失败: {e}")
                self.current_positions = {}
        else:
            self.logger.info("持仓文件不存在，初始化为空持仓")
            self.current_positions = {}
    
    def save_current_holdings(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float]
    ) -> None:
        """保存当前持仓"""
        new_positions = self.current_positions.copy()
        
        for stock, amount in buy_orders.items():
            new_positions[stock] = new_positions.get(stock, 0) + amount
        
        for stock, amount in sell_orders.items():
            if stock in new_positions:
                new_positions[stock] -= amount
                if new_positions[stock] <= 0:
                    del new_positions[stock]
        
        holdings_data = {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "update_date": self.today.strftime("%Y-%m-%d"),
            "positions": new_positions,
            "total_value": sum(new_positions.values()),
            "num_stocks": len(new_positions),
        }
        
        holdings_path = DATA_PROCESSED_PATH / "real_holdings.json"
        
        try:
            with open(holdings_path, 'w', encoding='utf-8') as f:
                json.dump(holdings_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"持仓已更新: {len(new_positions)} 只股票")
            self.current_positions = new_positions
        except Exception as e:
            self.logger.error(f"保存持仓文件失败: {e}")
    
    def update_market_data(self) -> bool:
        """更新市场数据"""
        self.logger.info("开始更新市场数据...")
        
        try:
            # 检查今日缓存
            ohlcv_path = DATA_RAW_PATH / f"ohlcv_{self.today.strftime('%Y%m%d')}.parquet"
            if ohlcv_path.exists():
                try:
                    self.ohlcv_data = pd.read_parquet(ohlcv_path)
                    if not self.ohlcv_data.empty:
                        self.logger.info(f"使用缓存数据: {ohlcv_path.name}")
                        return True
                except Exception as e:
                    self.logger.warning(f"读取缓存失败: {e}")
            
            data_config = self.config.get("data", {})
            stock_pool = data_config.get("stock_pool", "hs300")
            
            end_date = self.today.strftime("%Y%m%d")
            update_days = data_config.get("update_days", 5)
            start_date = (self.today - timedelta(days=update_days * 2)).strftime("%Y%m%d")
            
            # 获取股票列表
            if stock_pool == "all":
                stock_list = self.tushare_loader.fetch_all_stocks()
            else:
                stock_list = self.tushare_loader.fetch_index_constituents(stock_pool)
            
            if not stock_list:
                self.logger.error(f"无法获取 {stock_pool} 股票列表")
                return False
            
            self.logger.info(f"股票池: {stock_pool}, 股票数量: {len(stock_list)}")
            
            # 批量获取日线数据
            self.ohlcv_data = self.tushare_loader.fetch_daily_data_batch(
                stock_list, start_date, end_date
            )
            
            if self.ohlcv_data is None or self.ohlcv_data.empty:
                self.logger.error("未获取到任何 OHLCV 数据")
                return False
            
            self.logger.info(f"OHLCV 数据更新完成，共 {len(self.ohlcv_data)} 条记录")
            
            # 保存数据
            self.ohlcv_data.to_parquet(ohlcv_path)
            self._current_stock_list = stock_list
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {e}")
            return False
    
    def update_financial_data(self) -> bool:
        """更新财务数据"""
        self.logger.info("开始更新财务数据...")
        
        financial_path = DATA_RAW_PATH / f"financial_{self.today.strftime('%Y%m%d')}.parquet"
        if financial_path.exists():
            try:
                self.financial_data = pd.read_parquet(financial_path)
                if not self.financial_data.empty:
                    self.logger.info(f"使用缓存数据: {financial_path.name}")
                    return True
            except Exception as e:
                self.logger.warning(f"读取缓存失败: {e}")
        
        try:
            if self.ohlcv_data is None:
                return False
            
            stocks = self.ohlcv_data['stock_code'].unique().tolist()
            
            # 获取每日基础指标
            basic_df = self.tushare_loader.fetch_daily_basic(stock_list=stocks)
            fina_df = self.tushare_loader.fetch_financial_batch(stocks, show_progress=True)
            
            if not basic_df.empty and not fina_df.empty:
                merged_df = basic_df.merge(
                    fina_df[['stock_code', 'roe']].drop_duplicates(),
                    on='stock_code',
                    how='left'
                )
            elif not basic_df.empty:
                merged_df = basic_df
            elif not fina_df.empty:
                merged_df = fina_df
            else:
                return False
            
            self.financial_data = merged_df
            self.financial_data.to_parquet(financial_path)
            
            self.logger.info(f"财务数据更新完成: {len(self.financial_data)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"财务数据获取失败: {e}")
            return False
    
    def update_benchmark_data(self) -> bool:
        """更新基准指数数据（用于大盘风控）"""
        self.logger.info("开始更新基准指数数据...")
        
        try:
            risk_config = self.config.get("risk", {}).get("market_risk", {})
            benchmark_code = risk_config.get("benchmark", "000300")
            
            end_date = self.today.strftime("%Y%m%d")
            start_date = (self.today - timedelta(days=120)).strftime("%Y%m%d")
            
            self.benchmark_data = self.tushare_loader.fetch_index_daily(
                benchmark_code, start_date, end_date
            )
            
            if self.benchmark_data is not None and not self.benchmark_data.empty:
                self.logger.info(f"基准指数数据更新完成: {len(self.benchmark_data)} 条")
                return True
            else:
                self.logger.warning("未获取到基准指数数据，大盘风控可能不生效")
                return False
                
        except Exception as e:
            self.logger.warning(f"基准指数数据获取失败: {e}")
            return False
    
    def calculate_factors(self) -> bool:
        """计算因子数据"""
        self.logger.info("开始计算因子数据...")
        
        try:
            if self.ohlcv_data is None or self.ohlcv_data.empty:
                self.logger.error("OHLCV 数据为空，无法计算因子")
                return False
            
            # ========================================
            # 数据单位一致性检查
            # ========================================
            self._validate_data_units(self.ohlcv_data, "ohlcv")
            
            # 合并 OHLCV 和财务数据
            df = self.ohlcv_data.copy()
            
            if self.financial_data is not None and not self.financial_data.empty:
                self._validate_data_units(self.financial_data, "financial")
                df = df.merge(
                    self.financial_data,
                    on='stock_code',
                    how='left',
                    suffixes=('', '_fin')
                )
            
            # 计算技术因子（按股票分组）
            factor_dfs = []
            
            for stock_code, group in df.groupby('stock_code'):
                group = group.sort_values('trade_date')
                
                # RSI
                if 'close' in group.columns:
                    delta = group['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = (-delta).where(delta < 0, 0)
                    
                    avg_gain = gain.ewm(alpha=1/20, min_periods=20).mean()
                    avg_loss = loss.ewm(alpha=1/20, min_periods=20).mean()
                    rs = avg_gain / avg_loss.replace(0, np.nan)
                    group['rsi_20'] = 100 - (100 / (1 + rs))
                
                # 换手率5日均值
                if 'turnover_rate' in group.columns:
                    group['turnover_5d'] = group['turnover_rate'].rolling(5).mean()
                
                # 20日收益率（动量）
                if 'close' in group.columns:
                    group['return_20'] = group['close'].pct_change(20)
                    
                    # 路径效率
                    abs_changes = group['close'].diff().abs().rolling(20).sum()
                    net_change = group['close'].diff(20).abs()
                    group['efficiency_20'] = net_change / abs_changes.replace(0, np.nan)
                    
                    # 夏普比率（简化版）
                    returns = group['close'].pct_change()
                    group['sharpe_20'] = (
                        returns.rolling(20).mean() / returns.rolling(20).std().replace(0, np.nan)
                    ) * np.sqrt(252)
                
                factor_dfs.append(group)
            
            self.factor_data = pd.concat(factor_dfs, ignore_index=True)
            
            # Z-Score 标准化
            zscore_cols = {
                'rsi_20': 'rsi_20_zscore',
                'turnover_5d': 'turnover_5d_zscore',
                'return_20': 'momentum_zscore',
                'sharpe_20': 'sharpe_20_zscore',
                'efficiency_20': 'efficiency_20_zscore',
            }
            
            for src_col, dst_col in zscore_cols.items():
                if src_col in self.factor_data.columns:
                    mean = self.factor_data[src_col].mean()
                    std = self.factor_data[src_col].std()
                    if std > 0:
                        self.factor_data[dst_col] = (self.factor_data[src_col] - mean) / std
            
            # 小市值因子
            if 'circ_mv' in self.factor_data.columns:
                log_mv = np.log(self.factor_data['circ_mv'].replace(0, np.nan))
                mean = log_mv.mean()
                std = log_mv.std()
                if std > 0:
                    self.factor_data['small_cap_zscore'] = -(log_mv - mean) / std
            
            # 保存因子数据
            factor_path = DATA_PROCESSED_PATH / f"factors_{self.today.strftime('%Y%m%d')}.parquet"
            self.factor_data.to_parquet(factor_path)
            
            self.logger.info(f"因子计算完成: {len(self.factor_data)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"因子计算失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def is_rebalance_day(self) -> bool:
        """判断今天是否是调仓日"""
        frequency = self.strategy.rebalance_frequency
        
        if frequency == "weekly":
            return self.today.dayofweek == 4  # 周五
        else:  # monthly
            next_day = self.today + timedelta(days=1)
            return self.today.month != next_day.month
    
    def generate_target_positions(self) -> bool:
        """生成目标持仓"""
        self.logger.info("生成目标持仓...")
        
        try:
            if self.factor_data is None or self.factor_data.empty:
                self.logger.error("因子数据为空")
                return False
            
            # 过滤当日数据
            latest_date = pd.to_datetime(self.factor_data['trade_date']).max()
            day_data = self.factor_data[
                pd.to_datetime(self.factor_data['trade_date']) == latest_date
            ]
            
            if day_data.empty:
                self.logger.error("当日数据为空")
                return False
            
            # 选股
            selected_stocks = self.strategy.select_top_stocks(
                day_data,
                n=self.strategy.top_n,
                date=latest_date
            )
            
            if not selected_stocks:
                self.logger.warning("未选出任何股票")
                self.target_positions = {}
                return True
            
            # 生成等权重持仓
            portfolio_config = self.config.get("portfolio", {})
            total_capital = portfolio_config.get("total_capital", 300000)
            weight = 1.0 / len(selected_stocks)
            
            self.target_positions = {
                stock: total_capital * weight
                for stock in selected_stocks
            }
            
            self.logger.info(
                f"目标持仓生成完成: {len(self.target_positions)} 只股票, "
                f"每只约 ¥{total_capital * weight:,.0f}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成目标持仓失败: {e}")
            return False
    
    def calculate_trade_orders(self) -> tuple:
        """计算交易订单"""
        buy_orders: Dict[str, float] = {}
        sell_orders: Dict[str, float] = {}
        
        # 卖出：当前持有但目标不持有的股票
        for stock, amount in self.current_positions.items():
            if stock not in self.target_positions:
                sell_orders[stock] = amount
            elif self.target_positions[stock] < amount:
                sell_orders[stock] = amount - self.target_positions[stock]
        
        # 买入：目标持有但当前不持有或需要加仓的股票
        for stock, target_amount in self.target_positions.items():
            current_amount = self.current_positions.get(stock, 0)
            if target_amount > current_amount:
                buy_orders[stock] = target_amount - current_amount
        
        return buy_orders, sell_orders
    
    def generate_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        format: str = "markdown"
    ) -> str:
        """生成报告"""
        strategy_info = {
            'name': self.strategy.name,
            'value_weight': self.strategy.value_weight,
            'quality_weight': self.strategy.quality_weight,
            'momentum_weight': self.strategy.momentum_weight,
            'top_n': self.strategy.top_n,
        }
        
        return self.report_generator.generate_report(
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            target_positions=self.target_positions,
            strategy_info=strategy_info,
            report_date=self.today.strftime('%Y-%m-%d'),
            format=format
        )
    
    def save_report(self, report_content: str, format: str = "markdown") -> Path:
        """保存报告"""
        return self.report_generator.save_report(
            report_content,
            self.today.strftime('%Y%m%d'),
            format
        )
    
    def run(self, force_rebalance: bool = False) -> bool:
        """
        执行完整的每日更新流程
        
        Parameters
        ----------
        force_rebalance : bool
            是否强制调仓
        
        Returns
        -------
        bool
            执行是否成功
        """
        self.logger.info("=" * 50)
        self.logger.info(f"开始每日更新任务: {self.today.strftime('%Y-%m-%d')}")
        self.logger.info("=" * 50)
        
        # Step 1: 更新市场数据
        self.logger.info("Step 1/6: 更新市场数据")
        if not self.update_market_data():
            self.logger.error("市场数据更新失败")
            return False
        
        # Step 2: 更新财务数据
        self.logger.info("Step 2/6: 更新财务数据")
        if not self.update_financial_data():
            self.logger.error("财务数据更新失败")
            return False
        
        # Step 3: 更新基准指数
        self.logger.info("Step 3/6: 更新基准指数")
        self.update_benchmark_data()
        
        # Step 4: 计算因子
        self.logger.info("Step 4/6: 计算因子数据")
        if not self.calculate_factors():
            self.logger.error("因子计算失败")
            return False
        
        # Step 5: 判断是否调仓日
        is_rebalance = force_rebalance or self.is_rebalance_day()
        
        if is_rebalance:
            self.logger.info("Step 5/6: 生成目标持仓（调仓日）")
            if not self.generate_target_positions():
                self.logger.error("目标持仓生成失败")
                return False
        else:
            self.logger.info("Step 5/6: 非调仓日，跳过持仓生成")
            self.target_positions = self.current_positions.copy()
        
        # Step 6: 生成报告
        self.logger.info("Step 6/6: 生成交易报告")
        buy_orders, sell_orders = self.calculate_trade_orders()
        
        for fmt in ["markdown", "html"]:
            report_content = self.generate_report(buy_orders, sell_orders, format=fmt)
            self.save_report(report_content, format=fmt)
        
        # 更新持仓
        self.save_current_holdings(buy_orders, sell_orders)
        
        self.logger.info("=" * 50)
        self.logger.info("每日更新任务完成")
        self.logger.info("=" * 50)
        
        return True


def run_daily_update(
    force_rebalance: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    运行每日更新
    
    Parameters
    ----------
    force_rebalance : bool
        是否强制调仓
    config : Optional[Dict[str, Any]]
        配置参数
    
    Returns
    -------
    bool
        执行是否成功
    """
    runner = DailyUpdateRunner(config)
    return runner.run(force_rebalance)

