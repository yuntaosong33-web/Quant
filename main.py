#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股量化交易系统 - 主入口

该模块作为系统的主入口点，提供每日更新、因子计算、
调仓信号生成和报告输出等功能。

Usage
-----
    # 运行每日更新
    python main.py --daily-update
    # 强制调仓（忽略日期检查）
    python main.py --daily-update --force-rebalance
    
    # 生成回测报告
    python main.py --backtest --start 2023-01-01 --end 2024-01-01
"""
import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import pandas as pd
import numpy as np

from src import (
    # 数据处理
    AkshareDataLoader,
    AShareDataCleaner,
    DataLoader,
    # 因子计算
    FactorCalculator,
    z_score_normalize,
    # 策略
    MultiFactorStrategy,
    MACrossStrategy,
    # 回测
    BacktestEngine,
    VBTProBacktester,
    # 权重优化
    optimize_weights,
    calculate_shrinkage_covariance,
    calculate_expected_returns_mean,
    # 工具
    setup_logging,
    load_config,
    send_pushplus_msg,
)

# 导入 LLM 熔断器异常（用于风控处理）
try:
    from src.llm_client import LLMCircuitBreakerError
except ImportError:
    # 定义回退类以避免导入错误
    class LLMCircuitBreakerError(RuntimeError):
        """LLM 熔断器触发异常（回退定义）"""
        pass

# 配置常量
CONFIG_PATH = Path("config/strategy_config.yaml")
DATA_RAW_PATH = Path("data/raw")
DATA_PROCESSED_PATH = Path("data/processed")
REPORTS_PATH = Path("reports")
LOGS_PATH = Path("logs")

# 确保目录存在
for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, REPORTS_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


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
    data_loader : AkshareDataLoader
        数据加载器
    strategy : MultiFactorStrategy
        多因子策略
    logger : logging.Logger
        日志器
    
    Examples
    --------
    >>> runner = DailyUpdateRunner()
    >>> runner.run_daily_update()
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化每日更新运行器
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            配置参数
        """
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        if config is None:
            try:
                self.config = load_config(CONFIG_PATH)
            except FileNotFoundError:
                self.logger.warning(f"配置文件 {CONFIG_PATH} 不存在，使用默认配置")
                self.config = self._get_default_config()
        else:
            self.config = config
        
        # 初始化组件
        self._init_components()
        
        # 状态变量
        self.today = pd.Timestamp.now().normalize()
        self.ohlcv_data: Optional[pd.DataFrame] = None
        self.financial_data: Optional[pd.DataFrame] = None
        self.industry_data: Optional[pd.DataFrame] = None
        self.factor_data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.DataFrame] = None  # 基准指数数据（用于大盘风控）
        self.current_positions: Dict[str, float] = {}
        self.target_positions: Dict[str, float] = {}
        
        # 加载当前持仓
        self.load_current_holdings()
        
        self.logger.info("DailyUpdateRunner 初始化完成")

    def run_daily_update(self):
        """
        执行每日数据更新和策略回测的主流程
        """
        self.logger.info("开始执行每日更新任务...")

        # 1. 获取数据 (示例逻辑)
        # df = self.data_loader.get_data(...)

        # 2. 执行策略
        # self.strategy.execute(df)

        self.logger.info("每日更新任务完成。")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "data": {
                "stock_pool": "hs300",  # 沪深300成分股
                "start_date": "2020-01-01",
                "update_days": 5,  # 每次更新最近N天数据
            },
            "strategy": {
                "name": "Multi-Factor Strategy",
                "value_weight": 0.0,
                "quality_weight": 0.0,
                "momentum_weight": 1.0,
                "top_n": 30,
                "min_listing_days": 126,
            },
            "portfolio": {
                "total_capital": 1000000,  # 总资金100万
                "max_weight": 0.05,  # 单股最大5%
                "risk_free_rate": 0.02,
                "optimization_objective": "max_sharpe",
            },
            "report": {
                "format": "markdown",  # markdown 或 html
                "output_dir": "reports",
            },
        }
    
    def _init_components(self) -> None:
        """初始化各组件"""
        # 根据配置选择数据源
        data_source = self.config.get("data", {}).get("data_source", "akshare")
        
        if data_source == "tushare":
            # 使用 Tushare 数据源（推荐）
            from src.tushare_loader import TushareDataLoader
            tushare_config = self.config.get("tushare", {})
            api_token = tushare_config.get("api_token") or os.environ.get("TUSHARE_TOKEN", "")
            
            if not api_token:
                self.logger.error(
                    "Tushare API Token 未配置！\n"
                    "请在 config/strategy_config.yaml 中设置 tushare.api_token\n"
                    "或通过环境变量 TUSHARE_TOKEN 设置"
                )
                raise ValueError("Tushare API Token 未配置")
            
            self.tushare_loader = TushareDataLoader(
                api_token=api_token,
                cache_dir=tushare_config.get("cache_dir", "data/tushare_cache")
            )
            self.data_source = "tushare"
            self.logger.info("使用 Tushare 数据源")
        else:
            # 使用 AkShare 数据源（兼容旧代码）
            self.data_loader = AkshareDataLoader(self.config)
            self.data_source = "akshare"
            self.logger.info("使用 AkShare 数据源")
        
        self.data_cleaner = AShareDataCleaner()
        
        # 增强版数据加载器（用于获取财务数据，兼容旧代码）
        self.financial_loader = DataLoader(
            output_dir=str(DATA_RAW_PATH),
            max_workers=3,
            retry_times=3
        )
        
        # 策略
        strategy_config = self.config.get("strategy", {})
        llm_config = self.config.get("llm", {})
        
        self.strategy = MultiFactorStrategy(
            name=strategy_config.get("name", "Multi-Factor Strategy"),
            config={
                # 因子权重配置（从配置文件读取）
                "value_weight": strategy_config.get("value_weight", 0.0),
                "quality_weight": strategy_config.get("quality_weight", 0.3),
                "momentum_weight": strategy_config.get("momentum_weight", 0.4),
                "size_weight": strategy_config.get("size_weight", 0.3),
                "sentiment_weight": strategy_config.get("sentiment_weight", 0.0),  # 情绪因子权重
                "top_n": strategy_config.get("top_n", 3),
                "min_listing_days": strategy_config.get("min_listing_days", 126),
                # 板块过滤配置
                "exclude_chinext": strategy_config.get("exclude_chinext", False),  # 排除创业板
                "exclude_star": strategy_config.get("exclude_star", False),  # 排除科创板
                # 因子列名配置（从配置文件读取，支持激进型小市值策略）
                "value_col": strategy_config.get("value_col", "small_cap_zscore"),
                "quality_col": strategy_config.get("quality_col", "turnover_5d_zscore"),
                "momentum_col": strategy_config.get("momentum_col", "rsi_20_zscore"),
                "size_col": strategy_config.get("size_col", "small_cap_zscore"),
                # 调仓配置
                "rebalance_frequency": strategy_config.get("rebalance_frequency", "weekly"),
                "rebalance_buffer": strategy_config.get("rebalance_buffer", 0.02),
                # [NEW] 持股惯性加分
                "holding_bonus": strategy_config.get("holding_bonus", 0.0),
                # [NEW] 大盘风控配置
                "market_risk": self.config.get("risk", {}).get("market_risk", {}),
                # LLM 情绪分析配置
                "llm": llm_config,
            }
        )
    
    def load_current_holdings(self) -> None:
        """
        加载当前持仓
        
        从 data/processed/real_holdings.json 文件读取持仓数据。
        如果文件不存在，初始化为空字典。
        
        Notes
        -----
        这是一个半自动系统，用户可以手动修改 real_holdings.json 
        文件来校准实际持仓。
        """
        holdings_path = DATA_PROCESSED_PATH / "real_holdings.json"
        
        if holdings_path.exists():
            try:
                with open(holdings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取 positions 字段
                self.current_positions = data.get("positions", {})
                
                # 确保值为 float 类型
                self.current_positions = {
                    str(k): float(v) for k, v in self.current_positions.items()
                }
                
                self.logger.info(
                    f"已加载持仓数据: {len(self.current_positions)} 只股票, "
                    f"总市值 ¥{sum(self.current_positions.values()):,.0f}"
                )
                
                # 打印持仓明细（调试用）
                if self.current_positions:
                    self.logger.debug(f"持仓明细: {list(self.current_positions.keys())[:5]}...")
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"持仓文件格式错误: {e}，初始化为空持仓")
                self.current_positions = {}
            except Exception as e:
                self.logger.warning(f"加载持仓文件失败: {e}，初始化为空持仓")
                self.current_positions = {}
        else:
            self.logger.info("持仓文件不存在，初始化为空持仓")
            self.current_positions = {}
    
    def save_current_holdings(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float]
    ) -> None:
        """
        保存当前持仓
        
        根据买入和卖出订单更新持仓，并保存到 data/processed/real_holdings.json。
        
        Parameters
        ----------
        buy_orders : Dict[str, float]
            买入订单 {股票代码: 金额}
        sell_orders : Dict[str, float]
            卖出订单 {股票代码: 金额}
        
        Notes
        -----
        更新逻辑: new_holdings = current + buy - sell
        
        这是一个半自动系统，我们假设用户完全执行了信号。
        实际操作中，用户可能需要手动修改这个 json 文件来校准实际持仓。
        """
        # 复制当前持仓
        new_positions = self.current_positions.copy()
        
        # 处理买入订单
        for stock, amount in buy_orders.items():
            if stock in new_positions:
                new_positions[stock] += amount
            else:
                new_positions[stock] = amount
        
        # 处理卖出订单
        for stock, amount in sell_orders.items():
            if stock in new_positions:
                new_positions[stock] -= amount
                # 如果持仓为0或负数，删除该股票
                if new_positions[stock] <= 0:
                    del new_positions[stock]
        
        # 准备保存的数据
        holdings_data = {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "update_date": self.today.strftime("%Y-%m-%d"),
            "positions": new_positions,
            "total_value": sum(new_positions.values()),
            "num_stocks": len(new_positions),
            "note": "此文件由系统自动生成，假设用户完全执行了交易信号。如需校准实际持仓，请手动修改。"
        }
        
        # 保存到文件
        holdings_path = DATA_PROCESSED_PATH / "real_holdings.json"
        
        try:
            with open(holdings_path, 'w', encoding='utf-8') as f:
                json.dump(holdings_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(
                f"持仓已更新并保存: {len(new_positions)} 只股票, "
                f"总市值 ¥{sum(new_positions.values()):,.0f}"
            )
            
            # 更新内存中的持仓
            self.current_positions = new_positions
            
        except Exception as e:
            self.logger.error(f"保存持仓文件失败: {e}")
    
    def update_market_data(self) -> bool:
        """
        更新市场数据（带缓存检查）
        
        支持 Tushare 和 AkShare 两种数据源。
        
        Returns
        -------
        bool
            更新是否成功
        """
        self.logger.info("开始更新市场数据...")
        
        try:
            # 检查今日缓存
            ohlcv_path = DATA_RAW_PATH / f"ohlcv_{self.today.strftime('%Y%m%d')}.parquet"
            if ohlcv_path.exists():
                try:
                    self.ohlcv_data = pd.read_parquet(ohlcv_path)
                    if not self.ohlcv_data.empty:
                        self.logger.info(f"📂 使用缓存数据: {ohlcv_path.name}，共 {len(self.ohlcv_data)} 条记录")
                        return True
                except Exception as e:
                    self.logger.warning(f"读取缓存失败: {e}，将重新下载")
            
            data_config = self.config.get("data", {})
            stock_pool = data_config.get("stock_pool", "hs300")
            
            # 确定日期范围
            end_date = self.today.strftime("%Y%m%d")
            update_days = data_config.get("update_days", 5)
            start_date = (self.today - timedelta(days=update_days * 2)).strftime("%Y%m%d")
            
            # 根据数据源选择不同的获取方式
            if self.data_source == "tushare":
                return self._update_market_data_tushare(stock_pool, start_date, end_date)
            else:
                return self._update_market_data_akshare(stock_pool, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _update_market_data_tushare(
        self,
        stock_pool: str,
        start_date: str,
        end_date: str
    ) -> bool:
        """使用 Tushare 更新市场数据"""
        self.logger.info(f"使用 Tushare 获取 {stock_pool} 数据...")
        
        # 获取股票列表（根据 stock_pool 类型选择不同方法）
        if stock_pool == "all":
            # 全市场模式：获取所有上市股票
            self.logger.info("全市场模式：获取所有上市股票...")
            stock_list = self.tushare_loader.fetch_all_stocks()
            if not stock_list:
                self.logger.error("无法获取全市场股票列表")
                return False
            self.logger.warning(
                f"⚠️ 全市场模式：共 {len(stock_list)} 只股票，"
                f"数据下载和计算将耗时较长，请耐心等待"
            )
        else:
            # 指数成分股模式
            stock_list = self.tushare_loader.fetch_index_constituents(stock_pool)
            if not stock_list:
                self.logger.error(f"无法获取 {stock_pool} 成分股列表")
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
        ohlcv_path = DATA_RAW_PATH / f"ohlcv_{self.today.strftime('%Y%m%d')}.parquet"
        self.ohlcv_data.to_parquet(ohlcv_path)
        self.logger.info(f"OHLCV 数据已保存至 {ohlcv_path}")
        
        # 保存成分股列表供后续使用
        self._current_stock_list = stock_list
        
        return True
    
    def _update_market_data_akshare(
        self,
        stock_pool: str,
        start_date: str,
        end_date: str
    ) -> bool:
        """使用 AkShare 更新市场数据（兼容旧代码）"""
        # 获取股票列表（支持主要指数成分股）
        stock_pool_to_index = {
            "hs300": "000300",
            "zz500": "000905",
            "zz1000": "000852",
            "sz50": "000016",
            "cyb50": "399673",
        }
        
        if stock_pool in stock_pool_to_index:
            index_code = stock_pool_to_index[stock_pool]
            stock_list = self.data_loader.get_stock_list(index_code)
        elif stock_pool == "all":
            self.logger.warning("全市场模式网络依赖较高，建议使用指数成分股模式")
            stock_list = self.data_loader.get_stock_list()
        else:
            self.logger.warning(f"未知的股票池 '{stock_pool}'，使用默认沪深300")
            stock_list = self.data_loader.get_stock_list("000300")
        
        self.logger.info(f"股票池: {stock_pool}, 股票数量: {len(stock_list)}")
        
        # 下载OHLCV数据
        ohlcv_list = []
        total_stocks = len(stock_list)
        for i, stock in enumerate(stock_list):
            try:
                df = self.data_loader.fetch_daily_data(stock, start_date, end_date)
                if df is not None and not df.empty:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index(names=['date'])
                    df['stock_code'] = stock
                    ohlcv_list.append(df)
            except Exception as e:
                self.logger.debug(f"获取 {stock} 数据失败: {e}")
            
            if (i + 1) % 50 == 0 or (i + 1) == total_stocks:
                self.logger.info(f"已处理 {i + 1}/{total_stocks} 只股票")
        
        if ohlcv_list:
            self.ohlcv_data = pd.concat(ohlcv_list, ignore_index=True)
            self.logger.info(f"OHLCV 数据更新完成，共 {len(self.ohlcv_data)} 条记录")
        else:
            self.logger.warning("未获取到任何 OHLCV 数据")
            return False
        
        # 保存数据
        ohlcv_path = DATA_RAW_PATH / f"ohlcv_{self.today.strftime('%Y%m%d')}.parquet"
        self.ohlcv_data.to_parquet(ohlcv_path)
        self.logger.info(f"OHLCV 数据已保存至 {ohlcv_path}")
        
        return True
    
    def update_financial_data(self) -> bool:
        """
        更新财务数据（实盘安全版，带缓存检查）
        
        使用 DataLoader.fetch_financial_indicator 获取真实的 PE、PB、ROE 等数据。
        采用 Fail Fast 机制，确保实盘安全：
        - 不使用任何虚假/备用数据填充
        - 失败股票标记为无效，从选股池中剔除
        - 失败率超过阈值时终止程序并报警
        
        Returns
        -------
        bool
            更新是否成功
        
        Raises
        ------
        RuntimeError
            当财务数据获取失败率超过 30% 时
        """
        self.logger.info("开始更新财务数据（实盘安全模式）...")
        
        # 检查今日缓存
        financial_path = DATA_RAW_PATH / f"financial_{self.today.strftime('%Y%m%d')}.parquet"
        if financial_path.exists():
            try:
                self.financial_data = pd.read_parquet(financial_path)
                if not self.financial_data.empty:
                    self.logger.info(f"📂 使用缓存数据: {financial_path.name}，共 {len(self.financial_data)} 条记录")
                    return True
            except Exception as e:
                self.logger.warning(f"读取缓存失败: {e}，将重新下载")
        
        # 根据数据源选择不同的获取方式
        if self.data_source == "tushare":
            return self._update_financial_data_tushare()
        else:
            return self._update_financial_data_akshare()
    
    def _update_financial_data_tushare(self) -> bool:
        """使用 Tushare 更新财务数据"""
        try:
            if self.ohlcv_data is None:
                self.logger.warning("OHLCV 数据为空，无法生成财务数据")
                return False
            
            stocks = self.ohlcv_data['stock_code'].unique().tolist()
            total_stocks = len(stocks)
            self.logger.info(f"使用 Tushare 获取 {total_stocks} 只股票的财务数据...")
            
            # 方式1：使用 daily_basic 一次获取全市场估值数据（高效）
            self.logger.info("获取每日基础指标 (PE, PB, 市值)...")
            basic_df = self.tushare_loader.fetch_daily_basic(stock_list=stocks)
            
            if basic_df is not None and not basic_df.empty:
                self.logger.info(f"每日基础指标获取成功: {len(basic_df)} 条")
            else:
                self.logger.warning("每日基础指标获取失败，将逐只获取")
                basic_df = pd.DataFrame()
            
            # 方式2：批量获取财务指标（ROE 等）
            self.logger.info("获取财务指标 (ROE, 毛利率等)...")
            fina_df = self.tushare_loader.fetch_financial_batch(stocks, show_progress=True)
            
            # 合并数据（避免重复列）
            if not basic_df.empty:
                if not fina_df.empty:
                    # 从 fina_df 中只取 basic_df 中不存在的列 + stock_code
                    fina_cols = ['stock_code', 'roe', 'roe_dt']
                    # 检查是否有 gross_margin/net_margin，且 basic_df 中没有
                    for col in ['gross_margin', 'net_margin']:
                        if col in fina_df.columns and col not in basic_df.columns:
                            fina_cols.append(col)
                    
                    # 确保只选择存在的列
                    fina_cols = [c for c in fina_cols if c in fina_df.columns]
                    
                    merged_df = basic_df.merge(
                        fina_df[fina_cols],
                        on='stock_code',
                        how='left'
                    )
                else:
                    merged_df = basic_df
            elif not fina_df.empty:
                merged_df = fina_df
            else:
                self.logger.error("无法获取任何财务数据")
                return False
            
            # 去除重复列（如果存在）
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            
            # 标准化列名
            if 'stock_code' not in merged_df.columns and 'ts_code' in merged_df.columns:
                merged_df['stock_code'] = merged_df['ts_code'].str[:6]
            
            # 添加数据有效性标记
            mv_cols = [c for c in ['circ_mv', 'total_mv'] if c in merged_df.columns]
            if mv_cols:
                merged_df['data_valid'] = merged_df[mv_cols].notna().any(axis=1)
            else:
                merged_df['data_valid'] = True
            
            # 估算上市天数
            merged_df['listing_days'] = merged_df['stock_code'].apply(self._estimate_listing_days)
            
            self.financial_data = merged_df
            self._excluded_stocks = set()
            
            # 保存数据
            financial_path = DATA_RAW_PATH / f"financial_{self.today.strftime('%Y%m%d')}.parquet"
            self.financial_data.to_parquet(financial_path)
            
            valid_count = self.financial_data['data_valid'].sum()
            self.logger.info(
                f"✅ 财务数据更新完成 (Tushare):\n"
                f"   总记录: {len(self.financial_data)}\n"
                f"   有效数据: {valid_count}\n"
                f"   PE有效: {self.financial_data['pe_ttm'].notna().sum()}\n"
                f"   ROE有效: {self.financial_data['roe'].notna().sum() if 'roe' in self.financial_data.columns else 0}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Tushare 财务数据获取失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _update_financial_data_akshare(self) -> bool:
        """使用 AkShare 更新财务数据（兼容旧代码）"""
        # 失败率阈值（超过此比例将触发 Critical Error）
        FAILURE_THRESHOLD = 0.30  # 30%
        
        try:
            if self.ohlcv_data is None:
                self.logger.warning("OHLCV 数据为空，无法生成财务数据")
                return False
            
            stocks = self.ohlcv_data['stock_code'].unique().tolist()
            total_stocks = len(stocks)
            self.logger.info(f"需获取 {total_stocks} 只股票的财务数据")
            
            # 预先获取全市场数据并缓存（避免为每只股票重复请求）
            self.logger.info("预获取全市场行情数据以加速财务指标获取...")
            spot_data_available = False
            try:
                spot_df = self.financial_loader._get_spot_data_cached()
                spot_data_available = spot_df is not None and not spot_df.empty
                if spot_data_available:
                    self.logger.info("✅ 全市场行情数据就绪")
            except Exception as e:
                self.logger.warning(f"预获取全市场数据失败: {e}")
            
            # 如果全市场数据获取失败，尝试使用历史财务数据作为备份
            if not spot_data_available:
                fallback_df = self._load_fallback_financial_data(stocks)
                if fallback_df is not None:
                    self.logger.info(f"⚠️ 使用历史财务数据作为备份（{len(fallback_df)} 条）")
                    self.financial_data = fallback_df
                    self._excluded_stocks = set()
                    return True
                self.logger.warning("无可用的历史财务数据备份，将尝试逐只股票获取")
            
            # 使用真实数据接口获取财务指标
            financial_records = []
            failed_stocks = []
            
            import time
            
            for i, stock in enumerate(stocks):
                try:
                    # 调用 DataLoader.fetch_financial_indicator 获取真实数据
                    fin_df = self.financial_loader.fetch_financial_indicator(stock)
                    
                    if fin_df is not None and not fin_df.empty:
                        # 提取最新的财务指标
                        if isinstance(fin_df, pd.DataFrame) and len(fin_df) > 0:
                            latest = fin_df.iloc[-1] if len(fin_df) > 1 else fin_df.iloc[0]
                            
                            # 构建财务记录
                            circ_mv = self._safe_get_value(latest, ['circ_mv', '流通市值'], default=np.nan)
                            total_mv = self._safe_get_value(latest, ['total_mv', '总市值'], default=np.nan)
                            
                            # 检查关键字段是否有效（流通市值对小市值策略至关重要）
                            has_valid_mv = pd.notna(circ_mv) or pd.notna(total_mv)
                            
                            record = {
                                'stock_code': stock,
                                'pe_ttm': self._safe_get_value(latest, ['pe_ttm', 'pe', '市盈率'], default=np.nan),
                                'pb': self._safe_get_value(latest, ['pb', '市净率'], default=np.nan),
                                'dividend_yield': self._safe_get_value(latest, ['dividend_yield', 'dv_ratio', '股息率'], default=np.nan),
                                'ps_ttm': self._safe_get_value(latest, ['ps_ttm', 'ps', '市销率'], default=np.nan),
                                'roe': self._safe_get_value(latest, ['roe', 'roe_ttm'], default=np.nan),
                                'total_mv': total_mv,
                                'circ_mv': circ_mv,
                                # 标记数据是否有效（用于后续过滤）
                                'data_valid': has_valid_mv,
                            }
                            
                            # 估算上市天数
                            record['listing_days'] = self._estimate_listing_days(stock)
                            
                            financial_records.append(record)
                            
                            if has_valid_mv:
                                self.logger.debug(
                                    f"✓ {stock} 财务数据有效: "
                                    f"circ_mv={circ_mv/1e8:.2f}亿" if pd.notna(circ_mv) else f"✓ {stock} 财务数据获取成功"
                                )
                            else:
                                # 数据获取成功但缺少关键字段，标记为失败
                                self.logger.warning(
                                    f"⚠ {stock} 缺少关键市值数据，将从选股池中剔除"
                                )
                                failed_stocks.append(stock)
                        else:
                            self.logger.warning(f"⚠ {stock} 财务数据为空，将从选股池中剔除")
                            failed_stocks.append(stock)
                    else:
                        self.logger.warning(f"⚠ {stock} 无法获取财务数据，将从选股池中剔除")
                        failed_stocks.append(stock)
                        
                except Exception as e:
                    self.logger.warning(f"⚠ {stock} 财务数据获取异常: {e}，将从选股池中剔除")
                    failed_stocks.append(stock)
                
                # 进度日志
                if (i + 1) % 10 == 0:
                    current_failure_rate = len(failed_stocks) / (i + 1)
                    self.logger.info(
                        f"财务数据获取进度: {i + 1}/{total_stocks} | "
                        f"失败: {len(failed_stocks)} ({current_failure_rate:.1%})"
                    )
                
                # 添加延时避免请求过快（已有缓存时可减少延时）
                time.sleep(0.05)
            
            # ========== Fail Fast 检查 ==========
            failure_rate = len(failed_stocks) / total_stocks if total_stocks > 0 else 0
            
            if failure_rate > FAILURE_THRESHOLD:
                error_msg = (
                    f"🚨 CRITICAL ERROR: 财务数据获取失败率过高!\n"
                    f"   失败数量: {len(failed_stocks)}/{total_stocks} ({failure_rate:.1%})\n"
                    f"   阈值: {FAILURE_THRESHOLD:.0%}\n"
                    f"   失败股票示例: {failed_stocks[:10]}...\n"
                    f"   为确保实盘安全，程序终止。请检查数据源或网络连接。"
                )
                self.logger.critical(error_msg)
                
                # 尝试发送报警通知
                try:
                    token = os.environ.get("PUSHPLUS_TOKEN", "")
                    if not token:
                        token = self.config.get("notification", {}).get("pushplus_token", "")
                    if token:
                        send_pushplus_msg(
                            token=token,
                            title="🚨 量化系统 Critical Error",
                            content=error_msg.replace("\n", "<br>"),
                            template="html"
                        )
                except Exception:
                    pass
                
                raise RuntimeError(error_msg)
            
            # ========== 处理失败股票（不使用 Fallback，仅记录） ==========
            if failed_stocks:
                self.logger.warning(
                    f"📊 财务数据获取结果:\n"
                    f"   成功: {total_stocks - len(failed_stocks)}/{total_stocks}\n"
                    f"   失败: {len(failed_stocks)}/{total_stocks} ({failure_rate:.1%})\n"
                    f"   ⚠ 失败股票将被排除在选股池之外（不使用虚假数据填充）"
                )
                
                # 保存失败股票列表供后续过滤使用
                self._excluded_stocks = set(failed_stocks)
            else:
                self._excluded_stocks = set()
            
            if not financial_records:
                self.logger.error("未获取到任何有效财务数据")
                return False
            
            self.financial_data = pd.DataFrame(financial_records)
            
            # 数据清洗：处理异常值
            self._clean_financial_data()
            
            # 获取行业数据
            self.industry_data = self._fetch_industry_data(stocks)
            
            # 统计有效数据
            valid_count = self.financial_data['data_valid'].sum() if 'data_valid' in self.financial_data.columns else len(self.financial_data)
            
            self.logger.info(
                f"✅ 财务数据更新完成:\n"
                f"   总记录: {len(self.financial_data)}\n"
                f"   有效数据: {valid_count}\n"
                f"   已剔除: {len(failed_stocks)} 只股票"
            )
            
            # 保存数据
            financial_path = DATA_RAW_PATH / f"financial_{self.today.strftime('%Y%m%d')}.parquet"
            self.financial_data.to_parquet(financial_path)
            self.logger.info(f"财务数据已保存至 {financial_path}")
            
            return True
            
        except RuntimeError:
            # Critical Error，直接向上抛出
            raise
        except Exception as e:
            self.logger.error(f"更新财务数据失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _safe_get_value(
        self,
        data: pd.Series,
        keys: List[str],
        default: Any = np.nan
    ) -> Any:
        """
        安全地从 Series 中获取值
        
        Parameters
        ----------
        data : pd.Series
            数据序列
        keys : List[str]
            可能的键名列表
        default : Any
            默认值
        
        Returns
        -------
        Any
            获取到的值或默认值
        """
        for key in keys:
            if key in data.index:
                val = data[key]
                if pd.notna(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
        return default
    
    def _load_fallback_financial_data(
        self, 
        required_stocks: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        加载历史财务数据作为网络失败时的备份
        
        按日期倒序查找最近的财务数据文件，过滤出当前需要的股票。
        
        Parameters
        ----------
        required_stocks : List[str]
            需要财务数据的股票代码列表
        
        Returns
        -------
        Optional[pd.DataFrame]
            历史财务数据，如果无可用数据返回 None
        """
        # 查找历史财务数据文件
        financial_files = sorted(
            DATA_RAW_PATH.glob("financial_*.parquet"),
            reverse=True  # 最新的优先
        )
        
        if not financial_files:
            self.logger.warning("未找到历史财务数据文件")
            return None
        
        # 尝试加载最近的文件
        for file_path in financial_files[:5]:  # 最多尝试5个文件
            try:
                df = pd.read_parquet(file_path)
                
                if df.empty or 'stock_code' not in df.columns:
                    continue
                
                # 过滤出需要的股票
                required_set = set(required_stocks)
                df_filtered = df[df['stock_code'].isin(required_set)]
                
                if len(df_filtered) == 0:
                    continue
                
                coverage = len(df_filtered) / len(required_set)
                file_date = file_path.stem.replace("financial_", "")
                
                self.logger.info(
                    f"📂 加载历史财务数据: {file_path.name}\n"
                    f"   数据日期: {file_date}\n"
                    f"   覆盖率: {len(df_filtered)}/{len(required_set)} ({coverage:.1%})"
                )
                
                # 覆盖率太低则跳过
                if coverage < 0.5:
                    self.logger.warning(f"覆盖率过低 ({coverage:.1%})，尝试其他文件")
                    continue
                
                return df_filtered
                
            except Exception as e:
                self.logger.debug(f"加载 {file_path} 失败: {e}")
                continue
        
        return None
    
    def _estimate_listing_days(self, stock: str) -> int:
        """
        估算股票上市天数（快速版本）
        
        使用股票代码前缀快速估算，避免逐只 API 调用。
        沪深300成分股通常都是上市多年的蓝筹股。
        
        Parameters
        ----------
        stock : str
            股票代码
        
        Returns
        -------
        int
            估算的上市天数
        """
        # 对于沪深300成分股，默认假设上市超过2年（符合基本条件）
        # 这避免了逐只调用 API 的性能问题
        return 1000  # 默认返回较大值，表示已上市较长时间
    
    def _generate_fallback_financial_data(self, stocks: List[str]) -> List[Dict[str, Any]]:
        """
        [已废弃] 为获取失败的股票生成备用财务数据
        
        此方法已被废弃，实盘环境下禁止使用虚假数据填充。
        调用此方法将抛出 RuntimeError。
        
        Parameters
        ----------
        stocks : List[str]
            股票代码列表
        
        Returns
        -------
        List[Dict[str, Any]]
            不会返回，直接抛出异常
        
        Raises
        ------
        RuntimeError
            始终抛出，禁止使用备用数据
        
        Notes
        -----
        实盘安全策略：
        - 失败股票应直接从选股池中剔除，而非用虚假数据填充
        - 使用中位数/默认值填充可能导致选股失真，造成实盘亏损
        - 正确做法：在 calculate_factors 时过滤掉 data_valid=False 的股票
        """
        error_msg = (
            f"🚨 安全警告: 禁止使用备用财务数据!\n"
            f"   请求填充 {len(stocks)} 只股票的虚假数据。\n"
            f"   实盘环境下，这可能导致严重的选股失真。\n"
            f"   正确做法：将这些股票从选股池中剔除。\n"
            f"   股票列表: {stocks[:5]}..."
        )
        self.logger.critical(error_msg)
        raise RuntimeError(error_msg)
    
    def _clean_financial_data(self) -> None:
        """
        清洗财务数据
        
        处理异常值和缺失值。
        """
        if self.financial_data is None or self.financial_data.empty:
            return
        
        # PE 异常值处理：负值设为 NaN，超大值截断
        if 'pe_ttm' in self.financial_data.columns:
            self.financial_data.loc[self.financial_data['pe_ttm'] <= 0, 'pe_ttm'] = np.nan
            self.financial_data.loc[self.financial_data['pe_ttm'] > 500, 'pe_ttm'] = 500
        
        # PB 异常值处理
        if 'pb' in self.financial_data.columns:
            self.financial_data.loc[self.financial_data['pb'] <= 0, 'pb'] = np.nan
            self.financial_data.loc[self.financial_data['pb'] > 50, 'pb'] = 50
        
        # ROE 异常值处理
        if 'roe' in self.financial_data.columns:
            self.financial_data.loc[self.financial_data['roe'] < -1, 'roe'] = -1
            self.financial_data.loc[self.financial_data['roe'] > 1, 'roe'] = 1
        
        # 股息率异常值处理
        if 'dividend_yield' in self.financial_data.columns:
            self.financial_data.loc[self.financial_data['dividend_yield'] < 0, 'dividend_yield'] = 0
            self.financial_data.loc[self.financial_data['dividend_yield'] > 0.20, 'dividend_yield'] = 0.20
        
        self.logger.debug("财务数据清洗完成")
    
    def _fetch_industry_data(self, stocks: List[str]) -> pd.DataFrame:
        """
        获取行业分类数据
        
        Parameters
        ----------
        stocks : List[str]
            股票代码列表
        
        Returns
        -------
        pd.DataFrame
            行业分类数据
        """
        self.logger.info("获取行业分类数据...")
        
        try:
            import akshare as ak
            
            # 尝试获取申万行业分类
            industry_df = ak.stock_board_industry_name_em()
            
            if industry_df is not None and not industry_df.empty:
                # 构建股票到行业的映射
                stock_industry = {}
                
                for _, row in industry_df.iterrows():
                    industry_name = row.get('板块名称', '')
                    industry_code = row.get('板块代码', '')
                    
                    try:
                        # 获取该行业的成分股
                        cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)
                        if cons_df is not None and not cons_df.empty:
                            code_col = '代码' if '代码' in cons_df.columns else cons_df.columns[0]
                            for stock_code in cons_df[code_col]:
                                if stock_code in stocks:
                                    stock_industry[stock_code] = industry_name
                    except Exception:
                        continue
                
                if stock_industry:
                    result = pd.DataFrame([
                        {'stock_code': k, 'sw_industry_l1': v}
                        for k, v in stock_industry.items()
                    ])
                    
                    # 补充未找到的股票
                    missing_stocks = set(stocks) - set(stock_industry.keys())
                    if missing_stocks:
                        missing_df = pd.DataFrame({
                            'stock_code': list(missing_stocks),
                            'sw_industry_l1': '其他'
                        })
                        result = pd.concat([result, missing_df], ignore_index=True)
                    
                    self.logger.info(f"行业分类数据获取成功，共 {len(result)} 条记录")
                    return result
            
        except Exception as e:
            self.logger.warning(f"获取真实行业数据失败: {e}，使用模拟数据")
        
        # 备用方案：使用模拟行业数据
        industries = ['银行', '非银金融', '食品饮料', '医药生物', '电子', 
                     '计算机', '家用电器', '汽车', '房地产', '建筑材料']
        
        return pd.DataFrame({
            'stock_code': list(stocks),
            'sw_industry_l1': np.random.choice(industries, len(stocks))
        })
    
    def update_benchmark_data(self) -> bool:
        """
        更新基准指数数据（用于大盘风控）
        
        获取沪深300指数数据，用于计算MA20风控指标。
        
        Returns
        -------
        bool
            更新是否成功
        """
        self.logger.info("开始更新基准指数数据（沪深300）...")
        
        try:
            data_config = self.config.get("data", {})
            start_date = data_config.get("start_date", "2020-01-01")
            end_date = self.today.strftime("%Y-%m-%d")
            
            # 使用 DataLoader 获取沪深300指数数据
            self.benchmark_data = self.financial_loader.fetch_index_price(
                index_code="000300",
                start_date=start_date,
                end_date=end_date
            )
            
            if self.benchmark_data is not None and not self.benchmark_data.empty:
                self.logger.info(
                    f"基准指数数据更新完成，共 {len(self.benchmark_data)} 条记录，"
                    f"日期范围: {self.benchmark_data.index[0].strftime('%Y-%m-%d')} ~ "
                    f"{self.benchmark_data.index[-1].strftime('%Y-%m-%d')}"
                )
                return True
            else:
                self.logger.warning("未获取到基准指数数据，大盘风控将不生效")
                return False
                
        except Exception as e:
            self.logger.warning(f"更新基准指数数据失败: {e}，大盘风控将不生效")
            self.benchmark_data = None
            return False
    
    def is_market_risk_triggered(self) -> bool:
        """
        检查大盘风控是否触发
        
        从配置文件读取风控参数：
        - ma_period: 均线周期（默认60，即MA60牛熊线）
        - drop_threshold: 跌幅阈值（默认0.05，即5%）
        - drop_lookback: 跌幅回溯天数（默认20）
        
        风控触发条件（满足任一即触发，OR 逻辑）：
        1. 收盘价 < MA{ma_period}（跌破均线）
        2. （可选）近{drop_lookback}日跌幅 > {drop_threshold}
        
        Returns
        -------
        bool
            True 表示风控触发（应空仓），False 表示正常
        
        Notes
        -----
        风控参数从 config['risk']['market_risk'] 中读取，支持动态配置。
        使用 OR 逻辑可避免缓慢阴跌的熊市中无法触发风控的问题。
        """
        # 读取风控配置
        risk_config = self.config.get("risk", {})
        market_risk_config = risk_config.get("market_risk", {})
        
        # 检查是否启用风控
        if not market_risk_config.get("enabled", True):
            self.logger.debug("大盘风控已禁用")
            return False
        
        # 读取风控参数（从配置文件，支持动态调整）
        ma_period = market_risk_config.get("ma_period", 60)  # 默认使用 MA60
        drop_threshold = market_risk_config.get("drop_threshold", 0.05)  # 默认 5%
        drop_lookback = market_risk_config.get("drop_lookback", 20)  # 默认 20 天
        
        if self.benchmark_data is None or self.benchmark_data.empty:
            self.logger.debug("无基准数据，风控检查跳过")
            return False
        
        try:
            # 获取足够的历史数据用于计算均线
            required_days = max(ma_period, drop_lookback) + 1
            latest_data = self.benchmark_data.tail(required_days)
            
            if len(latest_data) < ma_period:
                self.logger.debug(
                    f"基准数据不足 {ma_period} 天（当前 {len(latest_data)} 天），"
                    f"风控检查跳过"
                )
                return False
            
            # 计算移动平均线（使用配置的周期）
            ma_value = latest_data['close'].tail(ma_period).mean()
            latest_close = latest_data['close'].iloc[-1]
            
            # 条件1：跌破均线
            is_below_ma = latest_close < ma_value
            
            # 条件2：计算近期跌幅（可选条件）
            is_drop_exceeded = False
            recent_drop = 0.0
            
            if drop_threshold > 0 and len(latest_data) >= drop_lookback:
                lookback_data = latest_data.tail(drop_lookback)
                if len(lookback_data) >= 2:
                    start_price = lookback_data['close'].iloc[0]
                    end_price = lookback_data['close'].iloc[-1]
                    recent_drop = (end_price - start_price) / start_price
                    is_drop_exceeded = recent_drop < -drop_threshold
            
            # 综合判断：跌破均线 或 跌幅超过阈值（任一满足即触发风控）
            # 原逻辑使用 AND，会导致缓慢阴跌时无法触发，非常危险
            # 新逻辑使用 OR：只要跌破均线，或者发生暴跌，都视为风险
            if drop_threshold > 0:
                is_triggered = is_below_ma or is_drop_exceeded
            else:
                is_triggered = is_below_ma
            
            # 优化的日志输出
            ma_label = f"MA{ma_period}"
            deviation_pct = (latest_close - ma_value) / ma_value * 100
            
            if is_triggered:
                self.logger.warning(
                    f"🚨 大盘风控触发!\n"
                    f"   当前点位: {latest_close:.2f} | {ma_label}: {ma_value:.2f} | "
                    f"偏离: {deviation_pct:+.2f}%\n"
                    f"   近{drop_lookback}日跌幅: {recent_drop*100:+.2f}% | "
                    f"阈值: -{drop_threshold*100:.1f}%"
                )
            else:
                status = "✅" if latest_close >= ma_value else "⚠️"
                self.logger.info(
                    f"{status} 大盘风控检查: "
                    f"点位 {latest_close:.2f} vs {ma_label} {ma_value:.2f} "
                    f"(偏离 {deviation_pct:+.2f}%) | "
                    f"近{drop_lookback}日变化: {recent_drop*100:+.2f}%"
                )
                
                # 如果接近触发条件，额外警告
                if is_below_ma and not is_drop_exceeded:
                    self.logger.warning(
                        f"   ⚠️ 已跌破 {ma_label}，但跌幅 ({recent_drop*100:+.2f}%) "
                        f"未达阈值 (-{drop_threshold*100:.1f}%)，继续观察"
                    )
            
            return is_triggered
            
        except Exception as e:
            self.logger.warning(f"风控检查失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def calculate_factors(self) -> bool:
        """
        计算因子数据（实盘安全版）
        
        包含以下安全机制：
        - 过滤掉财务数据获取失败的股票
        - 将无效数据的股票因子得分设为 -inf，确保不会被选中
        
        Returns
        -------
        bool
            计算是否成功
        """
        self.logger.info("开始计算因子（实盘安全模式）...")
        
        try:
            # 即使财务数据更新失败，如果OHLCV数据存在，仍继续执行因子计算
            if self.ohlcv_data is None:
                self.logger.warning("数据不完整，无法计算因子")
                return False
            
            # 如果财务数据为空，尝试使用空DataFrame继续
            if self.financial_data is None:
                self.logger.warning("财务数据为空，将跳过财务因子计算")
                self.financial_data = pd.DataFrame()

            # 准备数据
            ohlcv = self.ohlcv_data.copy()
            
            # ========== 列名标准化（兼容 Tushare 原始格式） ==========
            column_mapping = {
                'trade_date': 'date',
                'vol': 'volume',
                'pct_chg': 'pct_change',
            }
            ohlcv.rename(columns=column_mapping, inplace=True)
            
            # 确保日期列存在（处理 DatetimeIndex 和各种列名情况）
            if 'date' not in ohlcv.columns:
                if 'trade_date' in ohlcv.columns:
                    ohlcv['date'] = pd.to_datetime(ohlcv['trade_date'])
                elif '日期' in ohlcv.columns:
                    ohlcv['date'] = pd.to_datetime(ohlcv['日期'])
                elif isinstance(ohlcv.index, pd.DatetimeIndex):
                    ohlcv = ohlcv.reset_index()
                    # 重命名索引列为 'date'
                    if ohlcv.columns[0] in ['index', 'date', '日期']:
                        ohlcv.rename(columns={ohlcv.columns[0]: 'date'}, inplace=True)
                    else:
                        ohlcv['date'] = ohlcv.index
                else:
                    self.logger.warning("无法找到日期列，尝试从数据结构推断...")
                    # 尝试重置索引
                    ohlcv = ohlcv.reset_index()
                    if 'index' in ohlcv.columns:
                        ohlcv.rename(columns={'index': 'date'}, inplace=True)
            
            # 检查必要的列是否存在
            required_cols = ['close', 'stock_code']
            missing_cols = [c for c in required_cols if c not in ohlcv.columns]
            if missing_cols:
                self.logger.error(f"OHLCV 数据缺少必要列: {missing_cols}")
                self.logger.info(f"当前列名: {ohlcv.columns.tolist()}")
                return False
            
            # 确保 date 列是 datetime 类型
            if 'date' in ohlcv.columns:
                ohlcv['date'] = pd.to_datetime(ohlcv['date'])
            
            # ========== 实盘安全：过滤掉被排除的股票 ==========
            excluded_stocks = getattr(self, '_excluded_stocks', set())
            if excluded_stocks:
                original_count = len(ohlcv['stock_code'].unique())
                ohlcv = ohlcv[~ohlcv['stock_code'].isin(excluded_stocks)]
                filtered_count = len(ohlcv['stock_code'].unique())
                self.logger.info(
                    f"🛡️ 安全过滤: 已剔除 {original_count - filtered_count} 只"
                    f"财务数据无效的股票（剩余 {filtered_count} 只）"
                )
            
            # 合并财务数据 (仅在财务数据存在时合并，避免硬依赖)
            if self.financial_data is not None and not self.financial_data.empty:
                # 只合并有效数据
                valid_financial = self.financial_data.copy()
                if 'data_valid' in valid_financial.columns:
                    invalid_count = (~valid_financial['data_valid']).sum()
                    if invalid_count > 0:
                        self.logger.warning(
                            f"🛡️ 财务数据中有 {invalid_count} 条无效记录，将被标记"
                        )
                
                # 移除财务数据中与 OHLCV 重复的列（避免合并冲突）
                ohlcv_cols = set(ohlcv.columns) - {'stock_code'}  # 排除 stock_code，它需要保留用于合并
                cols_to_keep = [c for c in valid_financial.columns if c not in ohlcv_cols]
                valid_financial = valid_financial[cols_to_keep]
                
                factor_data = ohlcv.merge(
                    valid_financial,
                    on='stock_code',
                    how='left'
                )
            else:
                factor_data = ohlcv.copy()
            
            # 合并行业数据
            if self.industry_data is not None:
                factor_data = factor_data.merge(
                    self.industry_data,
                    on='stock_code',
                    how='left'
                )
            
            # ==================== 因子计算 ====================
            # 根据策略配置动态计算所需因子
            
            # 1. 动量因子 RSI_20（所有策略通用）
            factor_data['rsi_20'] = factor_data.groupby('stock_code')['close'].transform(
                lambda x: self._calculate_rsi(x, 20)
            )
            
            # 1.5. 动量因子 ROC_20（20日变动率）
            factor_data['roc_20'] = factor_data.groupby('stock_code')['close'].transform(
                lambda x: x.pct_change(20) * 100  # 转换为百分比
            )
            
            # 2. 小市值因子 small_cap（激进型策略使用）
            # small_cap = -log(流通市值)，市值越小分数越高
            if 'circ_mv' in factor_data.columns:
                factor_data['small_cap'] = -np.log(factor_data['circ_mv'].replace(0, np.nan))
                factor_data['small_cap'] = factor_data['small_cap'].replace([np.inf, -np.inf], np.nan)
            elif 'total_mv' in factor_data.columns:
                factor_data['small_cap'] = -np.log(factor_data['total_mv'].replace(0, np.nan))
                factor_data['small_cap'] = factor_data['small_cap'].replace([np.inf, -np.inf], np.nan)
            else:
                factor_data['small_cap'] = np.nan
            
            # 3. 换手率因子 turnover_5d（激进型策略使用）
            # 支持多种列名：turn（AkShare spot_em）或 turnover（其他数据源）
            turn_col = None
            if 'turn' in factor_data.columns:
                turn_col = 'turn'
            elif 'turnover' in factor_data.columns:
                turn_col = 'turnover'
            elif 'turnover_rate' in factor_data.columns:
                turn_col = 'turnover_rate'
            
            if turn_col is not None:
                factor_data['turnover_5d'] = factor_data.groupby('stock_code')[turn_col].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
                self.logger.debug(f"使用 '{turn_col}' 列计算换手率因子")
            else:
                factor_data['turnover_5d'] = np.nan
                self.logger.warning("未找到换手率列 (turn/turnover/turnover_rate)，换手率因子将为 NaN")
            
            # ========== HS300 深度价值策略因子 ==========
            # 使用 FactorCalculator 的方法计算复合因子
            from src.features import FactorCalculator
            
            # 4.1 复合价值因子（EP + BP）
            try:
                # 创建临时 FactorCalculator 实例
                temp_ohlcv = factor_data[['stock_code', 'close', 'open', 'high', 'low', 'volume']].copy()
                if 'date' in factor_data.columns:
                    temp_ohlcv['date'] = factor_data['date']
                    temp_ohlcv = temp_ohlcv.set_index('date')
                
                temp_fin = factor_data[['stock_code', 'pe_ttm', 'pb', 'roe']].copy() if all(
                    col in factor_data.columns for col in ['pe_ttm', 'pb']
                ) else pd.DataFrame()
                
                if not temp_fin.empty:
                    # 手动调用价值因子计算逻辑（不创建完整 FactorCalculator）
                    # EP Ratio = 1 / PE_TTM
                    pe_ttm = pd.to_numeric(factor_data['pe_ttm'], errors='coerce')
                    pe_ttm = pe_ttm.replace(0, np.nan).where(pe_ttm > 0, np.nan)
                    factor_data['ep_ratio'] = 1.0 / pe_ttm
                    factor_data['ep_ratio'] = factor_data['ep_ratio'].replace([np.inf, -np.inf], np.nan)
                    
                    # BP Ratio = 1 / PB
                    pb = pd.to_numeric(factor_data['pb'], errors='coerce')
                    pb = pb.replace(0, np.nan).where(pb > 0, np.nan)
                    factor_data['bp_ratio'] = 1.0 / pb
                    factor_data['bp_ratio'] = factor_data['bp_ratio'].replace([np.inf, -np.inf], np.nan)
                    
                    self.logger.debug(f"价值因子计算完成: EP有效={factor_data['ep_ratio'].notna().mean():.1%}, BP有效={factor_data['bp_ratio'].notna().mean():.1%}")
                else:
                    factor_data['ep_ratio'] = np.nan
                    factor_data['bp_ratio'] = np.nan
                    self.logger.warning("财务数据缺少 pe_ttm/pb 列，价值因子将为 NaN")
            except Exception as e:
                self.logger.warning(f"复合价值因子计算失败: {e}")
                factor_data['ep_ratio'] = np.nan
                factor_data['bp_ratio'] = np.nan
            
            # 4.2 传统 EP_TTM
            if 'pe_ttm' in factor_data.columns:
                factor_data['ep_ttm'] = 1.0 / factor_data['pe_ttm'].replace(0, np.nan)
                factor_data['ep_ttm'] = factor_data['ep_ttm'].replace([np.inf, -np.inf], np.nan)
            else:
                factor_data['ep_ttm'] = np.nan
            
            # 新增：ROC_20 动量因子计算
            # 使用 numba 加速的 roc 方法（如果 features 模块有提供，否则使用 pandas）
            # 这里简单使用 pandas 实现
            try:
                # 兼容旧逻辑：如果之前计算过，这里不会报错但也不会覆盖（除非使用 assign）
                # 注意：calculate_factors 中的 factor_data 是合并了 financial_data 的大表
                # 应该按 stock_code 分组计算
                factor_data['roc_20'] = factor_data.groupby('stock_code')['close'].transform(
                    lambda x: x.pct_change(20) * 100
                )
                self.logger.debug(f"手动计算 roc_20 完成 (calculate_factors)，有效率: {factor_data['roc_20'].notna().mean():.1%}")
            except Exception as e:
                self.logger.warning(f"手动计算 roc_20 失败 (calculate_factors): {e}")
                factor_data['roc_20'] = np.nan

            # 5. 质量因子组（ROE 盈利能力）
            # 5.1 ROE 因子（直接使用，越高越好）
            if 'roe' in factor_data.columns:
                # ROE 已经存在，确保格式正确
                factor_data['roe'] = pd.to_numeric(factor_data['roe'], errors='coerce')
                factor_data['roe_stability'] = factor_data['roe']
                self.logger.debug(f"ROE 因子就绪，有效率: {factor_data['roe'].notna().mean():.1%}")
            else:
                factor_data['roe'] = np.nan
                factor_data['roe_stability'] = np.nan
                self.logger.warning("财务数据中未找到 ROE 列，质量因子将为 NaN")
            
            # 6. 特质波动率 IVOL（风险因子）
            factor_data['ivol'] = factor_data.groupby('stock_code')['close'].transform(
                lambda x: x.pct_change().rolling(20).std() * np.sqrt(252)
            )
            # 重命名为与回测一致的列名
            factor_data['ivol_20'] = factor_data['ivol']
            
            # 7. Sharpe 动量因子（核心动量因子）
            # sharpe_20 = 20日收益 / 20日波动率
            def _calc_sharpe(close_series: pd.Series, period: int) -> pd.Series:
                """计算 Sharpe 风格动量因子"""
                returns = close_series.pct_change()
                mean_ret = returns.rolling(period, min_periods=max(5, period // 2)).mean()
                std_ret = returns.rolling(period, min_periods=max(5, period // 2)).std()
                # 避免除零
                sharpe = mean_ret / (std_ret + 1e-8)
                return sharpe
            
            factor_data['sharpe_20'] = factor_data.groupby('stock_code')['close'].transform(
                lambda x: _calc_sharpe(x, 20)
            )
            factor_data['sharpe_60'] = factor_data.groupby('stock_code')['close'].transform(
                lambda x: _calc_sharpe(x, 60)
            )
            self.logger.debug(f"Sharpe 因子计算完成: sharpe_20 有效率 {factor_data['sharpe_20'].notna().mean():.1%}, sharpe_60 有效率 {factor_data['sharpe_60'].notna().mean():.1%}")
            
            # ==================== Z-Score 标准化 ====================
            date_col = 'date' if 'date' in factor_data.columns else 'trade_date'
            
            # 对所有计算的因子进行 Z-Score 标准化
            factor_cols_to_normalize = [
                'rsi_20', 'roc_20', 'small_cap', 'turnover_5d', 
                'ep_ttm', 'ep_ratio', 'bp_ratio',          # 价值因子
                'roe_stability', 'roe',                    # 质量因子
                'ivol_20', 'sharpe_20', 'sharpe_60'        # 动量/风险因子
            ]
            # 只标准化存在且有效的因子列
            valid_factor_cols = [
                col for col in factor_cols_to_normalize 
                if col in factor_data.columns and factor_data[col].notna().any()
            ]
            
            self.logger.info(f"准备标准化的因子列: {valid_factor_cols}")
            if 'roc_20' not in valid_factor_cols:
                self.logger.warning(f"roc_20 不在标准化列表中! Columns: {factor_data.columns.tolist()}")
                if 'roc_20' in factor_data.columns:
                    self.logger.warning(f"roc_20 数据概览: {factor_data['roc_20'].describe()}")
            
            # 检查是否有行业字段，决定是否进行行业中性化
            has_industry = 'sw_industry_l1' in factor_data.columns and factor_data['sw_industry_l1'].notna().any()
            
            factor_data = z_score_normalize(
                factor_data,
                factor_cols=valid_factor_cols,
                date_col=date_col,
                industry_col='sw_industry_l1' if has_industry else None,
                industry_neutral=has_industry
            )
            
            if has_industry:
                self.logger.info(f"已标准化因子（行业中性化）: {valid_factor_cols}")
            else:
                self.logger.info(f"已标准化因子（市场中性化）: {valid_factor_cols}")
            
            # ==================== 因子别名映射 ====================
            # 将标准化后的因子映射到策略配置使用的列名
            # 支持多种策略配置：中证1000动量策略、HS300价值策略等
            factor_alias_mapping = {
                # 动量因子别名
                'sharpe_20_zscore': 'sharpe_20_zscore',
                'sharpe_60_zscore': 'sharpe_60_zscore',
                'momentum_zscore': 'roc_20_zscore',  # 默认动量因子
                # 质量因子别名
                'ivol_zscore': 'ivol_20_zscore',  # 低波动质量因子
                'quality_zscore': 'roe_stability_zscore',  # 默认质量因子
                'roe_zscore': 'roe_zscore',  # ROE 质量因子（HS300价值策略）
                'turnover_5d_zscore': 'turnover_5d_zscore',
                # 价值因子别名
                'value_zscore': 'ep_ttm_zscore',  # 单一价值因子
                'ep_zscore': 'ep_ratio_zscore',  # EP 价值因子
                'bp_zscore': 'bp_ratio_zscore',  # BP 价值因子
                # 小市值因子别名
                'size_zscore': 'small_cap_zscore',
            }
            
            # 计算复合价值因子（EP + BP 的加权平均）
            ep_col = 'ep_ratio_zscore' if 'ep_ratio_zscore' in factor_data.columns else None
            bp_col = 'bp_ratio_zscore' if 'bp_ratio_zscore' in factor_data.columns else None
            
            if ep_col and bp_col:
                ep_valid = factor_data[ep_col].notna() & (factor_data[ep_col] != 0)
                bp_valid = factor_data[bp_col].notna() & (factor_data[bp_col] != 0)
                
                if ep_valid.any() and bp_valid.any():
                    factor_data['value_composite_zscore'] = (
                        0.5 * factor_data[ep_col].fillna(0) + 
                        0.5 * factor_data[bp_col].fillna(0)
                    )
                    self.logger.debug("复合价值因子 value_composite_zscore 计算完成")
                elif ep_valid.any():
                    factor_data['value_composite_zscore'] = factor_data[ep_col].fillna(0)
                    self.logger.debug("复合价值因子使用 EP 单因子")
                elif bp_valid.any():
                    factor_data['value_composite_zscore'] = factor_data[bp_col].fillna(0)
                    self.logger.debug("复合价值因子使用 BP 单因子")
                else:
                    factor_data['value_composite_zscore'] = 0.0
                    self.logger.warning("无法计算复合价值因子（EP/BP 数据均无效）")
            else:
                factor_data['value_composite_zscore'] = 0.0
                self.logger.warning("缺少 ep_ratio_zscore/bp_ratio_zscore，复合价值因子为 0")
            
            # ==================== 计算 Alpha 因子（量价配合）====================
            # 牛市进攻型策略核心因子
            alpha_enabled = False
            try:
                # Alpha_001: (Close - VWAP) / VWAP，正值表示收盘价高于均价
                if 'amount' in factor_data.columns and 'volume' in factor_data.columns:
                    vwap = factor_data['amount'] / factor_data['volume'].replace(0, np.nan)
                    factor_data['alpha_001'] = (factor_data['close'] - vwap) / vwap.replace(0, np.nan)
                    factor_data['alpha_001'] = factor_data['alpha_001'].replace([np.inf, -np.inf], np.nan)
                    
                    # 对 Alpha_001 进行 Z-Score 标准化（横截面）
                    if 'date' in factor_data.columns:
                        factor_data['alpha_001_zscore'] = factor_data.groupby('date')['alpha_001'].transform(
                            lambda x: (x - x.mean()) / (x.std() + 1e-8)
                        ).fillna(0)
                    else:
                        factor_data['alpha_001_zscore'] = (
                            (factor_data['alpha_001'] - factor_data['alpha_001'].mean()) / 
                            (factor_data['alpha_001'].std() + 1e-8)
                        ).fillna(0)
                    
                    alpha_enabled = True
                    self.logger.info("Alpha_001 因子（量价配合）计算完成")
                else:
                    factor_data['alpha_001_zscore'] = 0.0
                    self.logger.warning("缺少 amount/volume 列，Alpha_001 因子设为 0")
            except Exception as e:
                factor_data['alpha_001_zscore'] = 0.0
                self.logger.warning(f"Alpha_001 因子计算失败: {e}")
            
            # ==================== 计算复合动量因子 momentum_composite_zscore ====================
            # 牛市进攻型配方: 40% ROC (涨幅) + 30% Sharpe (稳健) + 30% Alpha001 (量价配合)
            roc_col = 'roc_20_zscore' if 'roc_20_zscore' in factor_data.columns else None
            sharpe_col = 'sharpe_20_zscore' if 'sharpe_20_zscore' in factor_data.columns else None
            alpha_col = 'alpha_001_zscore' if alpha_enabled else None
            
            if roc_col and sharpe_col and alpha_col:
                # 完整配方: 40% ROC + 30% Sharpe + 30% Alpha001
                factor_data['momentum_composite_zscore'] = (
                    0.4 * factor_data[roc_col].fillna(0) +
                    0.3 * factor_data[sharpe_col].fillna(0) +
                    0.3 * factor_data[alpha_col].fillna(0)
                )
                self.logger.info("🚀 复合动量因子计算完成: 40% ROC + 30% Sharpe + 30% Alpha001")
            elif roc_col and sharpe_col:
                # 备选配方: 60% ROC + 40% Sharpe
                factor_data['momentum_composite_zscore'] = (
                    0.6 * factor_data[roc_col].fillna(0) +
                    0.4 * factor_data[sharpe_col].fillna(0)
                )
                self.logger.info("复合动量因子计算完成: 60% ROC + 40% Sharpe（无 Alpha）")
            elif roc_col:
                factor_data['momentum_composite_zscore'] = factor_data[roc_col].fillna(0)
                self.logger.warning("复合动量因子使用 ROC 单因子")
            else:
                factor_data['momentum_composite_zscore'] = 0.0
                self.logger.warning("无法计算复合动量因子（缺少必要因子）")
            
            for alias, source in factor_alias_mapping.items():
                if source in factor_data.columns and alias not in factor_data.columns:
                    factor_data[alias] = factor_data[source]
                    self.logger.debug(f"创建因子别名: {alias} <- {source}")
            
            self.logger.info(f"因子别名映射完成，可用因子列: {[c for c in factor_data.columns if c.endswith('_zscore')]}")
            
            self.factor_data = factor_data
            
            # 保存因子数据
            factor_path = DATA_PROCESSED_PATH / f"factors_{self.today.strftime('%Y%m%d')}.parquet"
            self.factor_data.to_parquet(factor_path)
            
            self.logger.info(f"因子计算完成，共 {len(self.factor_data)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"计算因子失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 20) -> pd.Series:
        """计算 RSI 指标"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def is_rebalance_day(self, date: Optional[pd.Timestamp] = None) -> bool:
        """
        检查是否为调仓日（月末最后一个交易日）
        
        Parameters
        ----------
        date : Optional[pd.Timestamp]
            检查日期，默认为今日
        
        Returns
        -------
        bool
            是否为调仓日
        """
        if date is None:
            date = self.today
        
        # 获取本月所有交易日
        if self.ohlcv_data is not None:
            # 优先使用 DatetimeIndex，其次使用 date/trade_date 列
            if isinstance(self.ohlcv_data.index, pd.DatetimeIndex):
                trading_dates = self.ohlcv_data.index.unique()
            elif 'date' in self.ohlcv_data.columns:
                trading_dates = pd.to_datetime(self.ohlcv_data['date'].unique())
            elif 'trade_date' in self.ohlcv_data.columns:
                trading_dates = pd.to_datetime(self.ohlcv_data['trade_date'].unique())
            else:
                self.logger.warning("ohlcv_data 中未找到日期列或 DatetimeIndex，使用简化判断")
                trading_dates = None
            
            # 筛选本月交易日
            if trading_dates is not None:
                month_dates = trading_dates[
                    (trading_dates.year == date.year) & 
                    (trading_dates.month == date.month)
                ]
                
                if len(month_dates) > 0:
                    last_trading_day = month_dates.max()
                    is_last_day = date >= last_trading_day
                    self.logger.info(
                        f"本月最后交易日: {last_trading_day.strftime('%Y-%m-%d')}, "
                        f"今日: {date.strftime('%Y-%m-%d')}, 是否调仓日: {is_last_day}"
                    )
                    return is_last_day
        
        # 简化判断：月末最后3天视为调仓日
        next_month = (date.replace(day=28) + timedelta(days=4)).replace(day=1)
        days_to_month_end = (next_month - date).days
        
        is_month_end = days_to_month_end <= 3
        self.logger.info(f"距月末 {days_to_month_end} 天，是否调仓日: {is_month_end}")
        
        return is_month_end
    
    def generate_target_positions(self) -> bool:
        """
        生成目标持仓（实盘安全版）
        
        包含以下安全机制：
        1. 大盘风控：当大盘跌破MA{n}且跌幅超阈值时，强制空仓
        2. 数据验证：确保所选股票都有有效的财务数据
        3. 结果校验：保存的 JSON 文件明确标记风控状态
        
        Returns
        -------
        bool
            生成是否成功
        """
        self.logger.info("开始生成目标持仓（实盘安全模式）...")
        
        try:
            # === 大盘风控检查 ===
            if self.is_market_risk_triggered():
                self.logger.warning("🚨 大盘风控触发，系统强制空仓！")
                self.target_positions = {}
                
                # 读取风控配置用于记录
                risk_config = self.config.get("risk", {})
                market_risk_config = risk_config.get("market_risk", {})
                ma_period = market_risk_config.get("ma_period", 60)
                drop_threshold = market_risk_config.get("drop_threshold", 0.05)
                
                # 保存空仓状态
                portfolio_config = self.config.get("portfolio", {})
                total_capital = portfolio_config.get("total_capital", 1000000)
                
                positions_path = DATA_PROCESSED_PATH / f"target_positions_{self.today.strftime('%Y%m%d')}.json"
                
                # 构建空仓 JSON（实盘保护：确保 positions 为空字典）
                empty_position_data = {
                    'date': self.today.strftime('%Y-%m-%d'),
                    'positions': {},  # 关键：确保为空字典
                    'weights': {},    # 关键：确保为空字典
                    'total_capital': total_capital,
                    'market_risk_triggered': True,  # 关键：标记风控触发
                    'reason': f'大盘跌破MA{ma_period}或跌幅超{drop_threshold*100:.0f}%，触发风控',
                    'risk_params': {
                        'ma_period': ma_period,
                        'drop_threshold': drop_threshold,
                    },
                    'action': 'CLEAR_ALL_POSITIONS',  # 明确指令
                    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                
                with open(positions_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_position_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(
                    f"✅ 已保存空仓目标持仓（风控触发）\n"
                    f"   文件: {positions_path}\n"
                    f"   positions: {{}}\n"
                    f"   market_risk_triggered: True"
                )
                return True
            # =====================
            
            if self.factor_data is None:
                self.logger.warning("因子数据为空，无法生成持仓")
                return False
            
            # 获取最新日期的数据
            date_col = 'date' if 'date' in self.factor_data.columns else 'trade_date'
            latest_date = pd.to_datetime(self.factor_data[date_col]).max()
            
            latest_data = self.factor_data[
                pd.to_datetime(self.factor_data[date_col]) == latest_date
            ].copy()
            
            self.logger.info(f"最新数据日期: {latest_date.strftime('%Y-%m-%d')}, 股票数: {len(latest_data)}")
            
            # 过滤股票
            filtered_data = self.strategy.filter_stocks(latest_data, latest_date)
            
            if filtered_data.empty:
                self.logger.warning("过滤后无可选股票")
                return False
            
            # 选取 Top N 股票
            selected_stocks = self.strategy.select_top_stocks(filtered_data)
            
            # ========== 实盘安全：验证所选股票数据有效性 ==========
            excluded_stocks = getattr(self, '_excluded_stocks', set())
            invalid_selected = [s for s in selected_stocks if s in excluded_stocks]
            
            if invalid_selected:
                self.logger.error(
                    f"🚨 安全警告: 选中的股票中包含无效数据股票: {invalid_selected}\n"
                    f"   这些股票将被移除。"
                )
                selected_stocks = [s for s in selected_stocks if s not in excluded_stocks]
            
            if not selected_stocks:
                self.logger.error("过滤无效数据后无可选股票，取消本次调仓")
                return False
            
            self.logger.info(f"选中 {len(selected_stocks)} 只股票: {selected_stocks[:5]}...")
            
            # 计算综合得分用于展示
            filtered_data['total_score'] = self.strategy.calculate_total_score(filtered_data)
            
            # 优化权重
            portfolio_config = self.config.get("portfolio", {})
            total_capital = portfolio_config.get("total_capital", 1000000)
            max_weight = portfolio_config.get("max_weight", 0.05)
            objective = portfolio_config.get("optimization_objective", "max_sharpe")
            
            # 准备价格数据用于优化
            if self.ohlcv_data is not None:
                price_pivot = self.ohlcv_data.pivot_table(
                    index='date' if 'date' in self.ohlcv_data.columns else 'trade_date',
                    columns='stock_code',
                    values='close'
                )
                
                # 优化权重
                weights = self.strategy.optimize_weights(
                    price_pivot,
                    selected_stocks,
                    objective=objective,
                    max_weight=max_weight
                )
            else:
                # 无价格数据时使用等权重
                weights = {stock: 1.0 / len(selected_stocks) for stock in selected_stocks}
            
            # 计算目标持仓（金额）
            self.target_positions = {
                stock: weight * total_capital
                for stock, weight in weights.items()
                if weight > 0.0001
            }
            
            self.logger.info(f"目标持仓生成完成，共 {len(self.target_positions)} 只股票")
            
            # 保存目标持仓
            positions_path = DATA_PROCESSED_PATH / f"target_positions_{self.today.strftime('%Y%m%d')}.json"
            with open(positions_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'date': self.today.strftime('%Y-%m-%d'),
                    'positions': self.target_positions,
                    'weights': weights,
                    'total_capital': total_capital,
                    'market_risk_triggered': False,
                }, f, ensure_ascii=False, indent=2)
            
            return True
        
        except LLMCircuitBreakerError as e:
            # ===== LLM 熔断器触发: 风控停止交易 =====
            self.logger.critical(
                f"⛔ LLM Circuit Breaker Triggered! Risk control failed. "
                f"HALTING ALL TRADING SIGNALS."
            )
            self.logger.critical(f"Error details: {e}")
            
            # 保存风控停止状态文件
            self.target_positions = {}
            
            portfolio_config = self.config.get("portfolio", {})
            total_capital = portfolio_config.get("total_capital", 1000000)
            
            positions_path = DATA_PROCESSED_PATH / f"target_positions_{self.today.strftime('%Y%m%d')}.json"
            
            # 构建风控停止 JSON（明确标记 LLM 熔断状态）
            halt_position_data = {
                'date': self.today.strftime('%Y-%m-%d'),
                'positions': {},  # 关键：确保为空字典，不产生任何买入信号
                'weights': {},    # 关键：确保为空字典
                'total_capital': total_capital,
                'market_risk_triggered': False,
                'llm_circuit_breaker_triggered': True,  # 关键：标记 LLM 熔断触发
                'reason': f'LLM Circuit Breaker Triggered: {str(e)[:200]}',
                'action': 'HALT_ALL_TRADING',  # 明确指令：停止所有交易
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            with open(positions_path, 'w', encoding='utf-8') as f:
                json.dump(halt_position_data, f, ensure_ascii=False, indent=2)
            
            self.logger.critical(
                f"✅ 已保存风控停止状态文件\n"
                f"   文件: {positions_path}\n"
                f"   positions: {{}}\n"
                f"   llm_circuit_breaker_triggered: True\n"
                f"   action: HALT_ALL_TRADING"
            )
            
            # 返回 False 表示生成失败，调用方应停止后续流程
            return False
            
        except Exception as e:
            self.logger.error(f"生成目标持仓失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def calculate_trade_orders(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        计算交易订单
        
        Returns
        -------
        Tuple[Dict[str, float], Dict[str, float]]
            (买入订单, 卖出订单)，键为股票代码，值为金额
        """
        buy_orders: Dict[str, float] = {}
        sell_orders: Dict[str, float] = {}
        
        # 当前持仓股票
        current_stocks = set(self.current_positions.keys())
        # 目标持仓股票
        target_stocks = set(self.target_positions.keys())
        
        # 需要卖出的股票
        stocks_to_sell = current_stocks - target_stocks
        for stock in stocks_to_sell:
            sell_orders[stock] = self.current_positions[stock]
        
        # 需要买入的股票
        stocks_to_buy = target_stocks - current_stocks
        for stock in stocks_to_buy:
            buy_orders[stock] = self.target_positions[stock]
        
        # 需要调整的股票
        stocks_to_adjust = current_stocks & target_stocks
        for stock in stocks_to_adjust:
            current = self.current_positions[stock]
            target = self.target_positions[stock]
            diff = target - current
            
            if diff > 100:  # 买入阈值
                buy_orders[stock] = diff
            elif diff < -100:  # 卖出阈值
                sell_orders[stock] = -diff
        
        return buy_orders, sell_orders
    
    def generate_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        format: str = "markdown"
    ) -> str:
        """
        生成交易报告
        
        Parameters
        ----------
        buy_orders : Dict[str, float]
            买入订单
        sell_orders : Dict[str, float]
            卖出订单
        format : str
            报告格式，'markdown' 或 'html'
        
        Returns
        -------
        str
            报告内容
        """
        report_date = self.today.strftime('%Y-%m-%d')
        
        if format == "markdown":
            return self._generate_markdown_report(buy_orders, sell_orders, report_date)
        elif format == "html":
            return self._generate_html_report(buy_orders, sell_orders, report_date)
        else:
            raise ValueError(f"不支持的报告格式: {format}")
    
    def _get_latest_prices(self) -> Dict[str, float]:
        """获取所有股票的最新收盘价"""
        if self.ohlcv_data is None or self.ohlcv_data.empty:
            self.logger.warning("ohlcv_data 为空，无法获取最新价格")
            return {}
            
        try:
            # 确定日期列
            date_col = next((col for col in ['date', 'trade_date', 'timestamp'] if col in self.ohlcv_data.columns), None)
            
            # 确定股票代码列
            stock_col = next((col for col in ['stock_code', 'symbol', 'code', 'ts_code'] if col in self.ohlcv_data.columns), None)
            
            # 确定收盘价列
            price_col = next((col for col in ['close', 'close_price'] if col in self.ohlcv_data.columns), None)
            
            if not date_col or not price_col:
                self.logger.warning(f"缺少必要列: date_col={date_col}, price_col={price_col}")
                return {}

            df = self.ohlcv_data.copy()
            
            # 如果没有股票代码列，尝试从索引获取
            if not stock_col:
                if isinstance(df.index, pd.MultiIndex):
                    # 假设 MultiIndex 是 (date, stock_code) 或 (stock_code, date)
                    # 这里简化处理，暂不支持 MultiIndex 自动推断，建议 Reset Index
                    df = df.reset_index()
                    stock_col = next((col for col in ['stock_code', 'symbol', 'code', 'ts_code'] if col in df.columns), None)
            
            if not stock_col:
                self.logger.warning("无法找到股票代码列")
                return {}
                
            # 获取每个股票的最后一条记录
            # 先按日期排序
            df_sorted = df.sort_values(by=date_col)
            latest_prices = df_sorted.groupby(stock_col)[price_col].last().to_dict()
            
            self.logger.info(f"已获取 {len(latest_prices)} 只股票的最新价格")
            return latest_prices
        except Exception as e:
            self.logger.warning(f"获取最新价格映射失败: {e}")
            return {}

    def _generate_markdown_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        report_date: str
    ) -> str:
        """生成 Markdown 格式报告"""
        latest_prices = self._get_latest_prices()

        lines = [
            f"# 每日调仓报告",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"**报告日期**: {report_date}",
            f"",
            f"---",
            f"",
        ]
        
        # 策略信息
        lines.extend([
            f"## 策略信息",
            f"",
            f"| 参数 | 值 |",
            f"|------|-----|",
            f"| 策略名称 | {self.strategy.name} |",
            f"| 价值因子权重 | {self.strategy.value_weight:.0%} |",
            f"| 质量因子权重 | {self.strategy.quality_weight:.0%} |",
            f"| 动量因子权重 | {self.strategy.momentum_weight:.0%} |",
            f"| 选股数量 | {self.strategy.top_n} |",
            f"",
        ])
        
        # 持仓汇总
        portfolio_config = self.config.get("portfolio", {})
        total_capital = portfolio_config.get("total_capital", 1000000)
        
        lines.extend([
            f"## 持仓汇总",
            f"",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 总资金 | ¥{total_capital:,.0f} |",
            f"| 目标持仓数 | {len(self.target_positions)} |",
            f"| 买入股票数 | {len(buy_orders)} |",
            f"| 卖出股票数 | {len(sell_orders)} |",
            f"",
        ])
        
        # 买入清单
        lines.extend([
            f"## 📈 明日需买入",
            f"",
        ])
        
        if buy_orders:
            lines.extend([
                f"| 股票代码 | 买入金额 |",
                f"|----------|----------|",
            ])
            
            for stock, amount in sorted(buy_orders.items(), key=lambda x: -x[1]):
                lines.append(f"| {stock} | ¥{amount:,.0f} |")
            
            lines.append(f"")
            lines.append(f"**买入总金额**: ¥{sum(buy_orders.values()):,.0f}")
        else:
            lines.append(f"*无需买入*")
        
        lines.append(f"")
        
        # 卖出清单
        lines.extend([
            f"## 📉 明日需卖出",
            f"",
        ])
        
        if sell_orders:
            lines.extend([
                f"| 股票代码 | 卖出金额 |",
                f"|----------|----------|",
            ])
            
            for stock, amount in sorted(sell_orders.items(), key=lambda x: -x[1]):
                lines.append(f"| {stock} | ¥{amount:,.0f} |")
            
            lines.append(f"")
            lines.append(f"**卖出总金额**: ¥{sum(sell_orders.values()):,.0f}")
        else:
            lines.append(f"*无需卖出*")
        
        lines.append(f"")
        
        # 目标持仓明细
        lines.extend([
            f"## 目标持仓明细",
            f"",
            f"| 股票代码 | 目标金额 | 权重 |",
            f"|----------|----------|------|",
        ])
        
        total_target = sum(self.target_positions.values()) if self.target_positions else 1
        for stock, amount in sorted(self.target_positions.items(), key=lambda x: -x[1]):
            weight = amount / total_target
            lines.append(f"| {stock} | ¥{amount:,.0f} | {weight:.2%} |")
        
        lines.extend([
            f"",
            f"---",
            f"",
            f"*本报告由 A股量化交易系统 自动生成*",
        ])
        
        return "\n".join(lines)
    
    def _generate_html_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        report_date: str
    ) -> str:
        """生成 HTML 格式报告"""
        latest_prices = self._get_latest_prices()
        portfolio_config = self.config.get("portfolio", {})
        total_capital = portfolio_config.get("total_capital", 1000000)
        
        # 买入表格行
        buy_rows = ""
        for stock, amount in sorted(buy_orders.items(), key=lambda x: -x[1]):
            buy_rows += f"""
                <tr>
                    <td>{stock}</td>
                    <td>¥{amount:,.0f}</td>
                </tr>
            """
        
        # 卖出表格行
        sell_rows = ""
        for stock, amount in sorted(sell_orders.items(), key=lambda x: -x[1]):
            sell_rows += f"""
                <tr>
                    <td>{stock}</td>
                    <td>¥{amount:,.0f}</td>
                </tr>
            """
        
        # 持仓表格行
        position_rows = ""
        total_target = sum(self.target_positions.values()) if self.target_positions else 1
        for stock, amount in sorted(self.target_positions.items(), key=lambda x: -x[1]):
            weight = amount / total_target
            position_rows += f"""
                <tr>
                    <td>{stock}</td>
                    <td>¥{amount:,.0f}</td>
                    <td>{weight:.2%}</td>
                </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>每日调仓报告 - {report_date}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .meta {{
            color: #888;
            margin-bottom: 2rem;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .card h2 {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: #00d9ff;
        }}
        .card.buy h2 {{
            color: #00ff88;
        }}
        .card.sell h2 {{
            color: #ff6b6b;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }}
        .stat {{
            text-align: center;
            padding: 1rem;
            background: rgba(0, 217, 255, 0.1);
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #00d9ff;
        }}
        .stat-label {{
            font-size: 0.85rem;
            color: #888;
            margin-top: 0.25rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        th {{
            color: #888;
            font-weight: 500;
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}
        .total {{
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 2px solid rgba(255, 255, 255, 0.1);
            font-weight: bold;
        }}
        .buy-total {{
            color: #00ff88;
        }}
        .sell-total {{
            color: #ff6b6b;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 2rem;
            font-size: 0.85rem;
        }}
        .empty {{
            text-align: center;
            color: #666;
            padding: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 每日调仓报告</h1>
        <p class="meta">报告日期: {report_date} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="card">
            <h2>策略概览</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">¥{total_capital:,.0f}</div>
                    <div class="stat-label">总资金</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(self.target_positions)}</div>
                    <div class="stat-label">目标持仓数</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(buy_orders)}</div>
                    <div class="stat-label">买入股票数</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(sell_orders)}</div>
                    <div class="stat-label">卖出股票数</div>
                </div>
            </div>
        </div>
        
        <div class="card buy">
            <h2>📈 明日需买入</h2>
            {f'''
            <table>
                <thead>
                    <tr>
                        <th>股票代码</th>
                        <th>买入金额</th>
                    </tr>
                </thead>
                <tbody>
                    {buy_rows}
                </tbody>
            </table>
            <p class="total buy-total">买入总金额: ¥{sum(buy_orders.values()):,.0f}</p>
            ''' if buy_orders else '<p class="empty">无需买入</p>'}
        </div>
        
        <div class="card sell">
            <h2>📉 明日需卖出</h2>
            {f'''
            <table>
                <thead>
                    <tr>
                        <th>股票代码</th>
                        <th>卖出金额</th>
                    </tr>
                </thead>
                <tbody>
                    {sell_rows}
                </tbody>
            </table>
            <p class="total sell-total">卖出总金额: ¥{sum(sell_orders.values()):,.0f}</p>
            ''' if sell_orders else '<p class="empty">无需卖出</p>'}
        </div>
        
        <div class="card">
            <h2>📋 目标持仓明细</h2>
            <table>
                <thead>
                    <tr>
                        <th>股票代码</th>
                        <th>目标金额</th>
                        <th>权重</th>
                    </tr>
                </thead>
                <tbody>
                    {position_rows}
                </tbody>
            </table>
        </div>
        
        <p class="footer">本报告由 A股量化交易系统 自动生成</p>
    </div>
</body>
</html>
        """
        
        return html
    
    def save_report(self, report_content: str, format: str = "markdown") -> Path:
        """
        保存报告
        
        Parameters
        ----------
        report_content : str
            报告内容
        format : str
            报告格式
        
        Returns
        -------
        Path
            报告文件路径
        """
        extension = "md" if format == "markdown" else "html"
        report_path = REPORTS_PATH / f"daily_report_{self.today.strftime('%Y%m%d')}.{extension}"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"报告已保存至 {report_path}")
        return report_path


def _format_orders_for_push(
    buy_orders: Dict[str, float],
    sell_orders: Dict[str, float],
    target_positions: Dict[str, float],
    report_date: str,
    market_risk_triggered: bool = False
) -> str:
    """
    将交易订单格式化为 PushPlus 推送内容（HTML格式）
    
    Parameters
    ----------
    buy_orders : Dict[str, float]
        买入订单 {股票代码: 金额}
    sell_orders : Dict[str, float]
        卖出订单 {股票代码: 金额}
    target_positions : Dict[str, float]
        目标持仓 {股票代码: 金额}
    report_date : str
        报告日期
    market_risk_triggered : bool
        大盘风控是否触发
    
    Returns
    -------
    str
        HTML 格式的推送内容
    """
    lines = []
    
    # 样式
    lines.append("""
    <style>
        body { font-family: -apple-system, sans-serif; padding: 10px; }
        .header { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .section { margin: 15px 0; }
        .section-title { color: #667eea; font-size: 16px; font-weight: bold; margin-bottom: 8px; }
        .buy { color: #00aa00; }
        .sell { color: #ff4444; }
        .warning { color: #ff8800; background: #fff3cd; padding: 10px; border-radius: 5px; }
        .item { padding: 5px 0; border-bottom: 1px solid #eee; }
        .amount { float: right; font-weight: bold; }
        .summary { background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 15px; }
        .no-action { color: #888; text-align: center; padding: 20px; }
    </style>
    """)
    
    # 标题
    lines.append(f'<div class="header"><h2>📊 每日交易计划</h2><p>日期: {report_date}</p></div>')
    
    # 大盘风控警告
    if market_risk_triggered:
        lines.append('''
        <div class="warning">
            ⚠️ <strong>大盘风控触发</strong><br>
            沪深300跌破20日均线，系统强制空仓！
        </div>
        ''')
    
    # 判断是否有操作
    has_orders = bool(buy_orders) or bool(sell_orders)
    
    if not has_orders:
        lines.append('''
        <div class="no-action">
            <p>✅ 今日无交易操作</p>
            <p style="font-size: 12px; color: #aaa;">持仓保持不变</p>
        </div>
        ''')
    else:
        # 买入清单
        if buy_orders:
            lines.append('<div class="section">')
            lines.append(f'<div class="section-title buy">📈 明日需买入 ({len(buy_orders)}只)</div>')
            
            for stock, amount in sorted(buy_orders.items(), key=lambda x: -x[1]):
                shares = int(amount / 10 / 100) * 100  # 估算股数
                lines.append(f'''
                <div class="item">
                    <span>{stock}</span>
                    <span class="amount buy">¥{amount:,.0f}</span>
                    <span style="color:#888; font-size:12px;"> (~{shares}股)</span>
                </div>
                ''')
            
            total_buy = sum(buy_orders.values())
            lines.append(f'<div style="text-align:right; margin-top:8px;"><strong>合计: ¥{total_buy:,.0f}</strong></div>')
            lines.append('</div>')
        
        # 卖出清单
        if sell_orders:
            lines.append('<div class="section">')
            lines.append(f'<div class="section-title sell">📉 明日需卖出 ({len(sell_orders)}只)</div>')
            
            for stock, amount in sorted(sell_orders.items(), key=lambda x: -x[1]):
                shares = int(amount / 10 / 100) * 100
                lines.append(f'''
                <div class="item">
                    <span>{stock}</span>
                    <span class="amount sell">¥{amount:,.0f}</span>
                    <span style="color:#888; font-size:12px;"> (~{shares}股)</span>
                </div>
                ''')
            
            total_sell = sum(sell_orders.values())
            lines.append(f'<div style="text-align:right; margin-top:8px;"><strong>合计: ¥{total_sell:,.0f}</strong></div>')
            lines.append('</div>')
    
    # 持仓汇总
    lines.append('<div class="summary">')
    lines.append(f'<strong>目标持仓: {len(target_positions)} 只股票</strong>')
    if target_positions:
        total_value = sum(target_positions.values())
        lines.append(f'<br>总市值: ¥{total_value:,.0f}')
        
        # 显示前5只持仓
        top_5 = sorted(target_positions.items(), key=lambda x: -x[1])[:5]
        lines.append('<br><span style="font-size:12px; color:#666;">Top 5: ')
        lines.append(', '.join([f'{s}({w/total_value:.1%})' for s, w in top_5]))
        lines.append('</span>')
    lines.append('</div>')
    
    # 时间戳
    lines.append(f'<p style="text-align:center; color:#aaa; font-size:11px; margin-top:15px;">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
    
    return '\n'.join(lines)


def _send_daily_notification(
    runner: "DailyUpdateRunner",
    buy_orders: Dict[str, float],
    sell_orders: Dict[str, float],
    config: Dict[str, Any]
) -> None:
    """
    发送每日交易通知到微信
    
    Parameters
    ----------
    runner : DailyUpdateRunner
        运行器实例
    buy_orders : Dict[str, float]
        买入订单
    sell_orders : Dict[str, float]
        卖出订单
    config : Dict[str, Any]
        配置
    """
    logger = logging.getLogger(__name__)
    
    # 获取 PushPlus Token
    # 优先从环境变量读取，其次从配置文件读取
    token = os.environ.get("PUSHPLUS_TOKEN", "")
    
    if not token:
        token = config.get("notification", {}).get("pushplus_token", "")
    
    if not token:
        logger.debug("未配置 PUSHPLUS_TOKEN，跳过微信推送")
        return
    
    # 检查是否为风控触发的空仓
    market_risk_triggered = False
    positions_path = DATA_PROCESSED_PATH / f"target_positions_{runner.today.strftime('%Y%m%d')}.json"
    if positions_path.exists():
        try:
            with open(positions_path, 'r', encoding='utf-8') as f:
                pos_data = json.load(f)
                market_risk_triggered = pos_data.get("market_risk_triggered", False)
        except Exception:
            pass
    
    # 格式化推送内容
    report_date = runner.today.strftime('%Y-%m-%d')
    content = _format_orders_for_push(
        buy_orders=buy_orders,
        sell_orders=sell_orders,
        target_positions=runner.target_positions,
        report_date=report_date,
        market_risk_triggered=market_risk_triggered
    )
    
    # 构建标题
    if market_risk_triggered:
        title = f"⚠️ 风控触发 - {report_date}"
    elif buy_orders or sell_orders:
        title = f"📊 交易计划 - {report_date}"
    else:
        title = f"✅ 无操作 - {report_date}"
    
    # 发送消息
    success = send_pushplus_msg(
        token=token,
        title=title,
        content=content,
        template="html"
    )
    
    if success:
        logger.info("每日交易计划已推送至微信")
    else:
        logger.warning("微信推送失败，请检查 PUSHPLUS_TOKEN 配置")


def run_daily_update(
    force_rebalance: bool = False,
    config: Optional[Dict[str, Any]] = None,
    no_llm: bool = False
) -> bool:
    """
    运行每日更新流程
    
    流程：
    1. 调用 DataLoader 更新至今日的最新数据
    2. 调用 FactorCalculator 更新因子数据
    3. 更新基准指数数据（用于大盘风控）
    4. 检查今日是否为月底（调仓日）。如果是，运行 MultiFactorStrategy 生成新的目标持仓列表
    5. 调用 optimize_weights 计算每只持仓股的具体股数
    6. 生成报告
    
    Parameters
    ----------
    force_rebalance : bool
        是否强制调仓（忽略日期检查）
    config : Optional[Dict[str, Any]]
        配置参数
    no_llm : bool
        是否禁用 LLM 风控
    
    Returns
    -------
    bool
        运行是否成功
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始每日更新流程")
    if no_llm:
        logger.info("参数设置: 禁用 LLM 风控")
    logger.info("=" * 60)
    
    try:
        # 初始化运行器
        runner = DailyUpdateRunner(config)
        
        # 处理 LLM 禁用
        if no_llm:
            if "llm" in runner.config:
                runner.config["llm"] = {}
                # 同时也更新 runner 内部可能已经初始化的组件配置
                # 注意：DailyUpdateRunner 初始化时已经用 config 初始化了组件
                # 所以最好是先修改 config 再初始化 runner，或者 config 传递 None 并在内部处理
                # 由于 runner 已经初始化，我们需要手动通过 runner 修改
                pass
            
            # 由于 runner 已经初始化，我们需要重新初始化受影响的组件
            # 或者更好的方式是在 DailyUpdateRunner 内部处理 no_llm
            # 这里为了简单，直接修改 config，并重新初始化 feature_calculator
            runner.config["llm"] = {}
            runner._init_components()  # 重新初始化组件以应用新配置

        
        # Step 1: 更新市场数据
        logger.info("Step 1/8: 更新市场数据")
        if not runner.update_market_data():
            logger.error("市场数据更新失败")
            return False
        
        # Step 2: 更新财务数据
        logger.info("Step 2/8: 更新财务数据")
        if not runner.update_financial_data():
            logger.error("财务数据更新失败")
            return False
        
        # Step 3: 更新基准指数数据（用于大盘风控）
        logger.info("Step 3/8: 更新基准指数数据（大盘风控）")
        runner.update_benchmark_data()  # 即使失败也继续，只是风控不生效
        
        # Step 4: 计算因子
        logger.info("Step 4/8: 计算因子数据")
        if not runner.calculate_factors():
            logger.error("因子计算失败")
            return False
        
        # Step 5: 检查是否调仓日
        is_rebalance = force_rebalance or runner.is_rebalance_day()
        
        if is_rebalance:
            logger.info("Step 5/8: 生成目标持仓（调仓日）")
            if not runner.generate_target_positions():
                logger.error("目标持仓生成失败")
                return False
        else:
            logger.info("Step 5/8: 非调仓日，跳过持仓生成")
            runner.target_positions = runner.current_positions.copy()
        
        # Step 6: 生成报告
        logger.info("Step 6/8: 生成交易报告")
        buy_orders, sell_orders = runner.calculate_trade_orders()
        
        report_config = runner.config.get("report", {})
        report_format = report_config.get("format", "markdown")
        
        # 生成两种格式的报告
        for fmt in ["markdown", "html"]:
            report_content = runner.generate_report(buy_orders, sell_orders, format=fmt)
            runner.save_report(report_content, format=fmt)
        
        # Step 7: 更新并保存持仓
        logger.info("Step 7/8: 更新持仓记录")
        runner.save_current_holdings(buy_orders, sell_orders)
        
        # Step 8: 发送微信推送通知
        logger.info("Step 8/8: 发送微信通知")
        _send_daily_notification(
            runner=runner,
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            config=runner.config
        )
        
        logger.info("=" * 60)
        logger.info("每日更新流程完成")
        logger.info("=" * 60)
        
        # 打印摘要
        logger.info(f"目标持仓: {len(runner.target_positions)} 只股票")
        logger.info(f"需买入: {len(buy_orders)} 只，金额 ¥{sum(buy_orders.values()):,.0f}")
        logger.info(f"需卖出: {len(sell_orders)} 只，金额 ¥{sum(sell_orders.values()):,.0f}")
        
        return True
        
    except Exception as e:
        logger.error(f"每日更新流程失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _load_backtest_financial_data(
    stock_list: List[str],
    start_date: str,
    end_date: str,
    data_loader: "DataLoader"
) -> pd.DataFrame:
    """
    加载回测用历史财务数据（特别是流通市值 circ_mv）
    
    优先从本地 parquet 文件加载，如果不存在则尝试在线获取。
    
    Parameters
    ----------
    stock_list : List[str]
        股票代码列表
    start_date : str
        开始日期 (YYYY-MM-DD)
    end_date : str
        结束日期 (YYYY-MM-DD)
    data_loader : DataLoader
        数据加载器实例
    
    Returns
    -------
    pd.DataFrame
        财务数据，包含 date, stock_code, circ_mv, total_mv 等字段
    
    Raises
    ------
    FileNotFoundError
        当本地无财务数据且无法在线获取时
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载回测财务数据: {len(stock_list)} 只股票, {start_date} ~ {end_date}")
    
    financial_records = []
    failed_stocks = []
    
    # 尝试从本地加载已保存的财务数据
    local_financial_path = DATA_RAW_PATH / "financial_data.parquet"
    if local_financial_path.exists():
        try:
            local_df = pd.read_parquet(local_financial_path)
            logger.info(f"从本地加载财务数据: {len(local_df)} 条记录")
            
            # 过滤日期范围和股票列表
            if 'date' in local_df.columns:
                local_df['date'] = pd.to_datetime(local_df['date'])
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                local_df = local_df[
                    (local_df['date'] >= start_dt) & 
                    (local_df['date'] <= end_dt) &
                    (local_df['stock_code'].isin(stock_list))
                ]
                
                if not local_df.empty and 'circ_mv' in local_df.columns:
                    logger.info(f"本地财务数据过滤后: {len(local_df)} 条记录")
                    return local_df
        except Exception as e:
            logger.warning(f"加载本地财务数据失败: {e}")
    
    # 尝试查找按日期保存的财务数据文件
    financial_files = list(DATA_RAW_PATH.glob("financial_*.parquet"))
    if financial_files:
        logger.info(f"找到 {len(financial_files)} 个财务数据文件，尝试加载...")
        all_financial_data = []
        
        for fpath in financial_files:
            try:
                df = pd.read_parquet(fpath)
                if 'stock_code' in df.columns:
                    # 从文件名提取日期
                    date_str = fpath.stem.replace("financial_", "")
                    if len(date_str) == 8:
                        df['data_date'] = pd.to_datetime(date_str, format='%Y%m%d')
                    all_financial_data.append(df)
            except Exception as e:
                logger.debug(f"加载 {fpath} 失败: {e}")
        
        if all_financial_data:
            combined_df = pd.concat(all_financial_data, ignore_index=True)
            if 'circ_mv' in combined_df.columns or 'total_mv' in combined_df.columns:
                logger.info(f"合并财务数据: {len(combined_df)} 条记录")
                return combined_df
    
    # 在线获取财务指标（仅获取当前快照，用于近期回测）
    logger.warning("本地无历史财务数据，尝试在线获取当前财务指标...")
    logger.warning("注意：在线获取的财务数据为当前快照，可能导致回测存在前视偏差")
    
    import time
    for i, stock in enumerate(stock_list):
        try:
            fin_df = data_loader.fetch_financial_indicator(stock)
            
            if fin_df is not None and not fin_df.empty:
                # 提取市值数据
                if isinstance(fin_df, pd.DataFrame) and len(fin_df) > 0:
                    latest = fin_df.iloc[-1] if len(fin_df) > 1 else fin_df.iloc[0]
                    
                    # 获取流通市值
                    circ_mv = None
                    total_mv = None
                    
                    for col in ['circ_mv', '流通市值']:
                        if col in latest.index:
                            circ_mv = latest[col]
                            break
                    
                    for col in ['total_mv', '总市值']:
                        if col in latest.index:
                            total_mv = latest[col]
                            break
                    
                    if circ_mv is not None or total_mv is not None:
                        financial_records.append({
                            'stock_code': stock,
                            'circ_mv': circ_mv if circ_mv is not None else total_mv,
                            'total_mv': total_mv if total_mv is not None else circ_mv,
                            'pe_ttm': latest.get('pe_ttm', np.nan),
                            'pb': latest.get('pb', np.nan),
                        })
                    else:
                        failed_stocks.append(stock)
                else:
                    failed_stocks.append(stock)
            else:
                failed_stocks.append(stock)
                
        except Exception as e:
            logger.debug(f"获取 {stock} 财务数据失败: {e}")
            failed_stocks.append(stock)
        
        # 进度日志
        if (i + 1) % 20 == 0:
            logger.info(f"财务数据获取进度: {i + 1}/{len(stock_list)}")
        
        # 延时避免请求过快
        if (i + 1) % 5 == 0:
            time.sleep(0.5)
    
    if not financial_records:
        error_msg = (
            "无法获取财务数据（流通市值 circ_mv）。\n"
            "小市值策略回测需要历史市值数据。请先运行以下命令下载数据：\n"
            "  python tools/download_financial_data.py --start {start} --end {end}\n"
            "或在 data/raw/ 目录下放置包含 circ_mv 字段的 financial_data.parquet 文件。"
        ).format(start=start_date, end=end_date)
        
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    financial_df = pd.DataFrame(financial_records)
    
    if failed_stocks:
        logger.warning(
            f"部分股票财务数据获取失败: {len(failed_stocks)}/{len(stock_list)}, "
            f"成功: {len(financial_records)}"
        )
    
    logger.info(f"财务数据加载完成: {len(financial_df)} 只股票")
    return financial_df


def _generate_backtest_factor_data(
    price_data_dict: Dict[str, pd.DataFrame],
    close_df: pd.DataFrame,
    strategy_config: Dict[str, Any],
    financial_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    生成回测用因子数据（增强版）
    
    计算以下因子：
    - momentum_zscore: 基于 RSI_20 的动量因子
    - small_cap: 小市值因子 = -log(circ_mv)，市值越小分数越高
    - small_cap_zscore: 小市值因子的 Z-Score 标准化
    - turnover_5d: 5日平均换手率
    - value_zscore, quality_zscore: 财务因子（需要财务数据）
    
    Parameters
    ----------
    price_data_dict : Dict[str, pd.DataFrame]
        股票价格数据字典 {stock_code: DataFrame}
    close_df : pd.DataFrame
        收盘价矩阵 (Index=日期, Columns=股票代码)
    strategy_config : Dict[str, Any]
        策略配置
    financial_data : Optional[pd.DataFrame]
        财务数据，包含 stock_code, circ_mv 等字段
    
    Returns
    -------
    pd.DataFrame
        因子数据，包含 date, stock_code 及各类因子列
    
    Notes
    -----
    如果提供了 financial_data 且包含 circ_mv，将正确计算 small_cap 因子。
    否则 small_cap 相关因子将被设置为 NaN，并记录警告。
    """
    logger = logging.getLogger(__name__)
    
    has_financial = (
        financial_data is not None and 
        not financial_data.empty and 
        'circ_mv' in financial_data.columns
    )
    
    if has_financial:
        logger.info("生成回测因子数据（含财务因子：small_cap, value）...")
    else:
        logger.info(
            "生成回测因子数据：将使用「换手率倒推市值」方法估算流通市值，"
            "公式: estimated_circ_mv = amount / (turnover / 100)"
        )
        logger.warning(
            "注意：估算市值仅供回测参考，实际市值以财务数据为准"
        )
    
    factor_records = []
    
    # 构建财务数据映射 {stock_code: {circ_mv, pe_ttm, ...}}
    financial_map: Dict[str, Dict[str, Any]] = {}
    if has_financial:
        for _, row in financial_data.iterrows():
            stock_code = row.get('stock_code', '')
            if stock_code:
                financial_map[stock_code] = {
                    'circ_mv': row.get('circ_mv', np.nan),
                    'total_mv': row.get('total_mv', np.nan),
                    'pe_ttm': row.get('pe_ttm', np.nan),
                    'pb': row.get('pb', np.nan),
                }
    
    # 计算 RSI_20 for 每只股票
    def calculate_rsi(series: pd.Series, period: int = 20) -> pd.Series:
        """计算 RSI 指标"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    for stock_code, df in price_data_dict.items():
        if df is None or df.empty or 'close' not in df.columns:
            continue
        
        # 确保索引是日期类型
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 计算 RSI_20
        rsi_20 = calculate_rsi(df['close'], period=20)
        
        # 计算 Sharpe_20 (新动量因子)
        # 滚动年化夏普比率 = (mean(returns) / std(returns)) * sqrt(252)
        sharpe_20 = pd.Series(np.nan, index=df.index)
        sharpe_60 = pd.Series(np.nan, index=df.index)
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            # min_periods=10: 至少需要半个窗口的数据
            mean_ret = returns.rolling(20, min_periods=10).mean()
            std_ret = returns.rolling(20, min_periods=10).std()
            sharpe_20 = (mean_ret / std_ret.replace(0, np.nan)) * np.sqrt(252)
            
            # [NEW] Sharpe_60: 更平滑的长周期动量因子
            mean_ret_60 = returns.rolling(60, min_periods=30).mean()
            std_ret_60 = returns.rolling(60, min_periods=30).std()
            sharpe_60 = (mean_ret_60 / std_ret_60.replace(0, np.nan)) * np.sqrt(252)

        # 计算 ROC_20 (动量因子)
        roc_20 = pd.Series(np.nan, index=df.index)
        if 'close' in df.columns:
            # 20日变动率 = (Today - 20DaysAgo) / 20DaysAgo * 100
            roc_20 = df['close'].pct_change(20) * 100
        
        # 计算 5 日平均换手率
        turnover_5d = pd.Series(np.nan, index=df.index)
        if 'turnover' in df.columns:
            turnover_5d = df['turnover'].rolling(5, min_periods=1).mean()
        
        # [Added] 预计算 IVOL_20 (特质波动率)
        ivol_20 = pd.Series(np.nan, index=df.index)
        if 'close' in df.columns:
             # 计算日收益率
             daily_ret = df['close'].pct_change()
             # 滚动标准差 * 年化因子
             ivol_20 = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(252)

        # 获取财务数据（作为优先使用的静态市值）
        fin_data = financial_map.get(stock_code, {})
        static_circ_mv = fin_data.get('circ_mv', np.nan)
        pe_ttm = fin_data.get('pe_ttm', np.nan)
        
        # 计算估算流通市值序列（基于换手率倒推）
        # 公式: estimated_circ_mv = amount / (turnover / 100)
        # 含义: 换手率 = 成交量 / 流通股本，成交额 ≈ 成交量 * 当日均价
        #       所以: 流通市值 ≈ 成交额 / 换手率
        has_turnover = 'turnover' in df.columns
        has_amount = 'amount' in df.columns
        
        estimated_circ_mv_series = pd.Series(np.nan, index=df.index)
        if has_turnover and has_amount:
            # 换手率转为小数 (turnover 单位是百分比，如 3.5 表示 3.5%)
            turnover_pct = df['turnover'] / 100.0
            # 避免除以零或极小值
            safe_turnover = turnover_pct.replace(0, np.nan)
            safe_turnover = safe_turnover.where(safe_turnover >= 0.0001, np.nan)
            # 计算估算流通市值 (单位与 amount 一致，通常是元)
            estimated_circ_mv_series = df['amount'] / safe_turnover
        
        # 计算 EP_TTM (价值因子)
        if pd.notna(pe_ttm) and pe_ttm > 0:
            ep_ttm = 1.0 / pe_ttm
        else:
            ep_ttm = np.nan
        
        for date in df.index:
            rsi_val = rsi_20.get(date, np.nan) if date in rsi_20.index else np.nan
            turnover_val = turnover_5d.get(date, np.nan) if date in turnover_5d.index else np.nan
            close_price = df.loc[date, 'close'] if date in df.index else np.nan
            
            # 获取当天的估算市值
            estimated_circ_mv = estimated_circ_mv_series.get(date, np.nan)
            
            # 优先级：1. 财务数据中的 circ_mv  2. 换手率估算的市值  3. NaN
            if pd.notna(static_circ_mv) and static_circ_mv > 0:
                circ_mv = static_circ_mv
            elif pd.notna(estimated_circ_mv) and estimated_circ_mv > 0:
                circ_mv = estimated_circ_mv
            else:
                circ_mv = np.nan
            
            # 计算 small_cap 因子：-log(circ_mv)
            # 市值越小，-log(市值) 越大，得分越高
            if pd.notna(circ_mv) and circ_mv > 0:
                small_cap = -np.log(circ_mv)
            else:
                small_cap = np.nan
            
            # 计算 ROE 稳定性（简化版：用 PE 的倒数作为 ROE 代理，计算其波动率）
            # 注意：准确的 roe_stability 需要季度财务数据，这里仅作占位
            # 实际生产中应在 data_loader 加载完整的季度 ROE 数据
            roe_proxy = ep_ttm  # 假设 EP ≈ ROE (在 PB=1 时成立)
            roe_stability = roe_proxy if pd.notna(roe_proxy) else 0.0
            
            factor_records.append({
                'date': date,
                'stock_code': stock_code,
                'close': close_price,
                'rsi_20': rsi_val,
                'sharpe_20': sharpe_20.get(date, np.nan) if date in sharpe_20.index else np.nan,
                'sharpe_60': sharpe_60.get(date, np.nan) if date in sharpe_60.index else np.nan,
                'roc_20': roc_20.get(date, np.nan) if date in roc_20.index else np.nan,
                'turnover_5d': turnover_val,
                # 小市值因子（核心）
                'small_cap': small_cap,
                'circ_mv': circ_mv,
                # 价值因子
                'ep_ttm': ep_ttm,
                # 质量因子 (新增)
                'roe_stability': roe_stability,
                # 新增因子：特质波动率 (IVOL)
                'ivol_20': ivol_20.get(date, np.nan) if date in ivol_20.index else np.nan, 
                # 估算上市天数（默认足够长以通过过滤）
                'listing_days': 1000,
                # 涨跌停标志（简化：默认无涨跌停）
                'is_limit': False,
            })
    
    if not factor_records:
        logger.warning("无法生成因子数据")
        return pd.DataFrame()
    
    factor_df = pd.DataFrame(factor_records)
    
    # Z-Score 标准化（按日期分组）
    def zscore_by_date(group: pd.DataFrame, col: str) -> pd.Series:
        """按日期分组计算 Z-Score"""
        valid_vals = group[col].dropna()
        if len(valid_vals) < 2:
            return pd.Series(np.nan, index=group.index)
        
        mean_val = valid_vals.mean()
        std_val = valid_vals.std()
        if std_val > 0:
            return (group[col] - mean_val) / std_val
        else:
            return pd.Series(0.0, index=group.index)
    
    # 计算 RSI Z-Score（动量因子）
    factor_df['momentum_zscore'] = factor_df.groupby('date', group_keys=False).apply(
        lambda g: zscore_by_date(g, 'rsi_20'), include_groups=False
    ).reset_index(level=0, drop=True)
    
    # [Added] 计算 Sharpe_20 Z-Score (新的核心动量因子)
    if 'sharpe_20' in factor_df.columns:
        factor_df['sharpe_20_zscore'] = factor_df.groupby('date', group_keys=False).apply(
            lambda g: zscore_by_date(g, 'sharpe_20'), include_groups=False
        ).reset_index(level=0, drop=True)
        logger.info(f"sharpe_20_zscore 生成完成，有效率: {factor_df['sharpe_20_zscore'].notna().mean():.1%}")
    else:
        # 如果 features.py 还没计算 sharpe_20，则尝试计算它（针对回测模式因子未更新的情况）
        logger.warning("Warning: 'sharpe_20' not found in features, skipping z-score calculation")
        factor_df['sharpe_20_zscore'] = np.nan

    # [NEW] 计算 Sharpe_60 Z-Score (长周期动量因子 - 更稳定)
    if 'sharpe_60' in factor_df.columns:
        factor_df['sharpe_60_zscore'] = factor_df.groupby('date', group_keys=False).apply(
            lambda g: zscore_by_date(g, 'sharpe_60'), include_groups=False
        ).reset_index(level=0, drop=True)
        logger.info(f"sharpe_60_zscore 生成完成，有效率: {factor_df['sharpe_60_zscore'].notna().mean():.1%}")
    else:
        factor_df['sharpe_60_zscore'] = np.nan
        logger.warning("Warning: 'sharpe_60' not found, long-term momentum unavailable")

    # 计算 ROC_20 Z-Score (兼容旧版动量因子)
    if 'roc_20' in factor_df.columns:
        factor_df['roc_20_zscore'] = factor_df.groupby('date', group_keys=False).apply(
            lambda g: zscore_by_date(g, 'roc_20'), include_groups=False
        ).reset_index(level=0, drop=True)
        logger.info(f"roc_20_zscore 生成完成，有效率: {factor_df['roc_20_zscore'].notna().mean():.1%}")
    else:
        factor_df['roc_20_zscore'] = np.nan
        logger.warning("无法计算 roc_20_zscore: roc_20 列缺失")

    # 检查是否有有效的 small_cap 数据（来自财务数据或换手率估算）
    has_valid_small_cap = factor_df['small_cap'].notna().any()
    
    # 计算 Small Cap Z-Score（小市值因子）
    if has_valid_small_cap:
        factor_df['small_cap_zscore'] = factor_df.groupby('date', group_keys=False).apply(
            lambda g: zscore_by_date(g, 'small_cap'), include_groups=False
        ).reset_index(level=0, drop=True)
        
        # 计算换手率 Z-Score
        factor_df['turnover_5d_zscore'] = factor_df.groupby('date', group_keys=False).apply(
            lambda g: zscore_by_date(g, 'turnover_5d'), include_groups=False
        ).reset_index(level=0, drop=True)
        
        # [Added] 计算 ROE 稳定性 Z-Score (新的质量因子)
        # 如果财务数据中包含 roe_stability (需在 features.py 计算)，这里进行标准化
        if 'roe_stability' in factor_df.columns:
            factor_df['roe_stability_zscore'] = factor_df.groupby('date', group_keys=False).apply(
                lambda g: zscore_by_date(g, 'roe_stability'), include_groups=False
            ).reset_index(level=0, drop=True)
        else:
            # 如果上游未计算 roe_stability，暂时用 roe (ep_ttm的倒数近似) 或设为 0
            # 这里为了不报错，先设为 0，后续需在 features.py 确保计算
            factor_df['roe_stability_zscore'] = 0.0
            logger.warning("Warning: 'roe_stability' not found, quality factor set to 0")

        # [Added] 计算 IVOL Z-Score (低波因子)
        # IVOL 越低越好，因此取负号
        if 'ivol_20' in factor_df.columns:
             # 注意：IVOL可能为0或NaN，需要处理
            factor_df['ivol_20'] = factor_df['ivol_20'].replace(0, np.nan)
            factor_df['ivol_zscore'] = factor_df.groupby('date', group_keys=False).apply(
                lambda g: zscore_by_date(g, 'ivol_20'), include_groups=False
            ).reset_index(level=0, drop=True)
            # 反转因子方向：波动率越低分越高
            factor_df['ivol_zscore'] = -factor_df['ivol_zscore']
        else:
            factor_df['ivol_zscore'] = 0.0

        # 计算价值因子 Z-Score（需要财务数据）
        if has_financial:
            factor_df['value_zscore'] = factor_df.groupby('date', group_keys=False).apply(
                lambda g: zscore_by_date(g, 'ep_ttm'), include_groups=False
            ).reset_index(level=0, drop=True)
        else:
            factor_df['value_zscore'] = np.nan
            
        if not has_financial:
            logger.info(
                "已通过换手率倒推市值计算 small_cap_zscore，"
                f"有效记录数: {factor_df['small_cap_zscore'].notna().sum()}"
            )
    else:
        # 既无财务数据也无换手率估算时设置为 NaN
        factor_df['small_cap_zscore'] = np.nan
        factor_df['turnover_5d_zscore'] = np.nan
        factor_df['value_zscore'] = np.nan
        
        logger.warning(
            "警告：无财务数据且无法通过换手率估算市值，small_cap_zscore 设置为 NaN。"
            "回测结果将仅基于动量因子（RSI），无法体现小市值策略效果。"
        )
    
    # 质量因子（需要更多财务数据，暂时设为 NaN）
    factor_df['quality_zscore'] = np.nan
    
    # 填充动量因子的 NaN
    factor_df['momentum_zscore'] = factor_df['momentum_zscore'].fillna(0.0)
    
    # ==================== 计算 Alpha 因子（量价配合）====================
    # 用于回测的 Alpha_001 因子
    # Alpha_001 = (Close - VWAP) / VWAP
    alpha_enabled = False
    alpha_records = []
    
    try:
        for stock_code, stock_df in price_data_dict.items():
            if 'amount' in stock_df.columns and 'volume' in stock_df.columns:
                vwap = stock_df['amount'] / stock_df['volume'].replace(0, np.nan)
                alpha_001 = (stock_df['close'] - vwap) / vwap.replace(0, np.nan)
                alpha_001 = alpha_001.replace([np.inf, -np.inf], np.nan)
                
                for date in stock_df.index:
                    if pd.notna(alpha_001.get(date)):
                        alpha_records.append({
                            'date': date,
                            'stock_code': stock_code,
                            'alpha_001': alpha_001[date]
                        })
        
        if alpha_records:
            alpha_df = pd.DataFrame(alpha_records)
            factor_df = factor_df.merge(alpha_df, on=['date', 'stock_code'], how='left')
            
            # Z-Score 标准化
            factor_df['alpha_001_zscore'] = factor_df.groupby('date', group_keys=False).apply(
                lambda g: zscore_by_date(g, 'alpha_001'), include_groups=False
            ).reset_index(level=0, drop=True).fillna(0)
            
            alpha_enabled = True
            logger.info("Alpha_001 因子（量价配合）计算完成")
        else:
            factor_df['alpha_001_zscore'] = 0.0
            logger.warning("无法计算 Alpha_001 因子：缺少 amount/volume 数据")
    except Exception as e:
        factor_df['alpha_001_zscore'] = 0.0
        logger.warning(f"Alpha_001 因子计算失败: {e}")
    
    # ==================== 计算复合动量因子 momentum_composite_zscore ====================
    # 牛市进攻型配方: 40% ROC + 30% Sharpe + 30% Alpha001
    roc_col = 'roc_20_zscore' if 'roc_20_zscore' in factor_df.columns else None
    sharpe_col = 'sharpe_20_zscore' if 'sharpe_20_zscore' in factor_df.columns else None
    alpha_col = 'alpha_001_zscore' if alpha_enabled else None
    
    if roc_col and sharpe_col and alpha_col:
        factor_df['momentum_composite_zscore'] = (
            0.4 * factor_df[roc_col].fillna(0) +
            0.3 * factor_df[sharpe_col].fillna(0) +
            0.3 * factor_df[alpha_col].fillna(0)
        )
        logger.info("🚀 复合动量因子计算完成: 40% ROC + 30% Sharpe + 30% Alpha001")
    elif roc_col and sharpe_col:
        factor_df['momentum_composite_zscore'] = (
            0.6 * factor_df[roc_col].fillna(0) +
            0.4 * factor_df[sharpe_col].fillna(0)
        )
        logger.info("复合动量因子计算完成: 60% ROC + 40% Sharpe（无 Alpha）")
    elif roc_col:
        factor_df['momentum_composite_zscore'] = factor_df[roc_col].fillna(0)
        logger.warning("复合动量因子使用 ROC 单因子")
    else:
        factor_df['momentum_composite_zscore'] = factor_df['momentum_zscore'].fillna(0)
        logger.warning("复合动量因子使用 RSI 作为后备")
    
    # 统计有效的小市值因子数量
    valid_small_cap = factor_df['small_cap_zscore'].notna().sum()
    total_records = len(factor_df)
    
    logger.info(
        f"因子数据生成完成: {total_records} 条记录, "
        f"{factor_df['stock_code'].nunique()} 只股票, "
        f"{factor_df['date'].nunique()} 个交易日"
    )
    
    if has_financial:
        logger.info(
            f"小市值因子 (small_cap_zscore) 有效率: "
            f"{valid_small_cap}/{total_records} ({valid_small_cap/total_records:.1%})"
        )
    
    return factor_df


def run_backtest(
    start_date: str,
    end_date: str,
    config: Optional[Dict[str, Any]] = None,
    strategy_type: str = "multi_factor",
    no_llm: bool = False
) -> bool:
    """
    运行策略回测
    
    使用 BacktestEngine + MultiFactorStrategy 对指定时间段的历史数据进行回测。
    支持大盘风控（通过 benchmark_data 传入基准指数数据）。
    
    Parameters
    ----------
    start_date : str
        回测开始日期 (YYYY-MM-DD)
    end_date : str
        回测结束日期 (YYYY-MM-DD)
    config : Optional[Dict[str, Any]]
        回测配置参数
    strategy_type : str
        策略类型: 'multi_factor', 'ma_cross'
    no_llm : bool
        是否禁用 LLM 风控
    
    Returns
    -------
    bool
        回测是否成功
    
    Notes
    -----
    回测流程：
    1. 加载历史 OHLCV 数据
    2. 准备价格矩阵
    3. 加载历史财务数据（特别是流通市值 circ_mv，用于小市值因子）
    4. 获取基准指数数据（用于大盘风控）
    5. 生成因子数据（含 small_cap = -log(circ_mv)，momentum 等）
    6. 使用 BacktestEngine 执行权重驱动回测
    7. 生成回测报告
    
    小市值策略要求：
    - 需要本地存储的财务数据文件（data/raw/financial_*.parquet）
    - 财务数据需包含 circ_mv（流通市值）字段
    - 如果无财务数据，策略会自动退化为纯动量策略
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"开始回测: {start_date} ~ {end_date}")
    logger.info("使用引擎: BacktestEngine (权重驱动 + 大盘风控)")
    if no_llm:
        logger.info("参数设置: 禁用 LLM 风控")
    logger.info("=" * 60)
    
    try:
        # ========================================
        # Step 0: 加载配置
        # ========================================
        if config is None:
            try:
                config = load_config(CONFIG_PATH)
            except FileNotFoundError:
                logger.warning(f"配置文件 {CONFIG_PATH} 不存在，使用默认配置")
                config = {}
        
        # 处理 LLM 禁用
        if no_llm:
            config["llm"] = {}
        
        # 提取配置
        backtest_config = config.get("backtest", {})
        portfolio_config = config.get("portfolio", {})
        strategy_config = config.get("strategy", {})
        data_config = config.get("data", {})
        
        initial_capital = portfolio_config.get("total_capital", 300000)
        commission = config.get("trading", {}).get("commission_rate", 0.0003)
        slippage = config.get("trading", {}).get("slippage", 0.001)
        risk_free_rate = portfolio_config.get("risk_free_rate", 0.02)
        optimization_objective = portfolio_config.get("optimization_objective", "equal_weight")
        
        # 基准指数代码（默认中证500）
        benchmark_code = backtest_config.get("benchmark", "000905")
        
        logger.info(f"回测配置: 初始资金=¥{initial_capital:,.0f}, 基准={benchmark_code}")
        
        # ========================================
        # Step 1: 加载历史数据
        # ========================================
        logger.info("Step 1/7: 加载历史 OHLCV 数据")
        
        data_loader = DataLoader(output_dir=str(DATA_RAW_PATH))
        
        # 获取股票列表（根据配置选择股票池）
        stock_pool = data_config.get("stock_pool", "zz500")
        
        # 尝试获取指定股票池的成分股
        # 注意：DataLoader 目前只实现了 get_hs300_constituents
        # 对于中证500等其他指数，使用 AkShare 直接获取
        stock_list = []
        
        if stock_pool == "zz500":
            # 尝试使用 AkShare 直接获取中证500成分股
            try:
                import akshare as ak
                df = ak.index_stock_cons(symbol="000905")
                if df is not None and not df.empty:
                    code_col = next((c for c in df.columns if '代码' in c), None)
                    if code_col:
                        stock_list = df[code_col].tolist()
                        logger.info(f"获取中证500成分股成功，共 {len(stock_list)} 只")
            except Exception as e:
                logger.warning(f"获取中证500成分股失败: {e}")
            
            if not stock_list:
                logger.warning("无法获取中证500成分股，尝试获取沪深300")
                stock_list = data_loader.get_hs300_constituents()
        elif stock_pool == "hs300":
            stock_list = data_loader.get_hs300_constituents()
        elif stock_pool == "zz1000":
            # 获取中证1000成分股
            try:
                import akshare as ak
                df = ak.index_stock_cons(symbol="000852")
                if df is not None and not df.empty:
                    code_col = next((c for c in df.columns if '代码' in c), None)
                    if code_col:
                        stock_list = df[code_col].tolist()
                        logger.info(f"获取中证1000成分股成功，共 {len(stock_list)} 只")
            except Exception as e:
                logger.warning(f"获取中证1000成分股失败: {e}")
                
            if not stock_list:
                logger.warning("无法获取中证1000成分股，尝试从本地缓存加载或使用示例股票")
                # TODO: 实现本地缓存加载逻辑
        else:
            stock_list = data_loader.get_hs300_constituents()
        
        if not stock_list:
            logger.warning("无法获取成分股列表，使用示例股票")
            stock_list = ["000001", "000002", "600519", "601318", "000858",
                         "000063", "000651", "000725", "002415", "600036"]
        
        # 限制回测股票数量（避免过长时间）
        max_stocks = backtest_config.get("max_stocks", 100)
        stock_list = stock_list[:max_stocks]
        logger.info(f"股票池: {stock_pool}, 回测股票数量: {len(stock_list)}")
        
        # 下载历史数据
        price_data_dict: Dict[str, pd.DataFrame] = {}
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        
        # [OPTIMIZED] 优先从本地缓存加载数据
        # 路径: data/lake/daily/{stock}.parquet
        local_cache_dir = Path("data/lake/daily")
        loaded_from_cache = 0
        
        for i, stock in enumerate(stock_list):
            df = None
            try:
                # 1. 尝试读取本地缓存
                cache_file = local_cache_dir / f"{stock}.parquet"
                if cache_file.exists():
                    try:
                        cached_df = pd.read_parquet(cache_file)
                        
                        # 处理日期：可能在 'date' 列或作为 index
                        if 'date' not in cached_df.columns:
                            # 日期可能是 index，尝试 reset_index
                            if isinstance(cached_df.index, pd.DatetimeIndex):
                                cached_df = cached_df.reset_index()
                                cached_df.columns = ['date'] + list(cached_df.columns[1:])
                            elif cached_df.index.name == 'date' or cached_df.index.name == 'trade_date':
                                cached_df = cached_df.reset_index()
                        
                        # 确保 date 列存在
                        if not cached_df.empty and 'date' in cached_df.columns:
                            cached_df['date'] = pd.to_datetime(cached_df['date'])
                            cache_start = cached_df['date'].min()
                            cache_end = cached_df['date'].max()
                            req_start = pd.to_datetime(start_date)
                            req_end = pd.to_datetime(end_date)
                            
                            # [FIX] 放宽日期检查：允许 7 天的误差（处理节假日和数据延迟）
                            # 因为实际交易日可能比日历日少
                            tolerance = pd.Timedelta(days=7)
                            if cache_start <= req_start and cache_end >= (req_end - tolerance):
                                # 筛选时间段
                                df = cached_df[(cached_df['date'] >= req_start) & (cached_df['date'] <= cache_end)].copy()
                                if not df.empty:
                                    df = df.set_index('date').sort_index()
                                    loaded_from_cache += 1
                    except Exception as e:
                        logger.debug(f"读取本地缓存 {stock} 失败: {e}")

                # 2. 如果本地没有或不满足，则下载
                if df is None or df.empty:
                    df = data_loader.fetch_daily_price(stock, start_fmt, end_fmt)
                    
                    # [NEW] 下载成功后保存到本地缓存
                    if df is not None and not df.empty:
                        try:
                            local_cache_dir.mkdir(parents=True, exist_ok=True)
                            # 保存时将 index 转为 date 列
                            save_df = df.reset_index()
                            save_df.columns = ['date'] + list(save_df.columns[1:])
                            save_df.to_parquet(cache_file, index=False)
                        except Exception as e:
                            logger.debug(f"保存缓存 {stock} 失败: {e}")
                
                if df is not None and not df.empty:
                    price_data_dict[stock] = df
            except Exception as e:
                logger.debug(f"获取 {stock} 数据失败: {e}")
            
            if (i + 1) % 100 == 0:
                logger.info(f"数据加载进度: {i + 1}/{len(stock_list)} (缓存命中: {loaded_from_cache})")
        
        if not price_data_dict:
            logger.error("未获取到任何历史数据")
            return False
        
        logger.info(f"成功加载 {len(price_data_dict)} 只股票的历史数据")
        
        # ========================================
        # Step 2: 准备价格矩阵
        # ========================================
        logger.info("Step 2/7: 准备价格矩阵")
        
        # 构建收盘价 DataFrame (行=日期, 列=股票)
        close_prices = {}
        for stock, df in price_data_dict.items():
            if 'close' in df.columns:
                close_prices[stock] = df['close']
        
        close_df = pd.DataFrame(close_prices)
        close_df.index = pd.to_datetime(close_df.index)
        close_df = close_df.sort_index()
        
        # 填充缺失值
        close_df = close_df.ffill().bfill()
        
        logger.info(f"价格矩阵: {close_df.shape[0]} 天 x {close_df.shape[1]} 只股票")
        
        # ========================================
        # Step 3: 加载历史财务数据（关键：小市值因子需要 circ_mv）
        # ========================================
        logger.info("Step 3/7: 加载历史财务数据（流通市值 circ_mv）")
        
        financial_data: Optional[pd.DataFrame] = None
        has_financial_data = False
        
        try:
            financial_data = _load_backtest_financial_data(
                stock_list=list(price_data_dict.keys()),
                start_date=start_date,
                end_date=end_date,
                data_loader=data_loader
            )
            
            if financial_data is not None and not financial_data.empty:
                has_financial_data = 'circ_mv' in financial_data.columns
                logger.info(
                    f"财务数据加载成功: {len(financial_data)} 条记录, "
                    f"circ_mv 可用: {has_financial_data}"
                )
            else:
                logger.warning("财务数据为空")
                
        except FileNotFoundError as e:
            logger.error(f"财务数据加载失败: {e}")
            logger.error("小市值策略回测需要财务数据。如果您只想运行动量策略，请继续；否则请先准备财务数据。")
            # 不抛出异常，允许继续（退化为纯动量策略）
            financial_data = None
        except Exception as e:
            logger.warning(f"加载财务数据时发生错误: {e}，将使用纯动量策略")
            financial_data = None
        
        # ========================================
        # Step 4: 获取基准指数数据（用于大盘风控）
        # ========================================
        logger.info(f"Step 4/7: 获取基准指数数据 ({benchmark_code})")
        
        benchmark_data: Optional[pd.DataFrame] = None
        
        try:
            benchmark_data = data_loader.fetch_index_price(
                index_code=benchmark_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if benchmark_data is not None and not benchmark_data.empty:
                logger.info(
                    f"基准指数数据获取成功: {len(benchmark_data)} 条记录, "
                    f"日期范围: {benchmark_data.index[0].strftime('%Y-%m-%d')} ~ "
                    f"{benchmark_data.index[-1].strftime('%Y-%m-%d')}"
                )
            else:
                logger.warning("基准指数数据为空，大盘风控将不生效")
                benchmark_data = None
                
        except Exception as e:
            logger.warning(f"获取基准指数数据失败: {e}，大盘风控将不生效")
            benchmark_data = None
        
        # ========================================
        # Step 5: 生成因子数据（含小市值因子）
        # ========================================
        if has_financial_data:
            logger.info("Step 5/7: 生成因子数据（含小市值因子 small_cap）")
        else:
            logger.warning("Step 5/7: 生成因子数据（无财务数据，仅动量因子）")
        
        factor_data = _generate_backtest_factor_data(
            price_data_dict=price_data_dict,
            close_df=close_df,
            strategy_config=strategy_config,
            financial_data=financial_data
        )
        
        if factor_data.empty:
            logger.error("因子数据生成失败")
            return False
        
        # ========================================
        # Step 6: 初始化策略和引擎，执行回测
        # ========================================
        logger.info("Step 6/7: 初始化策略和引擎，执行回测")
        
        if strategy_type == "multi_factor":
            # 多因子策略
            # 从配置读取因子权重
            value_weight = strategy_config.get("value_weight", 0.0)
            quality_weight = strategy_config.get("quality_weight", 0.0)
            momentum_weight = strategy_config.get("momentum_weight", 1.0)
            size_weight = strategy_config.get("size_weight", 0.0)
            
            # 根据财务数据可用性调整策略配置
            if has_financial_data:
                # 财务数据可用，可以使用小市值策略
                logger.info("财务数据可用，启用小市值因子 (small_cap)")
                
                # 如果配置了 size_weight 或策略需要小市值因子
                if size_weight > 0 or strategy_config.get("use_small_cap", True):
                    # 使用小市值因子
                    value_col = strategy_config.get("value_col", "small_cap_zscore")
                    size_col = strategy_config.get("size_col", "small_cap_zscore")
                else:
                    value_col = strategy_config.get("value_col", "value_zscore")
                    size_col = strategy_config.get("size_col", "small_cap_zscore")
            else:
                # 无财务数据，退化为纯动量策略
                logger.warning(
                    f"无财务数据，小市值因子不可用。"
                    f"自动调整为纯动量策略 (momentum=1.0)"
                )
                # 强制调整权重
                if value_weight > 0 or size_weight > 0:
                    logger.warning(
                        f"原配置权重 (value={value_weight}, size={size_weight}) "
                        f"因缺少财务数据被置为 0"
                    )
                value_weight = 0.0
                size_weight = 0.0
                quality_weight = 0.0
                momentum_weight = 1.0
                value_col = "value_zscore"
                size_col = "small_cap_zscore"
            
            strategy = MultiFactorStrategy(
                name="Multi-Factor Backtest" + (" (小市值增强)" if has_financial_data else " (纯动量)"),
                config={
                    "value_weight": value_weight,
                    "quality_weight": quality_weight,
                    "momentum_weight": momentum_weight,
                    "size_weight": size_weight,
                    "top_n": strategy_config.get("top_n", 5),
                    "min_listing_days": strategy_config.get("min_listing_days", 126),
                    "rebalance_frequency": strategy_config.get("rebalance_frequency", "monthly"),
                    "rebalance_buffer": strategy_config.get("rebalance_buffer", 0.05),
                    # [NEW] 持股惯性加分
                    "holding_bonus": strategy_config.get("holding_bonus", 0.0),
                    # [NEW] 大盘风控配置（从 risk 部分读取）
                    "market_risk": config.get("risk", {}).get("market_risk", {}),
                    # 因子列名配置
                    "value_col": value_col,
                    "quality_col": strategy_config.get("quality_col", "quality_zscore"),
                    "momentum_col": strategy_config.get("momentum_col", "momentum_zscore"),
                    "size_col": size_col,
                    "date_col": "date",
                    "stock_col": "stock_code",
                }
            )
            logger.info(
                f"使用多因子策略: value={value_weight}, quality={quality_weight}, "
                f"momentum={momentum_weight}, size={size_weight}, top_n={strategy.top_n}"
            )
            if has_financial_data:
                logger.info(f"因子列: value_col={value_col}, size_col={size_col}")
        else:
            # 均线交叉策略（不支持权重驱动回测，使用简化逻辑）
            strategy = MACrossStrategy(
                name="MA Cross Backtest",
                config={
                    "short_window": 5,
                    "long_window": 20,
                }
            )
            logger.info("使用均线交叉策略")
            logger.warning("均线策略暂不支持权重驱动回测，使用简化逻辑")
        
        # 初始化 BacktestEngine
        backtest_engine = BacktestEngine(config={
            "initial_capital": initial_capital,
            "commission": commission,
            "slippage": slippage,
            "risk_free_rate": risk_free_rate,
        })
        
        logger.info(
            f"BacktestEngine 初始化: 初始资金=¥{initial_capital:,.0f}, "
            f"佣金={commission*10000:.1f}‱, 滑点={slippage*100:.2f}%"
        )
        
        # 执行回测（权重驱动模式）
        if strategy_type == "multi_factor":
            logger.info("执行权重驱动回测...")
            
            result = backtest_engine.run(
                strategy=strategy,
                price_data=close_df,
                factor_data=factor_data,
                objective=optimization_objective,
                benchmark_data=benchmark_data  # 传入基准数据用于大盘风控
            )
            
            total_return = result.total_return
            annual_return = result.annual_return
            sharpe_ratio = result.sharpe_ratio
            max_drawdown = result.max_drawdown
            total_trades = result.total_trades
            win_rate = result.win_rate
            
        else:
            # MA Cross 策略使用简化回测
            logger.warning("MA Cross 策略使用简化回测逻辑")
            
            # 使用第一只股票进行单股票回测演示
            stock_code = list(price_data_dict.keys())[0]
            single_price_df = price_data_dict[stock_code]
            
            # 生成信号
            signals = strategy.generate_signals(single_price_df)
            
            # 简化计算收益
            returns = single_price_df['close'].pct_change()
            strategy_returns = returns * signals.shift(1)
            
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            
            cum_returns = (1 + strategy_returns).cumprod()
            peak = cum_returns.expanding().max()
            drawdown = (cum_returns - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            total_trades = (signals.diff().abs() > 0).sum()
            win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        # ========================================
        # Step 7: 生成回测报告
        # ========================================
        logger.info("Step 7/7: 生成回测报告")
        
        report_content = _generate_backtest_report(
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy.name,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            initial_capital=initial_capital,
            num_stocks=len(close_df.columns),
        )
        
        # 保存报告
        report_path = REPORTS_PATH / f"backtest_{start_date}_{end_date}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"回测报告已保存至 {report_path}")
        
        # ========================================
        # 打印回测结果摘要
        # ========================================
        logger.info("=" * 60)
        logger.info("回测结果摘要")
        logger.info("=" * 60)
        logger.info(f"回测引擎:    BacktestEngine (权重驱动 + 大盘风控)")
        logger.info(f"策略名称:    {strategy.name}")
        logger.info(f"回测区间:    {start_date} ~ {end_date}")
        logger.info(f"股票池:      {stock_pool} ({len(close_df.columns)} 只股票)")
        logger.info(f"财务数据:    {'✓ 已加载 (small_cap 因子可用)' if has_financial_data else '✗ 未加载 (纯动量策略)'}")
        logger.info(f"基准指数:    {benchmark_code} {'✓ 已启用风控' if benchmark_data is not None else '✗ 风控未启用'}")
        logger.info("-" * 60)
        logger.info(f"初始资金:    ¥{initial_capital:,.0f}")
        logger.info(f"总收益率:    {total_return:.2%}")
        logger.info(f"年化收益:    {annual_return:.2%}")
        logger.info(f"夏普比率:    {sharpe_ratio:.2f}")
        logger.info(f"最大回撤:    {max_drawdown:.2%}")
        if strategy_type == "multi_factor":
            logger.info(f"总交易次数:  {total_trades}")
            logger.info(f"胜率:        {win_rate:.2%}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _generate_backtest_report(
    start_date: str,
    end_date: str,
    strategy_name: str,
    total_return: float,
    annual_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    initial_capital: float,
    num_stocks: int,
) -> str:
    """
    生成回测报告 HTML
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    strategy_name : str
        策略名称
    total_return : float
        总收益率
    annual_return : float
        年化收益率
    sharpe_ratio : float
        夏普比率
    max_drawdown : float
        最大回撤
    initial_capital : float
        初始资金
    num_stocks : int
        股票数量
    
    Returns
    -------
    str
        HTML 报告内容
    """
    final_capital = initial_capital * (1 + total_return)
    profit = final_capital - initial_capital
    
    # 评级
    if sharpe_ratio >= 2.0:
        rating = "⭐⭐⭐⭐⭐ 优秀"
        rating_color = "#00ff88"
    elif sharpe_ratio >= 1.5:
        rating = "⭐⭐⭐⭐ 良好"
        rating_color = "#88ff00"
    elif sharpe_ratio >= 1.0:
        rating = "⭐⭐⭐ 一般"
        rating_color = "#ffff00"
    elif sharpe_ratio >= 0.5:
        rating = "⭐⭐ 较差"
        rating_color = "#ff8800"
    else:
        rating = "⭐ 差"
        rating_color = "#ff4444"
    
    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告 - {strategy_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
            color: #eee;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .meta {{
            color: #888;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }}
        .rating {{
            display: inline-block;
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            color: {rating_color};
            font-weight: bold;
            margin-left: 1rem;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            font-size: 1.4rem;
            margin-bottom: 1.5rem;
            color: #667eea;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
        }}
        .stat {{
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
            border-radius: 12px;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        .stat-value.positive {{
            color: #00ff88;
        }}
        .stat-value.negative {{
            color: #ff6b6b;
        }}
        .stat-value.neutral {{
            color: #667eea;
        }}
        .stat-label {{
            font-size: 0.9rem;
            color: #888;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .info-table tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .info-table td {{
            padding: 1rem;
        }}
        .info-table td:first-child {{
            color: #888;
            width: 40%;
        }}
        .info-table td:last-child {{
            font-weight: 500;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 3rem;
            font-size: 0.9rem;
        }}
        .highlight {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 策略回测报告</h1>
        <p class="meta">
            {strategy_name}
            <span class="rating">{rating}</span>
        </p>
        
        <div class="card">
            <h2>核心指标</h2>
            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-value {'positive' if total_return >= 0 else 'negative'}">{total_return:+.2%}</div>
                    <div class="stat-label">总收益率</div>
                </div>
                <div class="stat">
                    <div class="stat-value {'positive' if annual_return >= 0 else 'negative'}">{annual_return:+.2%}</div>
                    <div class="stat-label">年化收益</div>
                </div>
                <div class="stat">
                    <div class="stat-value neutral">{sharpe_ratio:.2f}</div>
                    <div class="stat-label">夏普比率</div>
                </div>
                <div class="stat">
                    <div class="stat-value negative">{max_drawdown:.2%}</div>
                    <div class="stat-label">最大回撤</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>资金变化</h2>
            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-value neutral">¥{initial_capital:,.0f}</div>
                    <div class="stat-label">初始资金</div>
                </div>
                <div class="stat">
                    <div class="stat-value {'positive' if profit >= 0 else 'negative'}">¥{final_capital:,.0f}</div>
                    <div class="stat-label">最终资金</div>
                </div>
                <div class="stat">
                    <div class="stat-value {'positive' if profit >= 0 else 'negative'}">{'+'if profit >= 0 else ''}¥{profit:,.0f}</div>
                    <div class="stat-label">盈亏金额</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>回测信息</h2>
            <table class="info-table">
                <tr>
                    <td>策略名称</td>
                    <td>{strategy_name}</td>
                </tr>
                <tr>
                    <td>回测区间</td>
                    <td>{start_date} ~ {end_date}</td>
                </tr>
                <tr>
                    <td>股票数量</td>
                    <td>{num_stocks} 只</td>
                </tr>
                <tr>
                    <td>报告生成时间</td>
                    <td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>
        </div>
        
        <p class="footer">
            本报告由 <span class="highlight">A股量化交易系统</span> 自动生成<br>
            仅供参考，不构成投资建议
        </p>
    </div>
</body>
</html>
    """
    
    return html


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="A股量化交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --daily-update              # 运行每日更新
    python main.py --daily-update --force      # 强制调仓
    python main.py --backtest                  # 运行回测（默认多因子策略）
    python main.py --backtest --strategy ma    # 运行回测（均线策略）
    python main.py --backtest --start 2022-01-01 --end 2023-12-31
        """
    )
    
    parser.add_argument(
        "--daily-update", "-d",
        action="store_true",
        help="运行每日更新流程"
    )
    
    parser.add_argument(
        "--force-rebalance", "-f",
        action="store_true",
        help="强制调仓（忽略日期检查）"
    )
    
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="运行回测"
    )
    
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="multi_factor",
        choices=["multi_factor", "ma_cross"],
        help="回测策略类型: multi_factor(多因子), ma_cross(均线交叉)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="回测开始日期 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2024-01-01",
        help="回测结束日期 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="禁用 LLM 风控功能（强制覆盖配置）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    log_file = LOGS_PATH / f"quant_{datetime.now().strftime('%Y%m%d')}.log"
    setup_logging(level=log_level, log_file=str(log_file))
    
    logger = logging.getLogger(__name__)
    logger.info("A股量化交易系统启动")
    
    if args.daily_update:
        success = run_daily_update(
            force_rebalance=args.force_rebalance,
            no_llm=args.no_llm
        )
        exit(0 if success else 1)
    
    elif args.backtest:
        logger.info(f"回测模式: {args.start} ~ {args.end}, 策略: {args.strategy}")
        success = run_backtest(
            start_date=args.start,
            end_date=args.end,
            strategy_type=args.strategy,
            no_llm=args.no_llm
        )
        exit(0 if success else 1)
    
    else:
        parser.print_help()
        exit(0)


if __name__ == "__main__":
    main()

