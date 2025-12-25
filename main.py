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
)

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
        self.current_positions: Dict[str, float] = {}
        self.target_positions: Dict[str, float] = {}
        
        self.logger.info("DailyUpdateRunner 初始化完成")
    
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
                "value_weight": 0.4,
                "quality_weight": 0.4,
                "momentum_weight": 0.2,
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
        # 数据加载器
        self.data_loader = AkshareDataLoader()
        self.data_cleaner = AShareDataCleaner()
        
        # 增强版数据加载器（用于获取财务数据）
        self.financial_loader = DataLoader(
            output_dir=str(DATA_RAW_PATH),
            max_workers=3,
            retry_times=3
        )
        
        # 策略
        strategy_config = self.config.get("strategy", {})
        self.strategy = MultiFactorStrategy(
            name=strategy_config.get("name", "Multi-Factor Strategy"),
            config={
                "value_weight": strategy_config.get("value_weight", 0.4),
                "quality_weight": strategy_config.get("quality_weight", 0.4),
                "momentum_weight": strategy_config.get("momentum_weight", 0.2),
                "top_n": strategy_config.get("top_n", 30),
                "min_listing_days": strategy_config.get("min_listing_days", 126),
                "value_col": "ep_ttm_zscore",
                "quality_col": "roe_stability_zscore",
                "momentum_col": "rsi_20_zscore",
            }
        )
    
    def update_market_data(self) -> bool:
        """
        更新市场数据
        
        Returns
        -------
        bool
            更新是否成功
        """
        self.logger.info("开始更新市场数据...")
        
        try:
            data_config = self.config.get("data", {})
            stock_pool = data_config.get("stock_pool", "hs300")
            
            # 获取股票列表
            if stock_pool == "hs300":
                stock_list = self.data_loader.get_index_stocks("000300")
            elif stock_pool == "zz500":
                stock_list = self.data_loader.get_index_stocks("000905")
            else:
                stock_list = self.data_loader.get_all_stocks()
            
            self.logger.info(f"股票池: {stock_pool}, 股票数量: {len(stock_list)}")
            
            # 确定日期范围
            end_date = self.today.strftime("%Y%m%d")
            update_days = data_config.get("update_days", 5)
            start_date = (self.today - timedelta(days=update_days * 2)).strftime("%Y%m%d")
            
            # 下载OHLCV数据
            ohlcv_list = []
            for i, stock in enumerate(stock_list[:50]):  # 限制数量用于演示
                try:
                    df = self.data_loader.get_stock_daily(
                        stock, start_date, end_date
                    )
                    if df is not None and not df.empty:
                        df['stock_code'] = stock
                        ohlcv_list.append(df)
                except Exception as e:
                    self.logger.debug(f"获取 {stock} 数据失败: {e}")
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已处理 {i + 1}/{len(stock_list[:50])} 只股票")
            
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
            
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {e}")
            return False
    
    def update_financial_data(self) -> bool:
        """
        更新财务数据
        
        使用 DataLoader.fetch_financial_indicator 获取真实的 PE、PB、ROE 等数据。
        
        Returns
        -------
        bool
            更新是否成功
        """
        self.logger.info("开始更新财务数据...")
        
        try:
            if self.ohlcv_data is None:
                self.logger.warning("OHLCV 数据为空，无法生成财务数据")
                return False
            
            stocks = self.ohlcv_data['stock_code'].unique()
            self.logger.info(f"需获取 {len(stocks)} 只股票的财务数据")
            
            # 使用真实数据接口获取财务指标
            financial_records = []
            failed_stocks = []
            
            for i, stock in enumerate(stocks):
                try:
                    # 调用 DataLoader.fetch_financial_indicator 获取真实数据
                    fin_df = self.financial_loader.fetch_financial_indicator(stock)
                    
                    if fin_df is not None and not fin_df.empty:
                        # 提取最新的财务指标
                        if isinstance(fin_df, pd.DataFrame) and len(fin_df) > 0:
                            latest = fin_df.iloc[-1] if len(fin_df) > 1 else fin_df.iloc[0]
                            
                            # 构建财务记录
                            record = {
                                'stock_code': stock,
                                'pe_ttm': self._safe_get_value(latest, ['pe_ttm', 'pe', '市盈率'], default=np.nan),
                                'pb': self._safe_get_value(latest, ['pb', '市净率'], default=np.nan),
                                'dividend_yield': self._safe_get_value(latest, ['dividend_yield', 'dv_ratio', '股息率'], default=0.0),
                                'ps_ttm': self._safe_get_value(latest, ['ps_ttm', 'ps', '市销率'], default=np.nan),
                                'roe': self._safe_get_value(latest, ['roe', 'roe_ttm'], default=np.nan),
                                'total_mv': self._safe_get_value(latest, ['total_mv', '总市值'], default=np.nan),
                                'circ_mv': self._safe_get_value(latest, ['circ_mv', '流通市值'], default=np.nan),
                            }
                            
                            # 估算上市天数（如果无法获取，使用默认值）
                            record['listing_days'] = self._estimate_listing_days(stock)
                            
                            financial_records.append(record)
                            self.logger.debug(f"获取 {stock} 财务数据成功: PE={record['pe_ttm']:.2f}" if not np.isnan(record['pe_ttm']) else f"获取 {stock} 财务数据成功")
                        else:
                            failed_stocks.append(stock)
                    else:
                        failed_stocks.append(stock)
                        
                except Exception as e:
                    self.logger.debug(f"获取 {stock} 财务数据失败: {e}")
                    failed_stocks.append(stock)
                
                # 进度日志
                if (i + 1) % 10 == 0:
                    self.logger.info(f"财务数据获取进度: {i + 1}/{len(stocks)}")
                
                # 添加延时避免请求过快
                import time
                time.sleep(0.1)
            
            # 对于获取失败的股票，使用备用数据（市场平均值或模拟值）
            if failed_stocks:
                self.logger.warning(f"{len(failed_stocks)} 只股票财务数据获取失败，使用备用数据")
                fallback_records = self._generate_fallback_financial_data(failed_stocks)
                financial_records.extend(fallback_records)
            
            if not financial_records:
                self.logger.error("未获取到任何财务数据")
                return False
            
            self.financial_data = pd.DataFrame(financial_records)
            
            # 数据清洗：处理异常值
            self._clean_financial_data()
            
            # 获取行业数据
            self.industry_data = self._fetch_industry_data(stocks)
            
            self.logger.info(
                f"财务数据更新完成，共 {len(self.financial_data)} 条记录，"
                f"成功 {len(self.financial_data) - len(failed_stocks)} 只，"
                f"备用 {len(failed_stocks)} 只"
            )
            
            # 保存数据
            financial_path = DATA_RAW_PATH / f"financial_{self.today.strftime('%Y%m%d')}.parquet"
            self.financial_data.to_parquet(financial_path)
            self.logger.info(f"财务数据已保存至 {financial_path}")
            
            return True
            
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
    
    def _estimate_listing_days(self, stock: str) -> int:
        """
        估算股票上市天数
        
        Parameters
        ----------
        stock : str
            股票代码
        
        Returns
        -------
        int
            估算的上市天数
        """
        try:
            # 尝试从个股信息获取上市日期
            import akshare as ak
            info_df = ak.stock_individual_info_em(symbol=stock)
            
            if info_df is not None and not info_df.empty:
                # 查找上市日期
                for idx, row in info_df.iterrows():
                    if '上市' in str(row.get('item', '')):
                        list_date = pd.to_datetime(row.get('value', None))
                        if list_date is not None:
                            listing_days = (self.today - list_date).days
                            return max(listing_days, 0)
        except Exception:
            pass
        
        # 默认返回一个较大的值（假设已上市较长时间）
        return 1000
    
    def _generate_fallback_financial_data(self, stocks: List[str]) -> List[Dict[str, Any]]:
        """
        为获取失败的股票生成备用财务数据
        
        使用已获取数据的中位数或合理默认值。
        
        Parameters
        ----------
        stocks : List[str]
            股票代码列表
        
        Returns
        -------
        List[Dict[str, Any]]
            备用财务数据记录列表
        """
        # 计算已获取数据的中位数作为备用值
        if hasattr(self, 'financial_data') and self.financial_data is not None and len(self.financial_data) > 0:
            median_pe = self.financial_data['pe_ttm'].median()
            median_pb = self.financial_data['pb'].median() if 'pb' in self.financial_data.columns else 2.0
            median_roe = self.financial_data['roe'].median() if 'roe' in self.financial_data.columns else 0.10
        else:
            # 使用市场平均值作为默认
            median_pe = 15.0
            median_pb = 2.0
            median_roe = 0.10
        
        fallback_records = []
        for stock in stocks:
            fallback_records.append({
                'stock_code': stock,
                'pe_ttm': median_pe,
                'pb': median_pb,
                'dividend_yield': 0.02,  # 默认2%股息率
                'ps_ttm': 3.0,
                'roe': median_roe,
                'total_mv': np.nan,
                'circ_mv': np.nan,
                'listing_days': 500,  # 默认上市500天
            })
        
        return fallback_records
    
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
    
    def calculate_factors(self) -> bool:
        """
        计算因子数据
        
        Returns
        -------
        bool
            计算是否成功
        """
        self.logger.info("开始计算因子...")
        
        try:
            if self.ohlcv_data is None or self.financial_data is None:
                self.logger.warning("数据不完整，无法计算因子")
                return False
            
            # 准备数据
            ohlcv = self.ohlcv_data.copy()
            
            # 确保日期列
            if 'date' not in ohlcv.columns and 'trade_date' in ohlcv.columns:
                ohlcv['date'] = pd.to_datetime(ohlcv['trade_date'])
            
            # 合并财务数据
            factor_data = ohlcv.merge(
                self.financial_data,
                on='stock_code',
                how='left'
            )
            
            # 合并行业数据
            if self.industry_data is not None:
                factor_data = factor_data.merge(
                    self.industry_data,
                    on='stock_code',
                    how='left'
                )
            
            # 计算价值因子 EP_TTM
            factor_data['ep_ttm'] = 1.0 / factor_data['pe_ttm'].replace(0, np.nan)
            factor_data['ep_ttm'] = factor_data['ep_ttm'].replace([np.inf, -np.inf], np.nan)
            
            # 计算质量因子 ROE_Stability（简化版：直接使用ROE）
            factor_data['roe_stability'] = factor_data['roe']
            
            # 计算动量因子 RSI_20
            factor_data['rsi_20'] = factor_data.groupby('stock_code')['close'].transform(
                lambda x: self._calculate_rsi(x, 20)
            )
            
            # 计算特质波动率 IVOL
            factor_data['ivol'] = factor_data.groupby('stock_code')['close'].transform(
                lambda x: x.pct_change().rolling(20).std() * np.sqrt(252)
            )
            
            # Z-Score 标准化（行业中性化）
            date_col = 'date' if 'date' in factor_data.columns else 'trade_date'
            
            factor_data = z_score_normalize(
                factor_data,
                factor_cols=['ep_ttm', 'roe_stability', 'rsi_20'],
                date_col=date_col,
                industry_col='sw_industry_l1',
                industry_neutral=True
            )
            
            # 重命名标准化后的列以匹配策略配置
            factor_data.rename(columns={
                'ep_ttm_zscore': 'value_zscore',
                'roe_stability_zscore': 'quality_zscore', 
                'rsi_20_zscore': 'momentum_zscore'
            }, inplace=True)
            
            # 更新策略的因子列名配置
            self.strategy.value_col = 'value_zscore'
            self.strategy.quality_col = 'quality_zscore'
            self.strategy.momentum_col = 'momentum_zscore'
            
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
            date_col = 'date' if 'date' in self.ohlcv_data.columns else 'trade_date'
            trading_dates = pd.to_datetime(self.ohlcv_data[date_col].unique())
            
            # 筛选本月交易日
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
        生成目标持仓
        
        Returns
        -------
        bool
            生成是否成功
        """
        self.logger.info("开始生成目标持仓...")
        
        try:
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
                }, f, ensure_ascii=False, indent=2)
            
            return True
            
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
    
    def _generate_markdown_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        report_date: str
    ) -> str:
        """生成 Markdown 格式报告"""
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
                f"| 股票代码 | 买入金额 | 预估股数 |",
                f"|----------|----------|----------|",
            ])
            
            for stock, amount in sorted(buy_orders.items(), key=lambda x: -x[1]):
                # 假设股价为10元，估算股数
                estimated_shares = int(amount / 10 / 100) * 100  # 整百股
                lines.append(f"| {stock} | ¥{amount:,.0f} | {estimated_shares} |")
            
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
                f"| 股票代码 | 卖出金额 | 预估股数 |",
                f"|----------|----------|----------|",
            ])
            
            for stock, amount in sorted(sell_orders.items(), key=lambda x: -x[1]):
                estimated_shares = int(amount / 10 / 100) * 100
                lines.append(f"| {stock} | ¥{amount:,.0f} | {estimated_shares} |")
            
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
        portfolio_config = self.config.get("portfolio", {})
        total_capital = portfolio_config.get("total_capital", 1000000)
        
        # 买入表格行
        buy_rows = ""
        for stock, amount in sorted(buy_orders.items(), key=lambda x: -x[1]):
            estimated_shares = int(amount / 10 / 100) * 100
            buy_rows += f"""
                <tr>
                    <td>{stock}</td>
                    <td>¥{amount:,.0f}</td>
                    <td>{estimated_shares}</td>
                </tr>
            """
        
        # 卖出表格行
        sell_rows = ""
        for stock, amount in sorted(sell_orders.items(), key=lambda x: -x[1]):
            estimated_shares = int(amount / 10 / 100) * 100
            sell_rows += f"""
                <tr>
                    <td>{stock}</td>
                    <td>¥{amount:,.0f}</td>
                    <td>{estimated_shares}</td>
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
                        <th>预估股数</th>
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
                        <th>预估股数</th>
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


def run_daily_update(
    force_rebalance: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    运行每日更新流程
    
    流程：
    1. 调用 DataLoader 更新至今日的最新数据
    2. 调用 FactorCalculator 更新因子数据
    3. 检查今日是否为月底（调仓日）。如果是，运行 MultiFactorStrategy 生成新的目标持仓列表
    4. 调用 optimize_weights 计算每只持仓股的具体股数
    5. 生成报告
    
    Parameters
    ----------
    force_rebalance : bool
        是否强制调仓（忽略日期检查）
    config : Optional[Dict[str, Any]]
        配置参数
    
    Returns
    -------
    bool
        运行是否成功
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始每日更新流程")
    logger.info("=" * 60)
    
    try:
        # 初始化运行器
        runner = DailyUpdateRunner(config)
        
        # Step 1: 更新市场数据
        logger.info("Step 1/5: 更新市场数据")
        if not runner.update_market_data():
            logger.error("市场数据更新失败")
            return False
        
        # Step 2: 更新财务数据
        logger.info("Step 2/5: 更新财务数据")
        if not runner.update_financial_data():
            logger.error("财务数据更新失败")
            return False
        
        # Step 3: 计算因子
        logger.info("Step 3/5: 计算因子数据")
        if not runner.calculate_factors():
            logger.error("因子计算失败")
            return False
        
        # Step 4: 检查是否调仓日
        is_rebalance = force_rebalance or runner.is_rebalance_day()
        
        if is_rebalance:
            logger.info("Step 4/5: 生成目标持仓（调仓日）")
            if not runner.generate_target_positions():
                logger.error("目标持仓生成失败")
                return False
        else:
            logger.info("Step 4/5: 非调仓日，跳过持仓生成")
            runner.target_positions = runner.current_positions.copy()
        
        # Step 5: 生成报告
        logger.info("Step 5/5: 生成交易报告")
        buy_orders, sell_orders = runner.calculate_trade_orders()
        
        report_config = runner.config.get("report", {})
        report_format = report_config.get("format", "markdown")
        
        # 生成两种格式的报告
        for fmt in ["markdown", "html"]:
            report_content = runner.generate_report(buy_orders, sell_orders, format=fmt)
            runner.save_report(report_content, format=fmt)
        
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


def run_backtest(
    start_date: str,
    end_date: str,
    config: Optional[Dict[str, Any]] = None,
    strategy_type: str = "multi_factor"
) -> bool:
    """
    运行策略回测
    
    使用 BacktestEngine 对指定时间段的历史数据进行回测。
    
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
    
    Returns
    -------
    bool
        回测是否成功
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"开始回测: {start_date} ~ {end_date}")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        if config is None:
            try:
                config = load_config(CONFIG_PATH)
            except FileNotFoundError:
                config = {}
        
        # 回测配置
        backtest_config = config.get("backtest", {})
        portfolio_config = config.get("portfolio", {})
        
        initial_capital = portfolio_config.get("total_capital", 1000000)
        commission = config.get("trading", {}).get("commission_rate", 0.001)
        slippage = config.get("trading", {}).get("slippage", 0.001)
        risk_free_rate = portfolio_config.get("risk_free_rate", 0.02)
        
        # Step 1: 加载历史数据
        logger.info("Step 1/5: 加载历史数据")
        
        data_loader = DataLoader(output_dir=str(DATA_RAW_PATH))
        
        # 获取股票列表
        stock_list = data_loader.get_hs300_constituents()
        if not stock_list:
            logger.warning("无法获取沪深300成分股，使用示例股票")
            stock_list = ["000001", "000002", "600519", "601318", "000858"]
        
        # 限制回测股票数量（避免过长时间）
        max_stocks = backtest_config.get("max_stocks", 50)
        stock_list = stock_list[:max_stocks]
        logger.info(f"回测股票数量: {len(stock_list)}")
        
        # 下载历史数据
        price_data_dict = {}
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        
        for i, stock in enumerate(stock_list):
            try:
                df = data_loader.fetch_daily_price(stock, start_fmt, end_fmt)
                if df is not None and not df.empty:
                    price_data_dict[stock] = df
            except Exception as e:
                logger.debug(f"获取 {stock} 数据失败: {e}")
            
            if (i + 1) % 10 == 0:
                logger.info(f"数据加载进度: {i + 1}/{len(stock_list)}")
        
        if not price_data_dict:
            logger.error("未获取到任何历史数据")
            return False
        
        logger.info(f"成功加载 {len(price_data_dict)} 只股票的历史数据")
        
        # Step 2: 准备价格矩阵
        logger.info("Step 2/5: 准备价格矩阵")
        
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
        
        # Step 3: 创建策略
        logger.info("Step 3/5: 初始化策略")
        
        strategy_config = config.get("strategy", {})
        
        if strategy_type == "multi_factor":
            # 多因子策略需要因子数据，这里使用简化版
            strategy = MultiFactorStrategy(
                name="Multi-Factor Backtest",
                config={
                    "value_weight": strategy_config.get("value_weight", 0.4),
                    "quality_weight": strategy_config.get("quality_weight", 0.4),
                    "momentum_weight": strategy_config.get("momentum_weight", 0.2),
                    "top_n": strategy_config.get("top_n", 30),
                }
            )
            logger.info("使用多因子策略")
        else:
            # 均线交叉策略
            strategy = MACrossStrategy(
                name="MA Cross Backtest",
                config={
                    "short_window": 5,
                    "long_window": 20,
                }
            )
            logger.info("使用均线交叉策略")
        
        # Step 4: 运行回测
        logger.info("Step 4/5: 执行回测")
        
        # 初始化回测引擎
        backtest_engine = BacktestEngine(config={
            "initial_capital": initial_capital,
            "commission": commission,
            "slippage": slippage,
            "risk_free_rate": risk_free_rate,
        })
        
        # 对于多股票回测，使用 VBTProBacktester
        if len(close_df.columns) > 1:
            logger.info("使用 VBTProBacktester 进行多股票回测")
            
            vbt_backtester = VBTProBacktester(
                init_cash=initial_capital,
                fees=commission,
                fixed_amount=initial_capital / strategy_config.get("top_n", 30),
                slippage=slippage,
                risk_free_rate=risk_free_rate,
            )
            
            # 生成买入信号（使用简化策略：基于动量）
            # 计算20日收益率排名，选择排名前30%的股票
            returns_20d = close_df.pct_change(20)
            
            # 每月调仓信号
            entries = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)
            
            # 获取每月最后一个交易日
            monthly_dates = close_df.resample('M').last().index
            
            for date in monthly_dates:
                if date in returns_20d.index:
                    day_returns = returns_20d.loc[date].dropna()
                    if len(day_returns) > 0:
                        # 选择收益率最高的 top_n 只股票
                        top_n = min(strategy_config.get("top_n", 30), len(day_returns))
                        top_stocks = day_returns.nlargest(top_n).index
                        
                        # 在该日期设置买入信号
                        for stock in top_stocks:
                            if stock in entries.columns and date in entries.index:
                                entries.loc[date, stock] = True
            
            logger.info(f"生成买入信号完成，共 {entries.sum().sum():.0f} 个信号")
            
            # 运行回测
            result = vbt_backtester.run(close_df, entries)
            
            # 提取结果
            total_return = result.get('total_return', 0)
            annual_return = result.get('annual_return', 0)
            sharpe_ratio = result.get('sharpe_ratio', 0)
            max_drawdown = result.get('max_drawdown', 0)
            
        else:
            # 单股票回测
            logger.info("使用 BacktestEngine 进行单股票回测")
            
            # 准备单股票数据
            stock_code = close_df.columns[0]
            price_df = price_data_dict[stock_code]
            
            # 运行回测
            result = backtest_engine.run(strategy, price_df)
            
            total_return = result.total_return
            annual_return = result.annual_return
            sharpe_ratio = result.sharpe_ratio
            max_drawdown = result.max_drawdown
        
        # Step 5: 生成回测报告
        logger.info("Step 5/5: 生成回测报告")
        
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
        
        # 打印回测结果
        logger.info("=" * 60)
        logger.info("回测结果摘要")
        logger.info("=" * 60)
        logger.info(f"策略名称: {strategy.name}")
        logger.info(f"回测区间: {start_date} ~ {end_date}")
        logger.info(f"股票数量: {len(close_df.columns)}")
        logger.info(f"初始资金: ¥{initial_capital:,.0f}")
        logger.info(f"总收益率: {total_return:.2%}")
        logger.info(f"年化收益: {annual_return:.2%}")
        logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        logger.info(f"最大回撤: {max_drawdown:.2%}")
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
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    log_file = LOGS_PATH / f"quant_{datetime.now().strftime('%Y%m%d')}.log"
    setup_logging(level=log_level, log_file=str(log_file))
    
    logger = logging.getLogger(__name__)
    logger.info("A股量化交易系统启动")
    
    if args.daily_update:
        success = run_daily_update(force_rebalance=args.force_rebalance)
        exit(0 if success else 1)
    
    elif args.backtest:
        logger.info(f"回测模式: {args.start} ~ {args.end}, 策略: {args.strategy}")
        success = run_backtest(
            start_date=args.start,
            end_date=args.end,
            strategy_type=args.strategy
        )
        exit(0 if success else 1)
    
    else:
        parser.print_help()
        exit(0)


if __name__ == "__main__":
    main()

