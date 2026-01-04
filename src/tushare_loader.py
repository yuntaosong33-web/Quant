"""
Tushare 数据加载器模块

该模块提供基于 Tushare Pro 的数据获取功能，替代不稳定的 AkShare。
支持获取日线数据、财务指标、指数成分股等。

Features
--------
- 日线行情数据 (daily, daily_basic)
- 财务指标数据 (fina_indicator)
- 指数成分股权重 (index_weight)
- 本地缓存机制
- 自动重试和限流

Notes
-----
使用前需要配置 Tushare API Token：
1. 在 config/strategy_config.yaml 中设置 tushare.api_token
2. 或通过环境变量 TUSHARE_TOKEN 设置
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
import os

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TushareDataLoader:
    """
    Tushare Pro 数据加载器
    
    提供稳定可靠的 A 股数据获取服务，包括：
    - 日线行情数据 (OHLCV + 基础指标)
    - 财务指标数据 (PE, PB, ROE 等)
    - 指数成分股权重
    - 股票基础信息
    
    Parameters
    ----------
    api_token : Optional[str]
        Tushare API Token，如果不提供则从环境变量 TUSHARE_TOKEN 读取
    cache_dir : str
        本地缓存目录，默认 "data/tushare_cache"
    
    Attributes
    ----------
    pro : tushare.pro_api
        Tushare Pro API 实例
    cache_dir : Path
        缓存目录路径
    
    Examples
    --------
    >>> loader = TushareDataLoader(api_token="your_token")
    >>> df = loader.fetch_daily_data("000001.SZ", "20240101", "20241231")
    >>> financial = loader.fetch_financial_indicator("000001.SZ")
    """
    
    # API 请求限流参数
    # 普通用户限制: 200 次/分钟 = 3.33 次/秒，安全起见设为 0.35 秒间隔
    REQUEST_INTERVAL = 0.35  # 每次请求间隔（秒）
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # 重试延迟
    RATE_LIMIT_DELAY = 60.0  # 触发频率限制后等待时间（秒）
    
    # 股票池代码映射
    INDEX_CODE_MAPPING = {
        "hs300": "000300.SH",
        "zz500": "000905.SH",
        "zz1000": "000852.SH",
        "sz50": "000016.SH",
        "cyb": "399006.SZ",  # 创业板指
    }
    
    def __init__(
        self,
        api_token: Optional[str] = None,
        cache_dir: str = "data/tushare_cache"
    ) -> None:
        """
        初始化 Tushare 数据加载器
        
        Parameters
        ----------
        api_token : Optional[str]
            Tushare API Token
        cache_dir : str
            缓存目录
        """
        # 获取 API Token
        self.api_token = api_token or os.environ.get("TUSHARE_TOKEN", "")
        
        if not self.api_token:
            raise ValueError(
                "Tushare API Token 未配置！\n"
                "请通过以下方式之一配置：\n"
                "1. 构造函数参数 api_token\n"
                "2. 环境变量 TUSHARE_TOKEN\n"
                "3. config/strategy_config.yaml 中的 tushare.api_token\n"
                "获取 Token: https://tushare.pro/register"
            )
        
        # 初始化 Tushare Pro API
        try:
            import tushare as ts
            self.pro = ts.pro_api(self.api_token)
            logger.info("Tushare Pro API 初始化成功")
        except ImportError:
            raise ImportError("请安装 tushare: pip install tushare")
        except Exception as e:
            raise RuntimeError(f"Tushare API 初始化失败: {e}")
        
        # 设置缓存目录
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 请求计数器（用于限流）
        self._last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """API 请求限流"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()
    
    def _fetch_with_retry(
        self,
        func,
        *args,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        带重试的 API 请求
        
        Parameters
        ----------
        func : callable
            Tushare API 函数
        *args, **kwargs
            函数参数
        
        Returns
        -------
        Optional[pd.DataFrame]
            返回数据，失败返回 None
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                result = func(*args, **kwargs)
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                error_msg = str(e)
                # 检查是否触发频率限制
                if "每分钟最多访问" in error_msg or "抱歉" in error_msg:
                    logger.warning(f"触发 API 频率限制，等待 {self.RATE_LIMIT_DELAY} 秒后重试...")
                    time.sleep(self.RATE_LIMIT_DELAY)
                else:
                    logger.warning(f"API 请求失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY * (attempt + 1))
        return None
    
    # ==================== 指数成分股 ====================
    
    def fetch_index_constituents(
        self,
        index_code: str = "hs300",
        trade_date: Optional[str] = None
    ) -> List[str]:
        """
        获取指数成分股列表
        
        Parameters
        ----------
        index_code : str
            指数代码，支持: hs300, zz500, zz1000, sz50, cyb
            或直接使用 Tushare 代码如 "000300.SH"
        trade_date : Optional[str]
            交易日期，格式 YYYYMMDD，默认最近交易日
        
        Returns
        -------
        List[str]
            成分股代码列表（6位代码，如 "000001"）
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> stocks = loader.fetch_index_constituents("hs300")
        >>> print(len(stocks))  # 约 300 只
        """
        # 转换指数代码
        ts_index_code = self.INDEX_CODE_MAPPING.get(index_code.lower(), index_code)
        
        # 默认使用最近交易日
        if trade_date is None:
            trade_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        
        logger.info(f"获取指数成分股: {ts_index_code}, 日期: {trade_date}")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"index_{index_code}_{trade_date[:6]}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.info(f"从缓存加载指数成分股: {len(df)} 只")
                    # 返回 6 位代码
                    return df["con_code"].str[:6].tolist()
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.index_weight,
            index_code=ts_index_code,
            start_date=trade_date,
            end_date=trade_date
        )
        
        if df is None or df.empty:
            # 尝试最近一个月的数据
            end_date = trade_date
            start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
            df = self._fetch_with_retry(
                self.pro.index_weight,
                index_code=ts_index_code,
                start_date=start_date,
                end_date=end_date
            )
        
        if df is None or df.empty:
            logger.warning(f"无法获取指数成分股: {ts_index_code}")
            return []
        
        # 取最新日期的成分股
        df = df.sort_values("trade_date", ascending=False)
        latest_date = df["trade_date"].iloc[0]
        df = df[df["trade_date"] == latest_date]
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
            logger.info(f"指数成分股已缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
        
        # 返回 6 位代码
        stock_list = df["con_code"].str[:6].tolist()
        logger.info(f"获取到 {len(stock_list)} 只成分股")
        return stock_list
    
    # ==================== 日线数据 ====================
    
    def fetch_daily_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        adj: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票日线数据
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位，如 "000001"）
        start_date : str
            开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date : str
            结束日期，格式 YYYYMMDD 或 YYYY-MM-DD
        adj : str
            复权方式: qfq(前复权), hfq(后复权), None(不复权)
        
        Returns
        -------
        Optional[pd.DataFrame]
            日线数据，包含 date, open, high, low, close, volume, amount 等
        """
        # 标准化股票代码
        ts_code = self._to_ts_code(stock_code)
        
        # 标准化日期格式
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"daily_{stock_code}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载日线数据: {stock_code}")
                    return self._standardize_daily_columns(df)
            except Exception:
                pass
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取日线数据失败: {stock_code}")
            return None
        
        # 前复权处理
        if adj == "qfq":
            adj_factor = self._fetch_with_retry(
                self.pro.adj_factor,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            if adj_factor is not None and not adj_factor.empty:
                df = df.merge(adj_factor[["trade_date", "adj_factor"]], on="trade_date", how="left")
                df["adj_factor"] = df["adj_factor"].fillna(1.0)
                latest_factor = df["adj_factor"].iloc[0]
                factor = df["adj_factor"] / latest_factor
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df[col] = df[col] * factor
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        return self._standardize_daily_columns(df)
    
    def fetch_daily_data_batch(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        adj: str = "qfq",
        show_progress: bool = True,
        batch_size: int = 150,
        batch_sleep: float = 20.0
    ) -> pd.DataFrame:
        """
        批量获取日线数据（带限流保护）
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        adj : str
            复权方式
        show_progress : bool
            是否显示进度
        batch_size : int
            每批次处理的股票数量（默认 150）
        batch_sleep : float
            每批次之间的休息时间（秒）
        
        Returns
        -------
        pd.DataFrame
            合并后的日线数据
        """
        all_data = []
        total = len(stock_list)
        
        for i, stock in enumerate(stock_list):
            df = self.fetch_daily_data(stock, start_date, end_date, adj)
            if df is not None and not df.empty:
                df["stock_code"] = stock
                all_data.append(df)
            
            # 进度日志
            if show_progress and (i + 1) % 50 == 0:
                logger.info(f"日线数据进度: {i + 1}/{total}")
            
            # 批次休息（避免触发频率限制）
            if (i + 1) % batch_size == 0 and (i + 1) < total:
                logger.info(f"已处理 {i + 1} 只，休息 {batch_sleep} 秒避免触发频率限制...")
                time.sleep(batch_sleep)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"批量获取日线数据完成: {len(stock_list)} 只股票, {len(result)} 条记录")
        return result
    
    # ==================== 财务指标 ====================
    
    def fetch_financial_indicator(
        self,
        stock_code: str,
        period: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票财务指标
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位）
        period : Optional[str]
            报告期，格式 YYYYMMDD，如 "20231231"
            如果不提供，返回最近 8 个季度的数据
        
        Returns
        -------
        Optional[pd.DataFrame]
            财务指标数据，包含：
            - roe: 净资产收益率
            - roe_dt: 扣非净资产收益率
            - roa: 总资产收益率
            - gross_margin: 毛利率
            - profit_to_gr: 净利率
            - eps: 每股收益
            - bps: 每股净资产
        """
        ts_code = self._to_ts_code(stock_code)
        
        # 尝试缓存
        cache_file = self.cache_dir / f"fina_{stock_code}.parquet"
        cache_valid = False
        
        if cache_file.exists():
            try:
                cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                # 缓存有效期 7 天
                if (datetime.now() - cache_mtime).days < 7:
                    df = pd.read_parquet(cache_file)
                    if not df.empty:
                        logger.debug(f"从缓存加载财务指标: {stock_code}")
                        cache_valid = True
                        return self._standardize_financial_columns(df)
            except Exception:
                pass
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.fina_indicator,
            ts_code=ts_code,
            period=period
        )
        
        if df is None or df.empty:
            logger.debug(f"获取财务指标失败: {stock_code}")
            return None
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        return self._standardize_financial_columns(df)
    
    def fetch_daily_basic(
        self,
        trade_date: Optional[str] = None,
        stock_list: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取每日基础指标（PE, PB, 市值等）
        
        这是获取估值数据最高效的方式，一次请求获取全市场数据。
        
        Parameters
        ----------
        trade_date : Optional[str]
            交易日期，格式 YYYYMMDD，默认最近交易日
        stock_list : Optional[List[str]]
            股票列表，用于过滤结果
        
        Returns
        -------
        Optional[pd.DataFrame]
            基础指标数据，包含：
            - pe_ttm: 市盈率 TTM
            - pb: 市净率
            - ps_ttm: 市销率 TTM
            - dv_ttm: 股息率 TTM
            - total_mv: 总市值（万元）
            - circ_mv: 流通市值（万元）
            - turnover_rate: 换手率
        """
        if trade_date is None:
            trade_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"daily_basic_{trade_date}.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.info(f"从缓存加载每日基础指标: {trade_date}, {len(df)} 条")
                    if stock_list:
                        df = df[df["ts_code"].str[:6].isin(stock_list)]
                    return self._standardize_basic_columns(df)
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.daily_basic,
            trade_date=trade_date
        )
        
        if df is None or df.empty:
            # 尝试前几天
            for days_ago in range(1, 8):
                alt_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=days_ago)).strftime("%Y%m%d")
                df = self._fetch_with_retry(
                    self.pro.daily_basic,
                    trade_date=alt_date
                )
                if df is not None and not df.empty:
                    logger.info(f"使用 {alt_date} 的基础指标数据")
                    break
        
        if df is None or df.empty:
            logger.warning(f"无法获取每日基础指标: {trade_date}")
            return None
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
            logger.info(f"每日基础指标已缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
        
        if stock_list:
            df = df[df["ts_code"].str[:6].isin(stock_list)]
        
        return self._standardize_basic_columns(df)
    
    def fetch_financial_batch(
        self,
        stock_list: List[str],
        show_progress: bool = True,
        batch_size: int = 100,
        batch_sleep: float = 30.0
    ) -> pd.DataFrame:
        """
        批量获取财务指标（带限流保护）
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        show_progress : bool
            是否显示进度
        batch_size : int
            每批次处理的股票数量（默认 100）
        batch_sleep : float
            每批次之间的休息时间（秒）
        
        Returns
        -------
        pd.DataFrame
            合并后的财务指标数据
        """
        all_data = []
        total = len(stock_list)
        
        for i, stock in enumerate(stock_list):
            df = self.fetch_financial_indicator(stock)
            if df is not None and not df.empty:
                # 只取最新一期
                df = df.sort_values("end_date", ascending=False).head(1)
                df["stock_code"] = stock
                all_data.append(df)
            
            # 进度日志
            if show_progress and (i + 1) % 50 == 0:
                logger.info(f"财务指标进度: {i + 1}/{total}")
            
            # 批次休息（避免触发频率限制）
            if (i + 1) % batch_size == 0 and (i + 1) < total:
                logger.info(f"已处理 {i + 1} 只，休息 {batch_sleep} 秒避免触发频率限制...")
                time.sleep(batch_sleep)
        
        if not all_data:
            return pd.DataFrame()
        
        # 过滤掉全空的 DataFrame，避免 FutureWarning
        valid_data = [df for df in all_data if not df.isna().all().all()]
        if not valid_data:
            return pd.DataFrame()
        
        result = pd.concat(valid_data, ignore_index=True)
        logger.info(f"批量获取财务指标完成: {len(stock_list)} 只股票, {len(result)} 条记录")
        return result
    
    # ==================== 指数日线 ====================
    
    def fetch_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        获取指数日线数据
        
        Parameters
        ----------
        index_code : str
            指数代码，如 "000300" 或 "hs300"
        start_date : str
            开始日期
        end_date : str
            结束日期
        
        Returns
        -------
        Optional[pd.DataFrame]
            指数日线数据
        """
        # 转换指数代码
        if index_code.lower() in self.INDEX_CODE_MAPPING:
            ts_code = self.INDEX_CODE_MAPPING[index_code.lower()]
        elif "." in index_code:
            ts_code = index_code
        else:
            # 假设是上证指数
            ts_code = f"{index_code}.SH"
        
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        df = self._fetch_with_retry(
            self.pro.index_daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            return None
        
        return self._standardize_daily_columns(df)
    
    # ==================== 辅助方法 ====================
    
    def _to_ts_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Tushare 格式
        
        Parameters
        ----------
        stock_code : str
            6位股票代码
        
        Returns
        -------
        str
            Tushare 格式代码，如 "000001.SZ"
        """
        if "." in stock_code:
            return stock_code
        
        code = stock_code.strip()
        
        # 根据首位判断交易所
        if code.startswith(("6", "5")):
            return f"{code}.SH"
        elif code.startswith(("0", "3", "2")):
            return f"{code}.SZ"
        elif code.startswith("8") or code.startswith("4"):
            return f"{code}.BJ"  # 北交所
        else:
            return f"{code}.SZ"
    
    def _standardize_daily_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化日线数据列名"""
        column_mapping = {
            "trade_date": "date",
            "ts_code": "ts_code",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "vol": "volume",
            "amount": "amount",
            "pct_chg": "pct_change",
            "change": "change",
        }
        
        df = df.rename(columns=column_mapping)
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
        
        # 成交量单位转换（Tushare 单位是手，转为股）
        if "volume" in df.columns:
            df["volume"] = df["volume"] * 100
        
        # 成交额单位转换（Tushare 单位是千元，转为元）
        if "amount" in df.columns:
            df["amount"] = df["amount"] * 1000
        
        return df
    
    def _standardize_basic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化每日基础指标列名"""
        column_mapping = {
            "trade_date": "date",
            "ts_code": "ts_code",
            "pe_ttm": "pe_ttm",
            "pe": "pe",
            "pb": "pb",
            "ps_ttm": "ps_ttm",
            "dv_ttm": "dividend_yield",
            "dv_ratio": "dividend_yield",
            "total_mv": "total_mv",
            "circ_mv": "circ_mv",
            "turnover_rate": "turn",
            "turnover_rate_f": "turn_free",
        }
        
        df = df.rename(columns=column_mapping)
        
        # 提取 6 位股票代码
        if "ts_code" in df.columns:
            df["stock_code"] = df["ts_code"].str[:6]
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # 市值单位转换（万元 -> 元）
        for col in ["total_mv", "circ_mv"]:
            if col in df.columns:
                df[col] = df[col] * 10000
        
        return df
    
    def _standardize_financial_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化财务指标列名"""
        column_mapping = {
            "ts_code": "ts_code",
            "ann_date": "ann_date",
            "end_date": "end_date",
            "roe": "roe",
            "roe_dt": "roe_dt",
            "roe_yearly": "roe_ttm",
            "roa": "roa",
            "grossprofit_margin": "gross_margin",
            "profit_to_gr": "net_margin",
            "eps": "eps",
            "bps": "bps",
            "netprofit_margin": "net_margin",
            "current_ratio": "current_ratio",
            "quick_ratio": "quick_ratio",
        }
        
        df = df.rename(columns=column_mapping)
        
        # 提取 6 位股票代码
        if "ts_code" in df.columns:
            df["stock_code"] = df["ts_code"].str[:6]
        
        return df


# ==================== 便捷函数 ====================

def create_tushare_loader(config: Optional[Dict] = None) -> TushareDataLoader:
    """
    创建 Tushare 数据加载器
    
    Parameters
    ----------
    config : Optional[Dict]
        配置字典，包含 tushare.api_token
    
    Returns
    -------
    TushareDataLoader
        数据加载器实例
    """
    api_token = None
    
    if config:
        api_token = config.get("tushare", {}).get("api_token")
    
    if not api_token:
        api_token = os.environ.get("TUSHARE_TOKEN")
    
    return TushareDataLoader(api_token=api_token)

