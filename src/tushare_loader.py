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

# 全局变量：追踪新闻 API 最后调用时间（跨实例共享）
_GLOBAL_NEWS_API_LAST_CALL = 0.0
_GLOBAL_NEWS_RATE_LIMIT_COUNT = 0


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
    # 普通用户限制: 200 次/分钟 = 3.33 次/秒
    # 付费用户限制更高，可适当降低间隔
    REQUEST_INTERVAL = 0.12  # 每次请求间隔（秒）- 激进模式
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # 重试延迟（秒）
    RATE_LIMIT_DELAY = 30.0  # 触发频率限制后等待时间（秒）
    HTTP_TIMEOUT = 60  # HTTP 请求超时（秒）
    
    # 新闻接口特殊限制：每分钟最多 1 次
    NEWS_API_INTERVAL = 61.0  # 新闻接口调用间隔（秒）
    
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
        # 获取 API Token (优先级: 参数 > 环境变量 > 配置文件)
        self.api_token = api_token or os.environ.get("TUSHARE_TOKEN", "")
        
        # 尝试从配置文件读取
        self._skip_news = False  # 默认不跳过新闻
        try:
            import yaml
            config_path = Path("config/strategy_config.yaml")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                tushare_config = config.get("tushare", {})
                
                # 读取 Token（如果还没有）
                if not self.api_token:
                    self.api_token = tushare_config.get("api_token", "")
                    if self.api_token:
                        logger.info("从配置文件加载 Tushare Token")
                
                # 读取 skip_news 配置
                self._skip_news = tushare_config.get("skip_news", False)
                if self._skip_news:
                    logger.info("📰 新闻获取已禁用 (tushare.skip_news=true)")
        except Exception as e:
            logger.debug(f"从配置文件读取配置失败: {e}")
        
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
            
            # 设置 Token 并初始化 API
            ts.set_token(self.api_token)
            self.pro = ts.pro_api()
            
            # 配置更长的 HTTP 超时（通过修改底层 DataApi）
            try:
                if hasattr(self.pro, '_DataApi__http'):
                    # 新版 Tushare 使用 __http 属性
                    self.pro._DataApi__http.timeout = self.HTTP_TIMEOUT
                elif hasattr(self.pro, 'timeout'):
                    self.pro.timeout = self.HTTP_TIMEOUT
                logger.info(f"Tushare Pro API 初始化成功 (timeout={self.HTTP_TIMEOUT}s)")
            except Exception:
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
                # 空结果也算成功，不需要重试
                if result is not None:
                    return result
            except Exception as e:
                error_msg = str(e)
                error_msg_lower = error_msg.lower()
                # 检查是否触发频率限制（多种错误格式）
                rate_limit_keywords = ["每分钟最多访问", "抱歉", "频率", "rate limit", "too many", "限制"]
                if any(kw in error_msg or kw in error_msg_lower for kw in rate_limit_keywords):
                    logger.warning(f"触发 API 频率限制，等待 {self.RATE_LIMIT_DELAY} 秒后重试... 错误: {error_msg[:100]}")
                    time.sleep(self.RATE_LIMIT_DELAY)
                # 网络超时：使用指数退避
                elif "timeout" in error_msg_lower or "timed out" in error_msg_lower:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)  # 指数退避: 2, 4, 8 秒
                    logger.warning(
                        f"网络超时 (尝试 {attempt + 1}/{self.MAX_RETRIES}), "
                        f"等待 {wait_time:.1f}s 后重试..."
                    )
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(wait_time)
                # 连接错误：可能是网络问题
                elif "connection" in error_msg_lower or "connect" in error_msg_lower:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"连接失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}, "
                        f"等待 {wait_time:.1f}s 后重试..."
                    )
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(wait_time)
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
    
    def fetch_all_stocks(
        self,
        exchange: Optional[str] = None,
        list_status: str = "L"
    ) -> List[str]:
        """
        获取全市场股票列表
        
        使用 Tushare stock_basic 接口获取所有上市股票。
        
        Parameters
        ----------
        exchange : Optional[str]
            交易所筛选：
            - None: 全部（默认）
            - "SSE": 上交所
            - "SZSE": 深交所
        list_status : str
            上市状态：
            - "L": 上市中（默认）
            - "D": 退市
            - "P": 暂停上市
        
        Returns
        -------
        List[str]
            股票代码列表（6位代码）
        
        Notes
        -----
        - 默认只获取上市中的股票
        - 会自动过滤 ST、退市风险警示股票
        - 结果会缓存到本地（当日有效）
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> all_stocks = loader.fetch_all_stocks()
        >>> print(f"全市场共 {len(all_stocks)} 只股票")
        """
        logger.info(f"🔍 获取全市场股票列表: exchange={exchange}, list_status={list_status}")
        
        # 尝试今日缓存
        today = datetime.now().strftime("%Y%m%d")
        cache_file = self.cache_dir / f"stock_basic_{today}.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                # 应用筛选条件
                if exchange:
                    df = df[df["exchange"] == exchange]
                stock_list = df["ts_code"].str[:6].tolist()
                logger.info(f"从缓存加载全市场股票列表: {len(stock_list)} 只")
                return stock_list
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        # 调用 API 获取股票基础信息
        df = self._fetch_with_retry(
            self.pro.stock_basic,
            exchange=exchange or "",
            list_status=list_status,
            fields="ts_code,symbol,name,area,industry,market,list_date,exchange"
        )
        
        if df is None or df.empty:
            # 网络失败时，尝试使用最近的缓存文件
            logger.warning("API 请求失败，尝试使用历史缓存...")
            cache_files = sorted(
                self.cache_dir.glob("stock_basic_*.parquet"),
                reverse=True
            )
            for old_cache in cache_files[:5]:  # 最多检查最近5个缓存
                try:
                    df = pd.read_parquet(old_cache)
                    if not df.empty:
                        if exchange:
                            df = df[df["exchange"] == exchange]
                        stock_list = df["ts_code"].str[:6].tolist()
                        logger.info(
                            f"使用历史缓存 {old_cache.name}: {len(stock_list)} 只股票"
                        )
                        return stock_list
                except Exception:
                    continue
            logger.warning("无可用缓存，无法获取全市场股票列表")
            return []
        
        # 过滤 ST 和退市风险股票
        if "name" in df.columns:
            st_mask = df["name"].str.contains(r"ST|\*ST|退|S\s|PT", na=False, regex=True)
            before_count = len(df)
            df = df[~st_mask]
            filtered_count = before_count - len(df)
            if filtered_count > 0:
                logger.info(f"过滤 ST/退市风险股票: {filtered_count} 只")
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
            logger.info(f"全市场股票列表已缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
        
        # 返回 6 位代码
        stock_list = df["ts_code"].str[:6].tolist()
        logger.info(f"获取全市场股票列表完成: {len(stock_list)} 只")
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
        batch_size: int = 200,
        batch_sleep: float = 5.0
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
        success_count = 0
        
        # 使用 tqdm 进度条
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(stock_list), 
                    total=total, 
                    desc="📊 获取日线数据",
                    unit="只",
                    ncols=80
                )
            except ImportError:
                iterator = enumerate(stock_list)
                logger.info(f"开始获取日线数据: {total} 只股票...")
        else:
            iterator = enumerate(stock_list)
        
        for i, stock in iterator:
            df = self.fetch_daily_data(stock, start_date, end_date, adj)
            if df is not None and not df.empty:
                df["stock_code"] = stock
                all_data.append(df)
                success_count += 1
            
            # 更新进度条后缀
            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({"成功": success_count, "当前": stock})
            
            # 批次休息（避免触发频率限制）
            if (i + 1) % batch_size == 0 and (i + 1) < total:
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description(f"📊 休息{batch_sleep}s")
                time.sleep(batch_sleep)
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description("📊 获取日线数据")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"批量获取日线数据完成: {success_count}/{total} 只股票成功, {len(result)} 条记录")
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
        batch_size: int = 150,
        batch_sleep: float = 8.0
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
        success_count = 0
        
        # 使用 tqdm 进度条
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(stock_list), 
                    total=total, 
                    desc="📈 获取财务指标",
                    unit="只",
                    ncols=80
                )
            except ImportError:
                iterator = enumerate(stock_list)
                logger.info(f"开始获取财务指标: {total} 只股票...")
        else:
            iterator = enumerate(stock_list)
        
        for i, stock in iterator:
            df = self.fetch_financial_indicator(stock)
            if df is not None and not df.empty:
                # 只取最新一期
                df = df.sort_values("end_date", ascending=False).head(1)
                df["stock_code"] = stock
                all_data.append(df)
                success_count += 1
            
            # 更新进度条后缀
            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({"成功": success_count, "当前": stock})
            
            # 批次休息（避免触发频率限制）
            if (i + 1) % batch_size == 0 and (i + 1) < total:
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description(f"📈 休息{batch_sleep}s")
                time.sleep(batch_sleep)
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description("📈 获取财务指标")
        
        if not all_data:
            return pd.DataFrame()
        
        # 过滤掉全空的 DataFrame，避免 FutureWarning
        valid_data = [df for df in all_data if not df.isna().all().all()]
        if not valid_data:
            return pd.DataFrame()
        
        result = pd.concat(valid_data, ignore_index=True)
        logger.info(f"批量获取财务指标完成: {success_count}/{total} 只股票成功, {len(result)} 条记录")
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
    
    # ==================== 新闻资讯 ====================
    
    def fetch_news(
        self,
        stock_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        src: str = "sina"
    ) -> Optional[pd.DataFrame]:
        """
        获取新闻资讯数据
        
        使用 Tushare Pro news 接口获取财经新闻。
        
        Parameters
        ----------
        stock_code : Optional[str]
            股票代码（6位），如果提供则过滤相关新闻
        start_date : Optional[str]
            开始日期，格式 YYYYMMDD
        end_date : Optional[str]
            结束日期，格式 YYYYMMDD
        src : str
            新闻来源，可选：sina(新浪), wallstreetcn(华尔街见闻), 
            10jqka(同花顺), eastmoney(东方财富), yuncaijing(云财经)
            默认 sina
        
        Returns
        -------
        Optional[pd.DataFrame]
            新闻数据，包含 datetime, title, content, channels 等字段
            失败返回 None
        
        Notes
        -----
        - Tushare Pro 新闻接口需要较高积分权限
        - 如果接口不可用，会返回空 DataFrame
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> news = loader.fetch_news(start_date="20240101", end_date="20240115")
        >>> print(news[['datetime', 'title']].head())
        """
        # 标准化日期格式
        if start_date:
            start_date = start_date.replace("-", "")
        if end_date:
            end_date = end_date.replace("-", "")
        
        global _GLOBAL_NEWS_API_LAST_CALL, _GLOBAL_NEWS_RATE_LIMIT_COUNT
        
        # 检查是否在配置中跳过新闻获取
        if getattr(self, '_skip_news', False):
            logger.debug("新闻获取已在配置中禁用 (tushare.skip_news=true)")
            return pd.DataFrame()
        
        # 检查是否应该跳过新闻获取（频率限制保护）
        if _GLOBAL_NEWS_RATE_LIMIT_COUNT >= 3:
            logger.warning("新闻接口频繁触发限制，本次跳过（需要更高积分权限）")
            return pd.DataFrame()
        
        # 尝试缓存（新闻按日期和来源缓存）
        cache_key = f"news_{src}_{start_date}_{end_date}"
        if stock_code:
            cache_key += f"_{stock_code.replace('.', '')[:6]}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if cache_file.exists():
            try:
                # 检查缓存是否在24小时内
                cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.now() - cache_mtime).total_seconds() < 86400:  # 24小时
                    df = pd.read_parquet(cache_file)
                    if not df.empty:
                        logger.info(f"从缓存加载新闻: {len(df)} 条")
                        return df
            except Exception:
                pass
        
        # 新闻接口特殊限流：每分钟最多 1 次（使用全局变量跨实例共享）
        elapsed = time.time() - _GLOBAL_NEWS_API_LAST_CALL
        if elapsed < self.NEWS_API_INTERVAL:
            wait_time = self.NEWS_API_INTERVAL - elapsed
            logger.info(f"⏳ 新闻接口限流（每分钟1次），等待 {wait_time:.0f} 秒...")
            time.sleep(wait_time)
        
        logger.info(f"获取新闻资讯: src={src}, {start_date} ~ {end_date}")
        
        try:
            # 更新全局最后调用时间
            _GLOBAL_NEWS_API_LAST_CALL = time.time()
            
            df = self._fetch_with_retry(
                self.pro.news,
                src=src,
                start_date=start_date,
                end_date=end_date
            )
            
            # 成功则重置全局计数器
            _GLOBAL_NEWS_RATE_LIMIT_COUNT = 0
            
            if df is None or df.empty:
                logger.debug("无新闻数据")
                return pd.DataFrame()
            
            # 如果指定了股票代码，尝试过滤相关新闻
            if stock_code:
                # 在标题或内容中搜索股票代码或名称
                stock_code_clean = stock_code.replace(".", "")[:6]
                mask = (
                    df["title"].str.contains(stock_code_clean, na=False) |
                    df["content"].str.contains(stock_code_clean, na=False)
                )
                df = df[mask]
            
            # 保存缓存
            if not df.empty:
                try:
                    df.to_parquet(cache_file, index=False)
                    logger.debug(f"新闻已缓存: {cache_file.name}")
                except Exception:
                    pass
            
            logger.info(f"获取新闻成功: {len(df)} 条")
            return df
            
        except Exception as e:
            error_msg = str(e)
            # 记录频率限制（使用全局变量）
            if "每小时" in error_msg:
                # 每小时限制 - 本次会话内不再尝试
                _GLOBAL_NEWS_RATE_LIMIT_COUNT = 10  # 设置高值直接跳过
                logger.warning(f"⚠️ 新闻接口每小时限制已达上限，本次跳过新闻获取")
                logger.warning(f"   提示：可在配置中设置 llm.enable_sentiment_filter: false 暂时禁用情绪分析")
            elif "每分钟" in error_msg or "频率" in error_msg.lower() or "抱歉" in error_msg:
                _GLOBAL_NEWS_RATE_LIMIT_COUNT += 1
                logger.warning(f"新闻接口频率限制 ({_GLOBAL_NEWS_RATE_LIMIT_COUNT}/3): {e}")
            else:
                logger.warning(f"获取新闻失败: {e}")
            return pd.DataFrame()
    
    def fetch_all_news_once(
        self,
        days_back: int = 7,
        src: str = "sina"
    ) -> pd.DataFrame:
        """
        一次性获取所有新闻（优化：避免多次 API 调用）
        
        获取最近几天的所有新闻，缓存后供多只股票使用。
        新闻接口每分钟只能调用1次，因此一次获取全部数据更高效。
        
        Parameters
        ----------
        days_back : int
            回溯天数，默认 7 天
        src : str
            新闻源，默认 sina
        
        Returns
        -------
        pd.DataFrame
            所有新闻数据
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        
        # 使用实例变量缓存，避免重复调用
        cache_key = f"_cached_all_news_{src}_{start_date}_{end_date}"
        if hasattr(self, cache_key):
            cached = getattr(self, cache_key)
            if cached is not None:
                logger.debug(f"使用内存缓存的新闻数据: {len(cached)} 条")
                return cached
        
        # 获取所有新闻（不带股票代码过滤）
        df = self.fetch_news(
            stock_code=None,  # 不过滤，获取全部
            start_date=start_date,
            end_date=end_date,
            src=src
        )
        
        # 缓存到实例变量
        setattr(self, cache_key, df if df is not None else pd.DataFrame())
        
        if df is not None and not df.empty:
            logger.info(f"📰 一次性获取新闻完成: {len(df)} 条，可供所有股票使用")
        
        return df if df is not None else pd.DataFrame()
    
    def fetch_stock_news(
        self,
        stock_code: str,
        days_back: int = 7
    ) -> str:
        """
        获取单只股票相关新闻（用于情感分析）
        
        从缓存的全量新闻中筛选与指定股票相关的新闻。
        优化：只调用一次 API 获取全量新闻，然后本地筛选。
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位）
        days_back : int
            回溯天数，默认 7 天
        
        Returns
        -------
        str
            合并的新闻文本，用于情感分析
            无新闻时返回空字符串
        """
        # 先获取全量新闻（会自动缓存，只调用一次 API）
        all_news_df = self.fetch_all_news_once(days_back=days_back)
        
        if all_news_df.empty:
            logger.debug(f"无新闻数据可用")
            return ""
        
        # 从全量新闻中筛选与该股票相关的
        stock_code_clean = stock_code.replace(".", "")[:6]
        
        # 在标题或内容中搜索股票代码
        mask = pd.Series([False] * len(all_news_df))
        if "title" in all_news_df.columns:
            mask = mask | all_news_df["title"].str.contains(stock_code_clean, na=False)
        if "content" in all_news_df.columns:
            mask = mask | all_news_df["content"].str.contains(stock_code_clean, na=False)
        
        filtered_df = all_news_df[mask]
        
        if filtered_df.empty:
            logger.debug(f"股票 {stock_code} 无相关新闻")
            return ""
        
        # 提取标题和内容
        all_news = []
        for _, row in filtered_df.head(5).iterrows():
            title = row.get("title", "")
            content = row.get("content", "")
            if title:
                all_news.append(str(title))
            if content and len(str(content)) < 500:
                all_news.append(str(content)[:200])
        
        if not all_news:
            return ""
        
        # 合并新闻文本
        combined = " | ".join(all_news)
        
        # 截断
        if len(combined) > 1500:
            combined = combined[:1500] + "..."
        
        logger.debug(f"获取股票新闻成功: {stock_code}, {len(all_news)} 条")
        return combined
    
    # ==================== 交易日历 ====================
    
    def fetch_trade_calendar(
        self,
        start_date: str,
        end_date: str,
        exchange: str = "SSE"
    ) -> pd.DatetimeIndex:
        """
        获取交易日历
        
        Parameters
        ----------
        start_date : str
            开始日期，格式 YYYY-MM-DD 或 YYYYMMDD
        end_date : str
            结束日期，格式 YYYY-MM-DD 或 YYYYMMDD
        exchange : str
            交易所，SSE(上交所，默认) 或 SZSE(深交所)
        
        Returns
        -------
        pd.DatetimeIndex
            交易日期索引
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> calendar = loader.fetch_trade_calendar("2024-01-01", "2024-12-31")
        >>> print(f"2024年共 {len(calendar)} 个交易日")
        """
        # 标准化日期格式
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        logger.info(f"获取交易日历: {start_date} ~ {end_date}")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"trade_cal_{start_date[:4]}.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                # 过滤日期范围
                df = df[
                    (df["cal_date"] >= start_date) & 
                    (df["cal_date"] <= end_date) &
                    (df["is_open"] == 1)
                ]
                if not df.empty:
                    calendar = pd.to_datetime(df["cal_date"])
                    logger.debug(f"从缓存加载交易日历: {len(calendar)} 天")
                    return pd.DatetimeIndex(sorted(calendar))
            except Exception:
                pass
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.trade_cal,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            logger.warning("无法获取交易日历，使用工作日近似")
            return pd.bdate_range(start=start_date, end=end_date)
        
        # 保存缓存（整年数据）
        try:
            full_year_df = self._fetch_with_retry(
                self.pro.trade_cal,
                exchange=exchange,
                start_date=f"{start_date[:4]}0101",
                end_date=f"{start_date[:4]}1231"
            )
            if full_year_df is not None and not full_year_df.empty:
                full_year_df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        # 过滤交易日
        trade_days = df[df["is_open"] == 1]["cal_date"]
        calendar = pd.to_datetime(trade_days)
        calendar = pd.DatetimeIndex(sorted(calendar))
        calendar.name = "date"
        
        logger.info(f"获取交易日历成功: {len(calendar)} 个交易日")
        return calendar
    
    def is_trade_day(self, date: Optional[str] = None) -> bool:
        """
        判断指定日期是否为交易日
        
        Parameters
        ----------
        date : Optional[str]
            日期，格式 YYYY-MM-DD 或 YYYYMMDD，默认今天
        
        Returns
        -------
        bool
            是否为交易日
        """
        from datetime import datetime
        
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        else:
            date = date.replace("-", "")
        
        calendar = self.fetch_trade_calendar(date, date)
        return len(calendar) > 0
    
    # ==================== 行业分类 ====================
    
    def fetch_industry_mapping(
        self,
        use_cache: bool = True
    ) -> Dict[str, str]:
        """
        获取股票行业分类映射
        
        返回股票代码到行业名称的映射字典。
        
        Parameters
        ----------
        use_cache : bool
            是否使用缓存，默认 True
        
        Returns
        -------
        Dict[str, str]
            股票代码（6位）到行业名称的映射
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> industry_map = loader.fetch_industry_mapping()
        >>> print(industry_map.get("000001"))  # 银行
        """
        logger.info("获取股票行业分类映射")
        
        # 尝试缓存
        today = datetime.now().strftime("%Y%m%d")
        cache_file = self.cache_dir / f"industry_mapping_{today[:6]}.parquet"
        
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                mapping = dict(zip(df["stock_code"], df["industry"]))
                logger.info(f"从缓存加载行业映射: {len(mapping)} 只股票")
                return mapping
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.stock_basic,
            list_status="L",
            fields="ts_code,symbol,name,industry,market,list_date"
        )
        
        if df is None or df.empty:
            logger.warning("无法获取行业分类数据")
            return {}
        
        # 提取 6 位股票代码
        df["stock_code"] = df["ts_code"].str[:6]
        
        # 保存缓存
        try:
            df[["stock_code", "industry"]].to_parquet(cache_file, index=False)
            logger.info(f"行业映射已缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
        
        # 构建映射
        mapping = dict(zip(df["stock_code"], df["industry"]))
        logger.info(f"获取行业映射成功: {len(mapping)} 只股票")
        return mapping
    
    def fetch_sw_industry_mapping(
        self,
        level: int = 1
    ) -> Dict[str, str]:
        """
        获取申万行业分类映射
        
        Parameters
        ----------
        level : int
            行业分类级别：1(一级), 2(二级), 3(三级)
            默认 1（一级行业）
        
        Returns
        -------
        Dict[str, str]
            股票代码（6位）到申万行业名称的映射
        
        Notes
        -----
        申万行业分类是 A 股最常用的行业分类标准。
        Tushare 需要较高权限才能使用申万行业接口。
        """
        logger.info(f"获取申万 {level} 级行业分类")
        
        # 尝试使用 stock_basic 的 industry 字段（通用行业分类）
        # 如果需要精确的申万分类，需要使用 index_member 接口
        
        try:
            # 尝试获取申万指数成分
            df = self._fetch_with_retry(
                self.pro.index_classify,
                level=f"L{level}",
                src="SW"
            )
            
            if df is not None and not df.empty:
                # 获取每个行业的成分股
                result = {}
                for _, row in df.iterrows():
                    index_code = row.get("index_code", "")
                    industry_name = row.get("industry_name", "")
                    
                    if index_code:
                        members = self._fetch_with_retry(
                            self.pro.index_member,
                            index_code=index_code
                        )
                        if members is not None and not members.empty:
                            for stock in members["con_code"].str[:6]:
                                result[stock] = industry_name
                
                if result:
                    logger.info(f"获取申万行业分类成功: {len(result)} 只股票")
                    return result
                    
        except Exception as e:
            logger.debug(f"申万分类接口不可用: {e}")
        
        # 降级到普通行业分类
        logger.info("使用普通行业分类替代申万分类")
        return self.fetch_industry_mapping()
    
    # ==================== 资金流向与北向资金 ====================
    
    def fetch_moneyflow(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        获取个股资金流向数据（大单/超大单）
        
        使用 Tushare Pro moneyflow 接口获取个股资金流向明细数据，
        包含大单、超大单、中单、小单的买入卖出金额。
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位），如 "000001"
        start_date : str
            开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date : str
            结束日期，格式 YYYYMMDD 或 YYYY-MM-DD
        
        Returns
        -------
        Optional[pd.DataFrame]
            资金流向数据，包含：
            - trade_date: 交易日期
            - buy_elg_vol: 特大单买入量（手）
            - buy_elg_amount: 特大单买入额（万元）
            - sell_elg_vol: 特大单卖出量（手）
            - sell_elg_amount: 特大单卖出额（万元）
            - buy_lg_vol: 大单买入量（手）
            - buy_lg_amount: 大单买入额（万元）
            - sell_lg_vol: 大单卖出量（手）
            - sell_lg_amount: 大单卖出额（万元）
            - net_mf_vol: 净流入量（手）
            - net_mf_amount: 净流入额（万元）
            失败返回 None
        
        Notes
        -----
        - Tushare Pro moneyflow 接口需要较高积分权限（约 2000 积分）
        - 如无权限可考虑使用 moneyflow_hsgt 替代部分功能
        - 大单定义：50万-100万为大单，100万以上为特大单
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> flow = loader.fetch_moneyflow("000001", "20240101", "20240115")
        >>> # 计算主力净流入（大单+超大单）
        >>> main_net = flow['buy_elg_amount'] - flow['sell_elg_amount'] + \\
        ...            flow['buy_lg_amount'] - flow['sell_lg_amount']
        """
        # 标准化股票代码
        ts_code = self._to_ts_code(stock_code)
        
        # 标准化日期格式
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        logger.debug(f"获取资金流向: {stock_code}, {start_date} ~ {end_date}")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"moneyflow_{stock_code}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载资金流向: {stock_code}")
                    return df
            except Exception:
                pass
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.moneyflow,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取资金流向失败: {stock_code}")
            return None
        
        # 添加 6 位股票代码
        df["stock_code"] = df["ts_code"].str[:6]
        
        # 日期标准化
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取资金流向成功: {stock_code}, {len(df)} 条")
        return df
    
    def fetch_moneyflow_batch(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        show_progress: bool = True,
        batch_size: int = 150,
        batch_sleep: float = 8.0
    ) -> pd.DataFrame:
        """
        批量获取资金流向数据
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        show_progress : bool
            是否显示进度条
        batch_size : int
            每批次处理的股票数量
        batch_sleep : float
            每批次之间的休息时间（秒）
        
        Returns
        -------
        pd.DataFrame
            合并后的资金流向数据
        """
        all_data = []
        total = len(stock_list)
        success_count = 0
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(stock_list),
                    total=total,
                    desc="💰 获取资金流向",
                    unit="只",
                    ncols=80
                )
            except ImportError:
                iterator = enumerate(stock_list)
                logger.info(f"开始获取资金流向: {total} 只股票...")
        else:
            iterator = enumerate(stock_list)
        
        for i, stock in iterator:
            df = self.fetch_moneyflow(stock, start_date, end_date)
            if df is not None and not df.empty:
                all_data.append(df)
                success_count += 1
            
            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({"成功": success_count, "当前": stock})
            
            # 批次休息
            if (i + 1) % batch_size == 0 and (i + 1) < total:
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description(f"💰 休息{batch_sleep}s")
                time.sleep(batch_sleep)
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description("💰 获取资金流向")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"批量获取资金流向完成: {success_count}/{total} 只, {len(result)} 条记录")
        return result
    
    def fetch_hk_hold(
        self,
        trade_date: str,
        stock_code: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取沪深港通持股数据（北向资金持仓）
        
        使用 Tushare Pro hk_hold 接口获取北向资金（沪股通/深股通）
        在 A 股的持股明细数据，是追踪"真外资"的核心数据。
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD 或 YYYY-MM-DD
        stock_code : Optional[str]
            股票代码（6位），如果提供则只返回该股票的数据
        
        Returns
        -------
        Optional[pd.DataFrame]
            北向持仓数据，包含：
            - trade_date: 交易日期
            - ts_code: 股票代码
            - stock_code: 6位股票代码
            - name: 股票名称
            - vol: 持股数量（股）
            - ratio: 持股占比（%）
            - exchange: 交易所（SH/SZ）
            失败返回 None
        
        Notes
        -----
        - 数据为 T+1 披露，即当日持仓需次日获取
        - 该数据极其精准，能有效过滤盘中的假外资
        - 建议关注持仓占比的环比变化（hk_hold_ratio_change）
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> # 获取全市场北向持仓
        >>> hk_data = loader.fetch_hk_hold("20240115")
        >>> # 计算持仓占比变化
        >>> today = loader.fetch_hk_hold("20240115")
        >>> yesterday = loader.fetch_hk_hold("20240114")
        >>> merged = today.merge(yesterday, on='stock_code', suffixes=('', '_prev'))
        >>> merged['ratio_change'] = merged['ratio'] - merged['ratio_prev']
        """
        # 标准化日期格式
        trade_date = trade_date.replace("-", "")
        
        logger.debug(f"获取北向持仓: {trade_date}")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"hk_hold_{trade_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载北向持仓: {trade_date}, {len(df)} 条")
                    if stock_code:
                        df = df[df["stock_code"] == stock_code[:6]]
                    return df
            except Exception:
                pass
        
        # API 获取（获取全市场数据）
        df = self._fetch_with_retry(
            self.pro.hk_hold,
            trade_date=trade_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取北向持仓失败: {trade_date}")
            return None
        
        # 添加 6 位股票代码
        df["stock_code"] = df["ts_code"].str[:6]
        
        # 保存缓存（全量数据）
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取北向持仓成功: {trade_date}, {len(df)} 条")
        
        # 如果指定了股票代码，过滤
        if stock_code:
            df = df[df["stock_code"] == stock_code[:6]]
        
        return df
    
    def fetch_hk_hold_change(
        self,
        stock_list: List[str],
        current_date: str,
        days_back: int = 5
    ) -> pd.DataFrame:
        """
        计算北向资金持仓占比变化
        
        获取指定股票列表的北向资金持仓变化数据，
        用于构建 SmartMoney 因子中的北向穿透因子。
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表（6位代码）
        current_date : str
            当前日期，格式 YYYYMMDD 或 YYYY-MM-DD
        days_back : int
            回溯天数，用于计算环比变化，默认 5 天
        
        Returns
        -------
        pd.DataFrame
            包含以下列：
            - stock_code: 股票代码
            - hk_ratio: 当前北向持仓占比
            - hk_ratio_prev: 前期北向持仓占比
            - hk_ratio_change: 持仓占比变化（百分点）
            - hk_hold_score: 标准化后的北向穿透得分 (0-1)
        
        Notes
        -----
        - 如果某股票无北向持仓数据，其得分默认为 0.5（中性）
        - hk_hold_score 使用排名分位数标准化，避免极端值影响
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> stocks = ["000001", "000002", "600000"]
        >>> hk_change = loader.fetch_hk_hold_change(stocks, "20240115")
        >>> # 筛选北向加仓明显的股票
        >>> bullish = hk_change[hk_change['hk_ratio_change'] > 0.1]
        """
        current_date = current_date.replace("-", "")
        
        # 获取交易日历
        start_date = (
            datetime.strptime(current_date, "%Y%m%d") - timedelta(days=days_back + 10)
        ).strftime("%Y%m%d")
        
        calendar = self.fetch_trade_calendar(start_date, current_date)
        
        if len(calendar) < 2:
            logger.warning("交易日不足，无法计算北向持仓变化")
            return pd.DataFrame()
        
        # 获取最近两个交易日的数据
        current_trade_date = calendar[-1].strftime("%Y%m%d")
        prev_trade_date = calendar[max(0, len(calendar) - days_back - 1)].strftime("%Y%m%d")
        
        logger.info(f"计算北向持仓变化: {prev_trade_date} -> {current_trade_date}")
        
        # 获取两日的北向持仓数据
        current_hold = self.fetch_hk_hold(current_trade_date)
        prev_hold = self.fetch_hk_hold(prev_trade_date)
        
        # 初始化结果 DataFrame
        result = pd.DataFrame({"stock_code": stock_list})
        result["hk_ratio"] = np.nan
        result["hk_ratio_prev"] = np.nan
        result["hk_ratio_change"] = 0.0
        
        # 合并当前持仓数据
        if current_hold is not None and not current_hold.empty:
            current_hold_subset = current_hold[["stock_code", "ratio"]].rename(
                columns={"ratio": "hk_ratio"}
            )
            result = result.merge(
                current_hold_subset, on="stock_code", how="left", suffixes=("_drop", "")
            )
            if "hk_ratio_drop" in result.columns:
                result["hk_ratio"] = result["hk_ratio"].fillna(result["hk_ratio_drop"])
                result = result.drop(columns=["hk_ratio_drop"])
        
        # 合并前期持仓数据
        if prev_hold is not None and not prev_hold.empty:
            prev_hold_subset = prev_hold[["stock_code", "ratio"]].rename(
                columns={"ratio": "hk_ratio_prev"}
            )
            result = result.merge(
                prev_hold_subset, on="stock_code", how="left", suffixes=("_drop", "")
            )
            if "hk_ratio_prev_drop" in result.columns:
                result["hk_ratio_prev"] = result["hk_ratio_prev"].fillna(
                    result["hk_ratio_prev_drop"]
                )
                result = result.drop(columns=["hk_ratio_prev_drop"])
        
        # 计算变化
        result["hk_ratio_change"] = (
            result["hk_ratio"].fillna(0) - result["hk_ratio_prev"].fillna(0)
        )
        
        # 计算标准化得分（使用排名分位数）
        valid_mask = result["hk_ratio_change"].notna()
        if valid_mask.sum() > 0:
            result.loc[valid_mask, "hk_hold_score"] = (
                result.loc[valid_mask, "hk_ratio_change"].rank(pct=True)
            )
        else:
            result["hk_hold_score"] = 0.5
        
        # 填充缺失值
        result["hk_hold_score"] = result["hk_hold_score"].fillna(0.5)
        
        logger.info(f"北向持仓变化计算完成: {len(result)} 只股票")
        return result
    
    def calculate_smart_money_score(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        north_weight: float = 0.6,
        large_order_weight: float = 0.4
    ) -> pd.DataFrame:
        """
        计算全息主力资金因子 (Holographic Smart Money Score)
        
        综合北向资金穿透因子和内资大单流向因子，构建复合主力资金得分。
        在火热行情中，该因子可有效跟踪"聪明钱"的流向。
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表（6位代码）
        start_date : str
            开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date : str
            结束日期（用于计算北向持仓变化的参考日）
        north_weight : float
            北向资金因子权重，默认 0.6
        large_order_weight : float
            大单流向因子权重，默认 0.4
        
        Returns
        -------
        pd.DataFrame
            包含以下列：
            - stock_code: 股票代码
            - north_score: 北向资金得分 (0-1)
            - large_order_score: 大单流向得分 (0-1)
            - smart_money_score: 综合主力资金得分 (0-1)
            - main_net_inflow: 主力净流入额（万元）
            - hk_ratio_change: 北向持仓占比变化
        
        Notes
        -----
        SmartMoney_Score = north_weight * North_Score + large_order_weight * Large_Order_Score
        
        因子逻辑：
        1. 北向穿透（真外资）：看北向资金持仓占比的环比变化，T+1 数据但极其精准
        2. 内资主力（大单）：大单/超大单的净流入率
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> stocks = loader.fetch_index_constituents("hs300")
        >>> smart_money = loader.calculate_smart_money_score(
        ...     stocks, "20240101", "20240115"
        ... )
        >>> # 筛选主力资金流入强势的股票
        >>> bullish = smart_money[smart_money['smart_money_score'] > 0.7]
        """
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        logger.info(f"📊 计算全息主力资金因子: {len(stock_list)} 只股票")
        
        # 1. 获取北向资金变化得分
        hk_change = self.fetch_hk_hold_change(
            stock_list=stock_list,
            current_date=end_date,
            days_back=5
        )
        
        # 2. 获取资金流向数据
        moneyflow_df = self.fetch_moneyflow_batch(
            stock_list=stock_list,
            start_date=start_date,
            end_date=end_date,
            show_progress=True
        )
        
        # 3. 计算大单流向得分
        if not moneyflow_df.empty:
            # 计算主力净流入 = (特大单买入 - 特大单卖出) + (大单买入 - 大单卖出)
            flow_cols = ["buy_elg_amount", "sell_elg_amount", "buy_lg_amount", "sell_lg_amount"]
            if all(col in moneyflow_df.columns for col in flow_cols):
                moneyflow_df["main_net_inflow"] = (
                    moneyflow_df["buy_elg_amount"] - moneyflow_df["sell_elg_amount"] +
                    moneyflow_df["buy_lg_amount"] - moneyflow_df["sell_lg_amount"]
                )
            elif "net_mf_amount" in moneyflow_df.columns:
                # 使用净流入额字段
                moneyflow_df["main_net_inflow"] = moneyflow_df["net_mf_amount"]
            else:
                moneyflow_df["main_net_inflow"] = 0
            
            # 按股票汇总最近的主力净流入
            flow_summary = moneyflow_df.groupby("stock_code").agg({
                "main_net_inflow": "sum"
            }).reset_index()
            
            # 标准化大单流向得分（排名分位数）
            if len(flow_summary) > 0:
                flow_summary["large_order_score"] = flow_summary["main_net_inflow"].rank(pct=True)
            else:
                flow_summary["large_order_score"] = 0.5
        else:
            flow_summary = pd.DataFrame({
                "stock_code": stock_list,
                "main_net_inflow": 0,
                "large_order_score": 0.5
            })
        
        # 4. 合并结果
        result = pd.DataFrame({"stock_code": stock_list})
        
        # 合并北向得分
        if not hk_change.empty:
            result = result.merge(
                hk_change[["stock_code", "hk_hold_score", "hk_ratio_change"]],
                on="stock_code",
                how="left"
            )
            result["north_score"] = result["hk_hold_score"].fillna(0.5)
        else:
            result["north_score"] = 0.5
            result["hk_ratio_change"] = 0.0
        
        # 合并大单得分
        result = result.merge(
            flow_summary[["stock_code", "large_order_score", "main_net_inflow"]],
            on="stock_code",
            how="left"
        )
        result["large_order_score"] = result["large_order_score"].fillna(0.5)
        result["main_net_inflow"] = result["main_net_inflow"].fillna(0)
        
        # 5. 计算综合得分
        result["smart_money_score"] = (
            north_weight * result["north_score"] +
            large_order_weight * result["large_order_score"]
        )
        
        # 清理中间列
        if "hk_hold_score" in result.columns:
            result = result.drop(columns=["hk_hold_score"])
        
        logger.info(
            f"✅ 主力资金因子计算完成: "
            f"均值={result['smart_money_score'].mean():.3f}, "
            f"top10均值={result.nlargest(10, 'smart_money_score')['smart_money_score'].mean():.3f}"
        )
        
        return result
    
    # ==================== 涨停板与龙头因子 ====================
    
    def fetch_limit_list(
        self,
        trade_date: str,
        limit_type: str = "U"
    ) -> Optional[pd.DataFrame]:
        """
        获取每日涨跌停股票列表
        
        使用 Tushare Pro limit_list 接口获取每日涨停/跌停股票明细，
        包含封单金额、封单比例、振幅、炸板次数等微观结构数据。
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD 或 YYYY-MM-DD
        limit_type : str
            涨跌停类型：
            - "U": 涨停（默认）
            - "D": 跌停
        
        Returns
        -------
        Optional[pd.DataFrame]
            涨跌停明细数据，包含：
            - trade_date: 交易日期
            - ts_code: 股票代码
            - stock_code: 6位股票代码
            - name: 股票名称
            - close: 收盘价
            - pct_chg: 涨跌幅
            - amp: 振幅(%)
            - fc_ratio: 封成比(%)（封单金额/成交额）
            - fl_ratio: 封流比(%)（封单手数/流通股本）
            - fd_amount: 封单金额（万元）
            - first_time: 首次涨停时间
            - last_time: 最后涨停时间
            - open_times: 打开次数
            - strth: 涨停强度（0-100）
            - limit: 涨跌停价格
            失败返回 None
        
        Notes
        -----
        - Tushare Pro limit_list 接口需要较高积分权限（约 2000 积分）
        - 涨停强度(strth)：综合考虑封单、振幅、炸板次数的强度评分
        - fc_ratio 封成比越大，说明买盘越坚决，次日高开概率大
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> limits = loader.fetch_limit_list("20240115")
        >>> # 筛选强势涨停（未炸板、封成比>10%）
        >>> strong = limits[(limits['open_times'] == 0) & (limits['fc_ratio'] > 10)]
        """
        # 标准化日期格式
        trade_date = trade_date.replace("-", "")
        
        logger.debug(f"获取涨跌停列表: {trade_date}, 类型={limit_type}")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"limit_list_{trade_date}_{limit_type}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载涨跌停列表: {trade_date}, {len(df)} 条")
                    return df
            except Exception:
                pass
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.limit_list,
            trade_date=trade_date,
            limit_type=limit_type
        )
        
        if df is None or df.empty:
            logger.debug(f"获取涨跌停列表失败: {trade_date}")
            return None
        
        # 添加 6 位股票代码
        df["stock_code"] = df["ts_code"].str[:6]
        
        # 重命名部分字段以统一
        rename_map = {
            "open_times": "open_num",  # 炸板次数
        }
        df = df.rename(columns=rename_map)
        
        # 确保关键字段存在
        if "open_num" not in df.columns:
            df["open_num"] = 0
        if "fc_ratio" not in df.columns and "fd_amount" in df.columns and "amount" in df.columns:
            # 计算封成比
            df["fc_ratio"] = df["fd_amount"] / df["amount"].replace(0, np.nan) * 100
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取涨跌停列表成功: {trade_date}, {len(df)} 条")
        return df
    
    def fetch_limit_list_batch(
        self,
        start_date: str,
        end_date: str,
        limit_type: str = "U",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        批量获取多日涨跌停列表
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        limit_type : str
            涨跌停类型："U"(涨停) 或 "D"(跌停)
        show_progress : bool
            是否显示进度条
        
        Returns
        -------
        pd.DataFrame
            合并后的涨跌停数据
        """
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        # 获取交易日历
        calendar = self.fetch_trade_calendar(start_date, end_date)
        
        all_data = []
        total = len(calendar)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    calendar,
                    desc="🔥 获取涨停数据",
                    unit="天",
                    ncols=80
                )
            except ImportError:
                iterator = calendar
                logger.info(f"开始获取涨停数据: {total} 个交易日...")
        else:
            iterator = calendar
        
        for date in iterator:
            date_str = date.strftime("%Y%m%d")
            df = self.fetch_limit_list(date_str, limit_type)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"批量获取涨停数据完成: {len(calendar)} 天, {len(result)} 条记录")
        return result
    
    def calculate_consecutive_limits(
        self,
        stock_code: str,
        end_date: str,
        days_back: int = 30
    ) -> int:
        """
        计算股票的连板天数
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位）
        end_date : str
            截止日期
        days_back : int
            回溯天数，默认 30 天
        
        Returns
        -------
        int
            连续涨停天数（0 表示非涨停或已中断）
        
        Notes
        -----
        只统计到 end_date 为止的连续涨停天数，
        如果 end_date 当天未涨停，返回 0。
        """
        end_date = end_date.replace("-", "")
        start_date = (
            datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days_back)
        ).strftime("%Y%m%d")
        
        # 获取交易日历
        calendar = self.fetch_trade_calendar(start_date, end_date)
        if len(calendar) == 0:
            return 0
        
        # 从最近的交易日向前回溯
        consecutive_count = 0
        
        for date in reversed(calendar):
            date_str = date.strftime("%Y%m%d")
            limit_df = self.fetch_limit_list(date_str, limit_type="U")
            
            if limit_df is None or limit_df.empty:
                if consecutive_count == 0:
                    continue  # 可能是非交易日数据缺失
                else:
                    break  # 连板中断
            
            # 检查该股票是否在涨停列表中
            if stock_code in limit_df["stock_code"].values:
                consecutive_count += 1
            else:
                if consecutive_count > 0:
                    break  # 连板中断
                # 如果从未涨停，继续向前检查（可能end_date当天未涨停）
                # 但如果已经回溯超过5天还没涨停，直接返回0
                if consecutive_count == 0:
                    date_diff = (datetime.strptime(end_date, "%Y%m%d") - date).days
                    if date_diff > 5:
                        return 0
        
        return consecutive_count
    
    def calculate_limit_strength(
        self,
        trade_date: str,
        min_fl_ratio: float = 1.0
    ) -> pd.DataFrame:
        """
        计算涨停封板强度因子（龙头信仰因子）
        
        针对"连板"和"涨停"股票的微观结构分析，
        用于捕捉强势股的溢价预期。
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD 或 YYYY-MM-DD
        min_fl_ratio : float
            最低封流比阈值（%），默认 1.0
        
        Returns
        -------
        pd.DataFrame
            涨停强度因子数据，包含：
            - stock_code: 股票代码
            - name: 股票名称
            - close: 收盘价
            - pct_chg: 涨跌幅
            - fd_amount: 封单金额（万元）
            - bid_strength: 封成比（封单金额/成交额）
            - fl_ratio: 封流比（封单/流通股本）
            - open_num: 炸板次数
            - is_strong_limit: 是否强势涨停
            - dragon_score: 龙头信仰得分 (0-1)
        
        Notes
        -----
        计算逻辑：
        1. 筛选涨停：只分析收盘涨停的股票
        2. 封板强度：封单金额 / 日成交额，比值越大买盘越坚决
        3. 质量过滤：封流比 > 1% 且炸板次数 == 0 为强势板
        4. 弱势降权：非强势板得分减半
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> dragon = loader.calculate_limit_strength("20240115")
        >>> # 筛选高龙头得分的股票
        >>> top_dragons = dragon[dragon['dragon_score'] > 0.8]
        """
        trade_date = trade_date.replace("-", "")
        
        logger.info(f"🐉 计算龙头信仰因子: {trade_date}")
        
        # 获取涨停列表
        limit_df = self.fetch_limit_list(trade_date, limit_type="U")
        
        if limit_df is None or limit_df.empty:
            logger.warning(f"无涨停数据: {trade_date}")
            return pd.DataFrame()
        
        result = limit_df.copy()
        
        # 1. 计算封成比 (封单金额 / 当日成交额)
        # 反映市场惜售程度和抢筹意愿
        if "fd_amount" in result.columns and "amount" in result.columns:
            result["bid_strength"] = (
                result["fd_amount"] / result["amount"].replace(0, np.nan)
            )
        elif "fc_ratio" in result.columns:
            # 如果已有封成比字段，转换为小数
            result["bid_strength"] = result["fc_ratio"] / 100
        else:
            result["bid_strength"] = 0.5  # 默认值
        
        # 2. 识别强势涨停
        # 条件：封流比 > min_fl_ratio% 且 炸板次数 == 0
        if "fl_ratio" not in result.columns:
            result["fl_ratio"] = 0
        if "open_num" not in result.columns:
            result["open_num"] = 0
        
        result["is_strong_limit"] = (
            (result["fl_ratio"] >= min_fl_ratio) & 
            (result["open_num"] == 0)
        )
        
        # 3. 计算龙头得分
        # 基础得分 = 封成比的标准化
        result["dragon_score"] = result["bid_strength"].rank(pct=True)
        
        # 对弱势板（炸板或封单弱）降权 50%
        result.loc[~result["is_strong_limit"], "dragon_score"] *= 0.5
        
        # 确保得分在 [0, 1] 范围内
        result["dragon_score"] = result["dragon_score"].clip(0, 1)
        
        # 填充缺失值
        result["dragon_score"] = result["dragon_score"].fillna(0)
        
        # 选择输出列
        output_cols = [
            "stock_code", "ts_code", "name", "close", "pct_chg",
            "fd_amount", "amount", "bid_strength", "fl_ratio", 
            "open_num", "is_strong_limit", "dragon_score"
        ]
        output_cols = [c for c in output_cols if c in result.columns]
        
        result = result[output_cols].copy()
        
        # 按龙头得分排序
        result = result.sort_values("dragon_score", ascending=False)
        
        strong_count = result["is_strong_limit"].sum()
        logger.info(
            f"✅ 龙头因子计算完成: {len(result)} 只涨停, "
            f"{strong_count} 只强势板, "
            f"top5得分={result['dragon_score'].head(5).mean():.3f}"
        )
        
        return result
    
    def calculate_dragon_head_factor(
        self,
        trade_date: str,
        days_back: int = 5,
        consecutive_weight: float = 0.3,
        strength_weight: float = 0.7
    ) -> pd.DataFrame:
        """
        计算完整龙头信仰因子（含连板溢价）
        
        综合封板强度和连板天数，构建完整的龙头信仰因子。
        
        Parameters
        ----------
        trade_date : str
            交易日期
        days_back : int
            计算连板天数时的回溯天数，默认 5 天
        consecutive_weight : float
            连板溢价权重，默认 0.3
        strength_weight : float
            封板强度权重，默认 0.7
        
        Returns
        -------
        pd.DataFrame
            龙头信仰因子数据，包含：
            - stock_code: 股票代码
            - dragon_score: 封板强度得分
            - consecutive_days: 连板天数
            - consecutive_score: 连板溢价得分
            - dragon_head_factor: 综合龙头因子
        
        Notes
        -----
        综合因子计算：
        Dragon_Head_Factor = strength_weight * dragon_score + 
                             consecutive_weight * consecutive_score
        
        连板溢价逻辑：
        - 1连板（首板）: 1.0
        - 2连板: 1.5
        - 3连板: 2.0
        - 4连板及以上: 2.5 + 0.2 * (n - 4)
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> dragon = loader.calculate_dragon_head_factor("20240115")
        >>> # 获取高位龙头（连板+强封单）
        >>> high_dragons = dragon[
        ...     (dragon['consecutive_days'] >= 2) & 
        ...     (dragon['dragon_head_factor'] > 0.8)
        ... ]
        """
        trade_date = trade_date.replace("-", "")
        
        logger.info(f"🐲 计算完整龙头信仰因子: {trade_date}")
        
        # 1. 获取封板强度因子
        strength_df = self.calculate_limit_strength(trade_date)
        
        if strength_df.empty:
            return pd.DataFrame()
        
        result = strength_df.copy()
        
        # 2. 计算连板天数
        logger.info(f"计算连板天数: {len(result)} 只股票...")
        
        consecutive_days_list = []
        for stock in result["stock_code"]:
            cons_days = self.calculate_consecutive_limits(
                stock, trade_date, days_back=days_back
            )
            consecutive_days_list.append(cons_days)
        
        result["consecutive_days"] = consecutive_days_list
        
        # 3. 计算连板溢价得分
        # 连板越多，溢价预期越高
        def calc_consecutive_score(days: int) -> float:
            """连板溢价函数"""
            if days <= 0:
                return 0.0
            elif days == 1:
                return 1.0
            elif days == 2:
                return 1.5
            elif days == 3:
                return 2.0
            else:
                return 2.5 + 0.2 * (days - 4)
        
        result["consecutive_premium"] = result["consecutive_days"].apply(calc_consecutive_score)
        
        # 标准化连板得分到 [0, 1]
        max_premium = result["consecutive_premium"].max()
        if max_premium > 0:
            result["consecutive_score"] = result["consecutive_premium"] / max_premium
        else:
            result["consecutive_score"] = 0
        
        # 4. 计算综合龙头因子
        result["dragon_head_factor"] = (
            strength_weight * result["dragon_score"] +
            consecutive_weight * result["consecutive_score"]
        )
        
        # 确保因子在 [0, 1] 范围
        result["dragon_head_factor"] = result["dragon_head_factor"].clip(0, 1)
        
        # 按综合因子排序
        result = result.sort_values("dragon_head_factor", ascending=False)
        
        # 统计信息
        multi_limit = (result["consecutive_days"] >= 2).sum()
        logger.info(
            f"✅ 龙头因子计算完成: "
            f"{len(result)} 只涨停, {multi_limit} 只连板, "
            f"最高连板={result['consecutive_days'].max()}天"
        )
        
        return result
    
    def get_dragon_candidates(
        self,
        trade_date: str,
        min_consecutive: int = 1,
        min_factor: float = 0.6,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        获取龙头候选股
        
        筛选满足条件的高质量龙头股，用于短线交易参考。
        
        Parameters
        ----------
        trade_date : str
            交易日期
        min_consecutive : int
            最低连板天数，默认 1（首板）
        min_factor : float
            最低龙头因子得分，默认 0.6
        top_n : int
            返回股票数量上限，默认 20
        
        Returns
        -------
        pd.DataFrame
            筛选后的龙头候选股列表
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> candidates = loader.get_dragon_candidates(
        ...     "20240115",
        ...     min_consecutive=2,  # 至少2连板
        ...     min_factor=0.7
        ... )
        >>> print(candidates[['stock_code', 'name', 'consecutive_days', 'dragon_head_factor']])
        """
        dragon_df = self.calculate_dragon_head_factor(trade_date)
        
        if dragon_df.empty:
            return pd.DataFrame()
        
        # 筛选条件
        mask = (
            (dragon_df["consecutive_days"] >= min_consecutive) &
            (dragon_df["dragon_head_factor"] >= min_factor) &
            (dragon_df["is_strong_limit"] == True)
        )
        
        candidates = dragon_df[mask].head(top_n).copy()
        
        logger.info(
            f"🎯 龙头候选股筛选完成: "
            f"{len(candidates)} 只 (条件: 连板>={min_consecutive}, 因子>={min_factor})"
        )
        
        return candidates
    
    # ==================== 融资融券与杠杆因子 ====================
    
    def fetch_margin_detail(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        获取个股融资融券明细数据
        
        使用 Tushare Pro margin_detail 接口获取个股融资融券交易明细，
        包含融资买入、融资偿还、融券卖出等数据。
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位），如 "000001"
        start_date : str
            开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date : str
            结束日期，格式 YYYYMMDD 或 YYYY-MM-DD
        
        Returns
        -------
        Optional[pd.DataFrame]
            融资融券明细数据，包含：
            - trade_date: 交易日期
            - ts_code: 股票代码
            - stock_code: 6位股票代码
            - rzye: 融资余额（元）
            - rzmre: 融资买入额（元）
            - rzche: 融资偿还额（元）
            - rqye: 融券余额（元）
            - rqyl: 融券余量（股）
            - rqmcl: 融券卖出量（股）
            - rqchl: 融券偿还量（股）
            失败返回 None
        
        Notes
        -----
        - Tushare Pro margin_detail 接口需要较高积分权限
        - 融资数据是判断散户杠杆情绪的重要指标
        - 当融资买入占比过高时，往往是阶段性顶部信号
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> margin = loader.fetch_margin_detail("000001", "20240101", "20240115")
        >>> # 计算融资买入占比趋势
        >>> print(margin[['trade_date', 'rzye', 'rzmre']].tail())
        """
        # 标准化股票代码
        ts_code = self._to_ts_code(stock_code)
        
        # 标准化日期格式
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        logger.debug(f"获取融资融券数据: {stock_code}, {start_date} ~ {end_date}")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"margin_{stock_code}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载融资融券数据: {stock_code}")
                    return df
            except Exception:
                pass
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.margin_detail,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取融资融券数据失败: {stock_code}")
            return None
        
        # 添加 6 位股票代码
        df["stock_code"] = df["ts_code"].str[:6]
        
        # 日期标准化
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取融资融券数据成功: {stock_code}, {len(df)} 条")
        return df
    
    def fetch_margin(
        self,
        trade_date: str
    ) -> Optional[pd.DataFrame]:
        """
        获取全市场融资融券汇总数据
        
        使用 Tushare Pro margin 接口获取全市场融资融券汇总，
        是快速获取市场杠杆情绪的高效方式。
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD 或 YYYY-MM-DD
        
        Returns
        -------
        Optional[pd.DataFrame]
            全市场融资融券汇总数据，包含：
            - trade_date: 交易日期
            - ts_code: 股票代码
            - stock_code: 6位股票代码
            - rzye: 融资余额（元）
            - rzmre: 融资买入额（元）
            - rzche: 融资偿还额（元）
            - rqye: 融券余额（元）
            - rqmcl: 融券卖出量（股）
            - rzrqye: 融资融券余额（元）
            失败返回 None
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> margin = loader.fetch_margin("20240115")
        >>> # 筛选融资余额最高的股票
        >>> top_margin = margin.nlargest(20, 'rzye')
        """
        trade_date = trade_date.replace("-", "")
        
        logger.debug(f"获取全市场融资融券数据: {trade_date}")
        
        # 尝试缓存
        cache_file = self.cache_dir / f"margin_{trade_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载融资融券数据: {trade_date}, {len(df)} 条")
                    return df
            except Exception:
                pass
        
        # API 获取
        df = self._fetch_with_retry(
            self.pro.margin,
            trade_date=trade_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取融资融券数据失败: {trade_date}")
            return None
        
        # 添加 6 位股票代码
        df["stock_code"] = df["ts_code"].str[:6]
        
        # 保存缓存
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取融资融券数据成功: {trade_date}, {len(df)} 条")
        return df
    
    def fetch_margin_batch(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        show_progress: bool = True,
        batch_size: int = 150,
        batch_sleep: float = 8.0
    ) -> pd.DataFrame:
        """
        批量获取融资融券数据
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        show_progress : bool
            是否显示进度条
        batch_size : int
            每批次处理的股票数量
        batch_sleep : float
            每批次之间的休息时间（秒）
        
        Returns
        -------
        pd.DataFrame
            合并后的融资融券数据
        """
        all_data = []
        total = len(stock_list)
        success_count = 0
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(stock_list),
                    total=total,
                    desc="📊 获取融资融券",
                    unit="只",
                    ncols=80
                )
            except ImportError:
                iterator = enumerate(stock_list)
                logger.info(f"开始获取融资融券数据: {total} 只股票...")
        else:
            iterator = enumerate(stock_list)
        
        for i, stock in iterator:
            df = self.fetch_margin_detail(stock, start_date, end_date)
            if df is not None and not df.empty:
                all_data.append(df)
                success_count += 1
            
            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({"成功": success_count, "当前": stock})
            
            # 批次休息
            if (i + 1) % batch_size == 0 and (i + 1) < total:
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description(f"📊 休息{batch_sleep}s")
                time.sleep(batch_sleep)
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description("📊 获取融资融券")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"批量获取融资融券完成: {success_count}/{total} 只, {len(result)} 条记录")
        return result
    
    def calculate_leverage_risk(
        self,
        stock_list: List[str],
        trade_date: str,
        lookback_days: int = 20
    ) -> pd.DataFrame:
        """
        计算杠杆过热因子 (Leverage Overheat Factor)
        
        典型的反向指标：当散户疯狂融资买入时，往往是阶段性顶部。
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表（6位代码）
        trade_date : str
            交易日期，格式 YYYYMMDD 或 YYYY-MM-DD
        lookback_days : int
            计算历史均值和标准差的回溯天数，默认 20 天
        
        Returns
        -------
        pd.DataFrame
            杠杆过热因子数据，包含：
            - stock_code: 股票代码
            - rzye: 融资余额（元）
            - rzmre: 融资买入额（元）
            - margin_buy_ratio: 融资买入占比（融资买入额/成交额）
            - margin_balance_ratio: 融资余额市值比（融资余额/总市值）
            - leverage_heat: 杠杆过热因子（标准化后，值越大风险越高）
            - leverage_risk_score: 风险得分 (0-1，越高越危险)
        
        Notes
        -----
        计算逻辑：
        1. 融资买入占比 = 融资买入额 / 当日成交额（交易拥挤度）
        2. 融资余额市值比 = 融资余额 / 总市值（存量杠杆压力）
        3. 过热因子 = (当日融资买入占比 - 20日均值) / 20日标准差
        
        风险警示：
        - leverage_heat > 2: 极度过热，高风险
        - leverage_heat > 1: 偏热，需警惕
        - leverage_heat < -1: 偏冷，可能是买入机会
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> stocks = ["000001", "000002", "600000"]
        >>> risk = loader.calculate_leverage_risk(stocks, "20240115")
        >>> # 筛选过热股票（可能见顶）
        >>> overheated = risk[risk['leverage_heat'] > 2]
        """
        trade_date = trade_date.replace("-", "")
        start_date = (
            datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days + 10)
        ).strftime("%Y%m%d")
        
        logger.info(f"🔥 计算杠杆过热因子: {len(stock_list)} 只股票, {trade_date}")
        
        # 1. 获取融资融券数据
        margin_df = self.fetch_margin_batch(
            stock_list=stock_list,
            start_date=start_date,
            end_date=trade_date,
            show_progress=True
        )
        
        # 2. 获取成交额和市值数据
        daily_basic = self.fetch_daily_basic(trade_date, stock_list)
        
        # 3. 获取日线成交额
        daily_data = self.fetch_daily_data_batch(
            stock_list=stock_list,
            start_date=start_date,
            end_date=trade_date,
            show_progress=False
        )
        
        if margin_df.empty:
            logger.warning("无融资融券数据，返回空结果")
            return pd.DataFrame({"stock_code": stock_list, "leverage_heat": 0, "leverage_risk_score": 0.5})
        
        # 4. 计算融资买入占比（需要合并成交额）
        if not daily_data.empty and "amount" in daily_data.columns:
            # 按股票和日期合并
            if "trade_date" in margin_df.columns:
                margin_df["trade_date"] = pd.to_datetime(margin_df["trade_date"])
            if "date" in daily_data.columns:
                daily_data = daily_data.rename(columns={"date": "trade_date"})
            
            margin_df = margin_df.merge(
                daily_data[["stock_code", "trade_date", "amount"]],
                on=["stock_code", "trade_date"],
                how="left"
            )
        
        # 5. 计算融资买入占比
        if "rzmre" in margin_df.columns and "amount" in margin_df.columns:
            margin_df["margin_buy_ratio"] = (
                margin_df["rzmre"] / margin_df["amount"].replace(0, np.nan)
            )
        else:
            margin_df["margin_buy_ratio"] = np.nan
        
        # 6. 计算历史均值和标准差
        result_list = []
        for stock in stock_list:
            stock_margin = margin_df[margin_df["stock_code"] == stock].copy()
            
            if stock_margin.empty:
                result_list.append({
                    "stock_code": stock,
                    "rzye": np.nan,
                    "rzmre": np.nan,
                    "margin_buy_ratio": np.nan,
                    "margin_balance_ratio": np.nan,
                    "leverage_heat": 0,
                    "leverage_risk_score": 0.5
                })
                continue
            
            # 按日期排序
            stock_margin = stock_margin.sort_values("trade_date")
            
            # 计算滚动统计
            stock_margin["ratio_mean"] = stock_margin["margin_buy_ratio"].rolling(
                window=lookback_days, min_periods=5
            ).mean()
            stock_margin["ratio_std"] = stock_margin["margin_buy_ratio"].rolling(
                window=lookback_days, min_periods=5
            ).std()
            
            # 取最新一天的数据
            latest = stock_margin.iloc[-1]
            
            # 计算过热因子（Z-score）
            if pd.notna(latest.get("ratio_std")) and latest["ratio_std"] > 0:
                leverage_heat = (
                    (latest["margin_buy_ratio"] - latest["ratio_mean"]) / latest["ratio_std"]
                )
            else:
                leverage_heat = 0
            
            # 计算融资余额市值比
            margin_balance_ratio = np.nan
            if daily_basic is not None and not daily_basic.empty:
                stock_basic = daily_basic[daily_basic["stock_code"] == stock]
                if not stock_basic.empty and "rzye" in latest:
                    total_mv = stock_basic["total_mv"].iloc[0]
                    if pd.notna(total_mv) and total_mv > 0:
                        margin_balance_ratio = latest.get("rzye", 0) / total_mv
            
            result_list.append({
                "stock_code": stock,
                "rzye": latest.get("rzye", np.nan),
                "rzmre": latest.get("rzmre", np.nan),
                "margin_buy_ratio": latest.get("margin_buy_ratio", np.nan),
                "margin_balance_ratio": margin_balance_ratio,
                "leverage_heat": leverage_heat,
            })
        
        result = pd.DataFrame(result_list)
        
        # 7. 计算风险得分（使用排名分位数）
        valid_mask = result["leverage_heat"].notna() & (result["leverage_heat"] != 0)
        if valid_mask.sum() > 0:
            result.loc[valid_mask, "leverage_risk_score"] = (
                result.loc[valid_mask, "leverage_heat"].rank(pct=True)
            )
        else:
            result["leverage_risk_score"] = 0.5
        
        result["leverage_risk_score"] = result["leverage_risk_score"].fillna(0.5)
        
        # 统计信息
        overheated_count = (result["leverage_heat"] > 2).sum()
        cold_count = (result["leverage_heat"] < -1).sum()
        
        logger.info(
            f"✅ 杠杆过热因子计算完成: "
            f"{len(result)} 只, 过热={overheated_count}, 偏冷={cold_count}"
        )
        
        return result
    
    def calculate_market_leverage_sentiment(
        self,
        trade_date: str,
        index_code: str = "hs300"
    ) -> Dict[str, Any]:
        """
        计算市场整体杠杆情绪
        
        获取指定指数成分股的杠杆情绪汇总，用于判断市场整体风险水平。
        
        Parameters
        ----------
        trade_date : str
            交易日期
        index_code : str
            指数代码，如 "hs300", "zz500"
        
        Returns
        -------
        Dict[str, Any]
            市场杠杆情绪指标：
            - avg_leverage_heat: 平均过热因子
            - overheated_ratio: 过热股票占比
            - cold_ratio: 偏冷股票占比
            - market_risk_level: 市场风险等级（low/medium/high/extreme）
            - signal: 信号建议（buy/hold/sell）
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> sentiment = loader.calculate_market_leverage_sentiment("20240115")
        >>> print(f"市场风险等级: {sentiment['market_risk_level']}")
        >>> print(f"建议操作: {sentiment['signal']}")
        """
        trade_date = trade_date.replace("-", "")
        
        logger.info(f"📈 计算市场整体杠杆情绪: {index_code}, {trade_date}")
        
        # 获取成分股
        stock_list = self.fetch_index_constituents(index_code)
        
        if not stock_list:
            return {
                "avg_leverage_heat": 0,
                "overheated_ratio": 0,
                "cold_ratio": 0,
                "market_risk_level": "unknown",
                "signal": "hold"
            }
        
        # 采样（避免请求过多）
        sample_size = min(100, len(stock_list))
        sampled_stocks = stock_list[:sample_size]
        
        # 计算杠杆因子
        leverage_df = self.calculate_leverage_risk(
            stock_list=sampled_stocks,
            trade_date=trade_date
        )
        
        if leverage_df.empty:
            return {
                "avg_leverage_heat": 0,
                "overheated_ratio": 0,
                "cold_ratio": 0,
                "market_risk_level": "unknown",
                "signal": "hold"
            }
        
        # 统计指标
        valid_heat = leverage_df["leverage_heat"].dropna()
        avg_heat = valid_heat.mean() if len(valid_heat) > 0 else 0
        overheated_ratio = (valid_heat > 2).sum() / len(valid_heat) if len(valid_heat) > 0 else 0
        cold_ratio = (valid_heat < -1).sum() / len(valid_heat) if len(valid_heat) > 0 else 0
        
        # 判断风险等级
        if avg_heat > 2 or overheated_ratio > 0.3:
            risk_level = "extreme"
            signal = "sell"
        elif avg_heat > 1 or overheated_ratio > 0.2:
            risk_level = "high"
            signal = "reduce"
        elif avg_heat > 0.5:
            risk_level = "medium"
            signal = "hold"
        elif avg_heat < -1 and cold_ratio > 0.3:
            risk_level = "low"
            signal = "buy"
        else:
            risk_level = "normal"
            signal = "hold"
        
        result = {
            "trade_date": trade_date,
            "index_code": index_code,
            "sample_size": len(sampled_stocks),
            "avg_leverage_heat": round(avg_heat, 3),
            "overheated_ratio": round(overheated_ratio, 3),
            "cold_ratio": round(cold_ratio, 3),
            "market_risk_level": risk_level,
            "signal": signal
        }
        
        logger.info(
            f"✅ 市场杠杆情绪: 风险={risk_level}, "
            f"过热={avg_heat:.2f}, 过热比={overheated_ratio:.1%}"
        )
        
        return result
    
    def get_leverage_warning_stocks(
        self,
        stock_list: List[str],
        trade_date: str,
        heat_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        获取杠杆过热预警股票
        
        筛选出融资过热的股票，提示潜在的回调风险。
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        trade_date : str
            交易日期
        heat_threshold : float
            过热阈值，默认 2.0（2倍标准差）
        
        Returns
        -------
        pd.DataFrame
            过热预警股票列表
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> stocks = loader.fetch_index_constituents("hs300")
        >>> warnings = loader.get_leverage_warning_stocks(stocks, "20240115")
        >>> print(f"发现 {len(warnings)} 只过热股票")
        """
        leverage_df = self.calculate_leverage_risk(stock_list, trade_date)
        
        if leverage_df.empty:
            return pd.DataFrame()
        
        # 筛选过热股票
        warnings = leverage_df[leverage_df["leverage_heat"] >= heat_threshold].copy()
        warnings = warnings.sort_values("leverage_heat", ascending=False)
        
        if len(warnings) > 0:
            logger.warning(
                f"⚠️ 发现 {len(warnings)} 只杠杆过热股票 "
                f"(heat >= {heat_threshold})"
            )
        
        return warnings


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

