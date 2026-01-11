"""
Tushare 数据加载器基础模块

该模块提供 TushareDataLoaderBase 基类，包含：
- API 初始化和限流机制
- 日线行情数据获取
- 财务指标数据获取
- 指数成分股和全市场股票
- 交易日历和行业分类

Notes
-----
使用前需要配置 Tushare API Token：
1. 在 config/strategy_config.yaml 中设置 tushare.api_token
2. 或通过环境变量 TUSHARE_TOKEN 设置
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
import os

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TushareDataLoaderBase:
    """
    Tushare Pro 数据加载器基类
    
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
    >>> loader = TushareDataLoaderBase(api_token="your_token")
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
        self._news_source = "auto"
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
                
                # 读取 news_source 配置
                self._news_source = tushare_config.get("news_source", "auto")
                if self._news_source == "akshare":
                    logger.info("📰 新闻数据源: AKShare (免费无限制)")
                elif self._news_source == "tushare":
                    logger.info("📰 新闻数据源: Tushare")
                else:
                    logger.debug("📰 新闻数据源: 自动切换 (Tushare -> AKShare)")
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
        rate_limit_count = 0  # 频率限制重试计数
        MAX_RATE_LIMIT_RETRIES = 2  # 频率限制最大重试次数
        
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
                rate_limit_keywords = ["每分钟最多访问", "每天最多访问", "每小时最多访问", 
                                       "抱歉", "频率", "rate limit", "too many", "限制"]
                is_rate_limit = any(kw in error_msg or kw in error_msg_lower for kw in rate_limit_keywords)
                
                if is_rate_limit:
                    rate_limit_count += 1
                    if rate_limit_count > MAX_RATE_LIMIT_RETRIES:
                        logger.warning(
                            f"API 频率限制重试次数超限 ({rate_limit_count} 次)，跳过该请求。"
                            f"错误: {error_msg[:80]}"
                        )
                        return None
                    
                    # 检查是否为每天/每小时限制（无法通过等待解决）
                    if "每天最多访问" in error_msg or "每小时最多访问" in error_msg:
                        logger.warning(
                            f"触发 Tushare 每日/每小时配额限制，无法继续请求。"
                            f"错误: {error_msg[:80]}"
                        )
                        return None
                    
                    logger.warning(
                        f"触发 API 频率限制 (重试 {rate_limit_count}/{MAX_RATE_LIMIT_RETRIES})，"
                        f"等待 {self.RATE_LIMIT_DELAY} 秒... 错误: {error_msg[:80]}"
                    )
                    time.sleep(self.RATE_LIMIT_DELAY)
                    continue  # 频率限制重试不消耗 attempt 次数
                
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
        >>> loader = TushareDataLoaderBase()
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
        >>> loader = TushareDataLoaderBase()
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
            每批次处理的股票数量（默认 200）
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
            每批次处理的股票数量（默认 150）
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
        
        # 过滤掉空的 DataFrame，避免 FutureWarning
        valid_data = []
        for df in all_data:
            if df is None or df.empty:
                continue
            # 删除全为 NaN 的列
            df_cleaned = df.dropna(axis=1, how='all')
            if not df_cleaned.empty and not df_cleaned.isna().all().all():
                valid_data.append(df_cleaned)
        
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
        >>> loader = TushareDataLoaderBase()
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
        >>> loader = TushareDataLoaderBase()
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
        """
        logger.info(f"获取申万{level}级行业分类")
        
        # 尝试缓存
        today = datetime.now().strftime("%Y%m%d")
        cache_file = self.cache_dir / f"sw_industry_L{level}_{today[:6]}.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                mapping = dict(zip(df["stock_code"], df["industry"]))
                logger.info(f"从缓存加载申万分类: {len(mapping)} 只股票")
                return mapping
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
        
        # Tushare 的申万分类接口
        try:
            # 方式1: index_classify 获取申万指数成分
            # 获取申万行业指数列表
            index_list = self._fetch_with_retry(
                self.pro.index_classify,
                index_code="",
                level=f"L{level}",
                src="SW2021"
            )
            
            if index_list is not None and not index_list.empty:
                # 遍历每个行业，获取成分股
                all_mappings = []
                
                for _, row in index_list.iterrows():
                    industry_code = row.get("index_code", "")
                    industry_name = row.get("industry_name", row.get("name", ""))
                    
                    if not industry_code:
                        continue
                    
                    # 获取该行业成分股
                    members = self._fetch_with_retry(
                        self.pro.index_member,
                        index_code=industry_code
                    )
                    
                    if members is not None and not members.empty:
                        for _, m in members.iterrows():
                            stock_code = m.get("con_code", "")[:6]
                            all_mappings.append({
                                "stock_code": stock_code,
                                "industry": industry_name
                            })
                
                if all_mappings:
                    df = pd.DataFrame(all_mappings)
                    # 去重，保留第一个
                    df = df.drop_duplicates(subset=["stock_code"], keep="first")
                    
                    # 保存缓存
                    try:
                        df.to_parquet(cache_file, index=False)
                        logger.info(f"申万分类已缓存: {cache_file}")
                    except Exception:
                        pass
                    
                    result = dict(zip(df["stock_code"], df["industry"]))
                    logger.info(f"获取申万分类成功: {len(result)} 只股票")
                    return result
                    
        except Exception as e:
            logger.debug(f"申万分类接口不可用: {e}")
        
        # 降级到普通行业分类
        logger.info("使用普通行业分类替代申万分类")
        return self.fetch_industry_mapping()
    
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
    
    # ==================== 数据单位验证工具 ====================
    
    @staticmethod
    def validate_data_units(
        df: pd.DataFrame,
        check_columns: Optional[List[str]] = None,
        expected_units: Optional[Dict[str, str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        验证数据单位一致性
        
        检查关键金额字段的数量级是否合理，帮助发现单位不一致的问题。
        
        Parameters
        ----------
        df : pd.DataFrame
            待验证的数据
        check_columns : Optional[List[str]]
            要检查的列名列表，默认检查常见金额列
        expected_units : Optional[Dict[str, str]]
            列名到预期单位的映射
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            每列的验证结果
        """
        if check_columns is None:
            # 默认检查的常见金额列
            check_columns = [
                'amount', 'volume', 'total_mv', 'circ_mv',
                'buy_elg_amount', 'sell_elg_amount', 'buy_lg_amount', 'sell_lg_amount',
                'net_mf_amount', 'fd_amount', 'rzye', 'rzmre', 'rqye'
            ]
        
        # Tushare 标准单位参考
        default_units = {
            'amount': '千元(daily)/万元(moneyflow)',
            'volume': '手',
            'total_mv': '万元',
            'circ_mv': '万元',
            'buy_elg_amount': '万元',
            'sell_elg_amount': '万元',
            'buy_lg_amount': '万元',
            'sell_lg_amount': '万元',
            'net_mf_amount': '万元',
            'fd_amount': '万元',
            'rzye': '元',
            'rzmre': '元',
            'rqye': '元',
        }
        
        if expected_units:
            default_units.update(expected_units)
        
        results = {}
        
        for col in check_columns:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            if len(series) == 0:
                results[col] = {"warning": "列为空"}
                continue
            
            min_val = series.min()
            max_val = series.max()
            mean_val = series.mean()
            
            # 计算数量级
            if mean_val > 0:
                magnitude = int(np.log10(abs(mean_val)))
            else:
                magnitude = 0
            
            # 推测单位
            likely_unit = "未知"
            warning = None
            
            if col in ['amount']:
                if magnitude >= 9:
                    likely_unit = "元"
                elif magnitude >= 6:
                    likely_unit = "千元"
                elif magnitude >= 3:
                    likely_unit = "万元"
                else:
                    likely_unit = "可能有问题"
                    warning = f"成交额数量级异常: 10^{magnitude}"
                    
            elif col in ['total_mv', 'circ_mv']:
                if magnitude >= 10:
                    likely_unit = "元"
                    warning = "市值单位可能是元，预期为万元"
                elif magnitude >= 6:
                    likely_unit = "万元（正常）"
                else:
                    likely_unit = "可能有问题"
                    warning = f"市值数量级异常: 10^{magnitude}"
                    
            elif col in ['rzye', 'rzmre']:
                if magnitude >= 8:
                    likely_unit = "元（正常）"
                elif magnitude >= 5:
                    likely_unit = "万元"
                    warning = "融资数据单位可能是万元，预期为元"
                else:
                    likely_unit = "可能有问题"
                    
            elif 'amount' in col.lower():  # 资金流向金额
                if magnitude >= 6:
                    likely_unit = "元"
                    warning = "资金流向单位可能是元，预期为万元"
                elif magnitude >= 2:
                    likely_unit = "万元（正常）"
                else:
                    likely_unit = "可能有问题"
            
            results[col] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "magnitude": magnitude,
                "likely_unit": likely_unit,
                "expected_unit": default_units.get(col, "未知"),
                "warning": warning
            }
        
        # 输出警告
        for col, info in results.items():
            if info.get("warning"):
                logger.warning(f"⚠️ 数据单位警告 - {col}: {info['warning']}")
        
        return results
    
    def print_data_summary(
        self,
        df: pd.DataFrame,
        title: str = "数据摘要"
    ) -> None:
        """
        打印数据摘要（用于调试数据单位问题）
        
        Parameters
        ----------
        df : pd.DataFrame
            要检查的数据
        title : str
            标题
        """
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print(f"{'='*60}")
        print(f"行数: {len(df)}, 列数: {len(df.columns)}")
        print(f"\n前5行样本:")
        print(df.head())
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print(f"\n数值列统计:")
            for col in numeric_cols[:10]:  # 最多显示10列
                series = df[col].dropna()
                if len(series) > 0:
                    print(f"  {col:25s}: min={series.min():>15,.2f}, "
                          f"max={series.max():>15,.2f}, "
                          f"mean={series.mean():>15,.2f}")
        
        # 单位验证
        validation = self.validate_data_units(df)
        if any(v.get("warning") for v in validation.values()):
            print(f"\n⚠️ 单位警告:")
            for col, info in validation.items():
                if info.get("warning"):
                    print(f"  {col}: {info['warning']}")
        
        print(f"{'='*60}\n")


# ==================== 便捷函数 ====================

def create_tushare_loader(config: Optional[Dict] = None) -> "TushareDataLoaderBase":
    """
    创建 Tushare 数据加载器
    
    Parameters
    ----------
    config : Optional[Dict]
        配置字典，包含 tushare.api_token
    
    Returns
    -------
    TushareDataLoaderBase
        数据加载器实例
    """
    api_token = None
    
    if config:
        api_token = config.get("tushare", {}).get("api_token")
    
    if not api_token:
        api_token = os.environ.get("TUSHARE_TOKEN")
    
    return TushareDataLoaderBase(api_token=api_token)

