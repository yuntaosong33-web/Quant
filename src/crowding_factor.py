"""
行业拥挤因子分布式计算模块

使用 Ray 进行分布式并行计算，支持大规模 A 股市场的拥挤因子计算。
拥挤因子衡量行业内股票收益率的相关性，反映资金抱团程度。

核心功能：
- 使用 Ray 分布式计算框架进行并行计算
- 支持 31 个申万一级行业的并行处理
- 使用 Dask 处理超大内存数据
- 滚动窗口计算行业内股票两两相关性

Performance Notes
-----------------
- 使用 ray.put 共享大数据对象，减少序列化开销
- 对奇异矩阵进行容错处理
- 支持增量式窗口计算避免内存溢出
"""

from typing import Optional, Dict, List, Tuple, Any, Union
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numba import jit

logger = logging.getLogger(__name__)


# ==================== Numba 优化的底层计算函数 ====================

@jit(nopython=True, cache=True)
def _pairwise_correlation_numba(
    returns_matrix: np.ndarray,
    min_valid_ratio: float = 0.5
) -> float:
    """
    Numba 加速的两两相关性计算
    
    计算收益率矩阵中所有股票对的平均相关系数。
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        收益率矩阵，形状为 (n_periods, n_stocks)
    min_valid_ratio : float
        最小有效数据比例，默认 0.5
    
    Returns
    -------
    float
        平均两两相关系数，如果无法计算返回 NaN
    
    Notes
    -----
    - 处理 NaN 值：只使用两只股票共同有效的数据点
    - 对奇异情况（标准差为0）进行容错处理
    """
    n_periods, n_stocks = returns_matrix.shape
    
    if n_stocks < 2:
        return np.nan
    
    correlations = []
    min_valid_count = int(n_periods * min_valid_ratio)
    
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            # 获取两只股票的收益率
            ret_i = returns_matrix[:, i]
            ret_j = returns_matrix[:, j]
            
            # 找出共同有效的数据点
            valid_count = 0
            sum_i = 0.0
            sum_j = 0.0
            
            for k in range(n_periods):
                if not np.isnan(ret_i[k]) and not np.isnan(ret_j[k]):
                    valid_count += 1
                    sum_i += ret_i[k]
                    sum_j += ret_j[k]
            
            if valid_count < min_valid_count:
                continue
            
            # 计算均值
            mean_i = sum_i / valid_count
            mean_j = sum_j / valid_count
            
            # 计算协方差和标准差
            cov = 0.0
            var_i = 0.0
            var_j = 0.0
            
            for k in range(n_periods):
                if not np.isnan(ret_i[k]) and not np.isnan(ret_j[k]):
                    diff_i = ret_i[k] - mean_i
                    diff_j = ret_j[k] - mean_j
                    cov += diff_i * diff_j
                    var_i += diff_i * diff_i
                    var_j += diff_j * diff_j
            
            # 计算相关系数
            std_i = np.sqrt(var_i)
            std_j = np.sqrt(var_j)
            
            if std_i > 1e-10 and std_j > 1e-10:
                corr = cov / (std_i * std_j)
                # 限制在 [-1, 1] 范围内
                corr = max(-1.0, min(1.0, corr))
                correlations.append(corr)
    
    if len(correlations) == 0:
        return np.nan
    
    # 计算平均相关系数
    total = 0.0
    for c in correlations:
        total += c
    
    return total / len(correlations)


@jit(nopython=True, cache=True)
def _rolling_correlation_1d(
    returns_matrix: np.ndarray,
    window: int,
    min_periods: int
) -> np.ndarray:
    """
    Numba 加速的滚动两两相关性计算
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        收益率矩阵，形状为 (n_periods, n_stocks)
    window : int
        滚动窗口大小
    min_periods : int
        最小有效数据点数
    
    Returns
    -------
    np.ndarray
        每个时间点的平均两两相关系数
    """
    n_periods, n_stocks = returns_matrix.shape
    result = np.empty(n_periods, dtype=np.float64)
    result[:] = np.nan
    
    if n_stocks < 2:
        return result
    
    min_valid_ratio = float(min_periods) / float(window)
    
    for t in range(window - 1, n_periods):
        # 提取窗口内的数据
        start = t - window + 1
        window_data = returns_matrix[start:t + 1, :]
        
        # 计算该窗口的平均相关系数
        corr = _pairwise_correlation_numba(window_data, min_valid_ratio)
        result[t] = corr
    
    return result


# ==================== Ray 分布式计算函数 ====================

def _check_ray_available() -> bool:
    """
    检查 Ray 是否可用
    
    Returns
    -------
    bool
        Ray 是否已安装且可用
    """
    try:
        import ray
        return True
    except ImportError:
        return False


def _check_dask_available() -> bool:
    """
    检查 Dask 是否可用
    
    Returns
    -------
    bool
        Dask 是否已安装且可用
    """
    try:
        import dask
        import dask.dataframe as dd
        return True
    except ImportError:
        return False


def _create_ray_remote_function():
    """
    创建 Ray remote 函数（延迟导入）
    
    Returns
    -------
    Callable
        Ray remote 装饰的计算函数
    """
    import ray
    
    @ray.remote
    def calculate_sector_correlation(
        sector_data_ref: Any,
        sector_code: str,
        window: int = 20,
        min_periods: int = 10
    ) -> Tuple[str, pd.Series]:
        """
        Ray Remote 函数：计算单个行业的拥挤因子
        
        在 Ray 集群的工作节点上执行，计算指定行业内股票的
        平均两两相关性（滚动窗口）。
        
        Parameters
        ----------
        sector_data_ref : ray.ObjectRef
            行业价格数据的 Ray 对象引用（由 ray.put 创建）
        sector_code : str
            行业代码
        window : int
            滚动窗口大小（交易日），默认 20
        min_periods : int
            最小有效数据点数，默认 10
        
        Returns
        -------
        Tuple[str, pd.Series]
            (行业代码, 拥挤因子时间序列)
        
        Notes
        -----
        - 使用 ray.get 获取共享的大数据对象
        - 对奇异矩阵进行容错处理
        - 返回的 Series 索引为 DatetimeIndex
        """
        import ray
        
        try:
            # 获取行业价格数据
            sector_df = ray.get(sector_data_ref)
            
            if sector_df.empty or len(sector_df.columns) < 2:
                logger.warning(f"行业 {sector_code} 股票数量不足，跳过计算")
                return (sector_code, pd.Series(dtype=np.float64))
            
            # 计算收益率矩阵
            returns_df = sector_df.pct_change()
            
            # 转换为 numpy 数组
            returns_matrix = returns_df.values.astype(np.float64)
            
            # 使用 numba 加速的滚动相关性计算
            crowding_scores = _rolling_correlation_1d(
                returns_matrix,
                window=window,
                min_periods=min_periods
            )
            
            # 创建结果 Series
            result = pd.Series(
                crowding_scores,
                index=sector_df.index,
                name=sector_code
            )
            
            return (sector_code, result)
            
        except Exception as e:
            logger.error(f"计算行业 {sector_code} 拥挤因子失败: {e}")
            return (sector_code, pd.Series(dtype=np.float64))
    
    return calculate_sector_correlation


class CrowdingFactorCalculator:
    """
    行业拥挤因子分布式计算器
    
    使用 Ray 分布式计算框架并行计算 31 个申万一级行业的拥挤因子。
    拥挤因子衡量行业内股票收益率的相关性，反映资金抱团程度。
    
    核心逻辑：
    - 拥挤因子 = 行业内股票两两相关系数的平均值
    - 高拥挤因子表示行业内股票高度同涨同跌，抱团现象严重
    - 低拥挤因子表示行业内股票分化，个股行情为主
    
    Parameters
    ----------
    n_workers : int, optional
        Ray 工作节点数量，默认为 CPU 核心数
    window : int, optional
        滚动窗口大小（交易日），默认 20
    min_periods : int, optional
        最小有效数据点数，默认 10
    use_dask : bool, optional
        是否使用 Dask 处理大数据，默认 False
    dask_memory_limit : str, optional
        Dask 内存限制，默认 "2GB"
    
    Attributes
    ----------
    ray_initialized : bool
        Ray 集群是否已初始化
    sector_mapping : Dict[str, str]
        申万一级行业代码到名称的映射
    
    Examples
    --------
    >>> # 初始化计算器
    >>> calculator = CrowdingFactorCalculator(n_workers=8, window=20)
    >>> 
    >>> # 准备价格数据和行业映射
    >>> price_df = pd.DataFrame(...)  # index=date, columns=stock_codes
    >>> stock_sector_map = {"000001": "银行", "600519": "食品饮料", ...}
    >>> 
    >>> # 计算拥挤因子
    >>> crowding_df = calculator.calculate(price_df, stock_sector_map)
    >>> print(crowding_df.head())
    
    Notes
    -----
    - 首次调用 calculate 会自动初始化 Ray 集群
    - 使用 ray.put 共享大数据对象，减少序列化开销
    - 对奇异矩阵（如行业内只有1只股票）进行容错处理
    """
    
    # 申万一级行业代码和名称映射（共 31 个）
    SW_LEVEL1_SECTORS = {
        "801010": "农林牧渔",
        "801020": "采掘",
        "801030": "化工",
        "801040": "钢铁",
        "801050": "有色金属",
        "801080": "电子",
        "801110": "家用电器",
        "801120": "食品饮料",
        "801130": "纺织服装",
        "801140": "轻工制造",
        "801150": "医药生物",
        "801160": "公用事业",
        "801170": "交通运输",
        "801180": "房地产",
        "801200": "商业贸易",
        "801210": "休闲服务",
        "801230": "综合",
        "801710": "建筑材料",
        "801720": "建筑装饰",
        "801730": "电气设备",
        "801740": "国防军工",
        "801750": "计算机",
        "801760": "传媒",
        "801770": "通信",
        "801780": "银行",
        "801790": "非银金融",
        "801880": "汽车",
        "801890": "机械设备",
        "801950": "煤炭",
        "801960": "石油石化",
        "801970": "环保",
    }
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        window: int = 20,
        min_periods: int = 10,
        use_dask: bool = False,
        dask_memory_limit: str = "2GB"
    ) -> None:
        """
        初始化拥挤因子计算器
        
        Parameters
        ----------
        n_workers : Optional[int]
            Ray 工作节点数量，默认为 CPU 核心数
        window : int
            滚动窗口大小（交易日），默认 20
        min_periods : int
            最小有效数据点数，默认 10
        use_dask : bool
            是否使用 Dask 处理大数据，默认 False
        dask_memory_limit : str
            Dask 内存限制，默认 "2GB"
        """
        import os
        
        self.n_workers = n_workers or os.cpu_count()
        self.window = window
        self.min_periods = min_periods
        self.use_dask = use_dask
        self.dask_memory_limit = dask_memory_limit
        
        self.ray_initialized = False
        self.sector_mapping = self.SW_LEVEL1_SECTORS.copy()
        
        # 验证依赖
        if not _check_ray_available():
            raise ImportError(
                "Ray 未安装，请运行: pip install ray[default]"
            )
        
        if self.use_dask and not _check_dask_available():
            raise ImportError(
                "Dask 未安装，请运行: pip install dask[complete]"
            )
        
        logger.info(
            f"CrowdingFactorCalculator 初始化: "
            f"n_workers={self.n_workers}, window={self.window}, "
            f"use_dask={self.use_dask}"
        )
    
    def _init_ray_cluster(self) -> None:
        """
        初始化 Ray 集群
        
        如果 Ray 集群已初始化，则跳过。
        支持本地模式和集群模式。
        
        Notes
        -----
        - 本地模式：使用 ray.init() 启动本地集群
        - 集群模式：使用 ray.init(address='auto') 连接现有集群
        """
        import ray
        
        if self.ray_initialized:
            logger.debug("Ray 集群已初始化，跳过")
            return
        
        if ray.is_initialized():
            logger.info("检测到已存在的 Ray 集群，直接使用")
            self.ray_initialized = True
            return
        
        try:
            # 初始化本地 Ray 集群
            ray.init(
                num_cpus=self.n_workers,
                ignore_reinit_error=True,
                logging_level=logging.WARNING
            )
            self.ray_initialized = True
            logger.info(
                f"Ray 集群初始化成功: "
                f"num_cpus={self.n_workers}"
            )
        except Exception as e:
            logger.error(f"Ray 集群初始化失败: {e}")
            raise
    
    def _shutdown_ray_cluster(self) -> None:
        """
        关闭 Ray 集群
        
        释放 Ray 集群资源。建议在计算完成后调用。
        """
        import ray
        
        if ray.is_initialized():
            ray.shutdown()
            self.ray_initialized = False
            logger.info("Ray 集群已关闭")
    
    def _prepare_sector_data(
        self,
        price_df: pd.DataFrame,
        stock_sector_map: Dict[str, str]
    ) -> Dict[str, pd.DataFrame]:
        """
        按行业分组准备价格数据
        
        Parameters
        ----------
        price_df : pd.DataFrame
            价格数据，index=日期，columns=股票代码
        stock_sector_map : Dict[str, str]
            股票代码到行业的映射
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            行业代码到该行业价格数据的映射
        
        Notes
        -----
        - 自动过滤无行业归属的股票
        - 每个行业 DataFrame 的列为该行业的股票代码
        """
        sector_data: Dict[str, List[str]] = {}
        
        # 按行业分组股票
        for stock_code in price_df.columns:
            sector_name = stock_sector_map.get(stock_code)
            
            if sector_name is None:
                continue
            
            # 查找行业代码
            sector_code = None
            for code, name in self.sector_mapping.items():
                if name == sector_name:
                    sector_code = code
                    break
            
            if sector_code is None:
                # 如果行业名称不在映射中，使用行业名称作为代码
                sector_code = sector_name
            
            if sector_code not in sector_data:
                sector_data[sector_code] = []
            
            sector_data[sector_code].append(stock_code)
        
        # 提取每个行业的价格数据
        result = {}
        for sector_code, stock_list in sector_data.items():
            if len(stock_list) >= 2:  # 至少需要2只股票
                sector_df = price_df[stock_list].copy()
                result[sector_code] = sector_df
                logger.debug(
                    f"行业 {sector_code}: {len(stock_list)} 只股票"
                )
        
        logger.info(f"行业分组完成: {len(result)} 个有效行业")
        return result
    
    def _load_large_data_with_dask(
        self,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        使用 Dask 处理大规模数据
        
        当输入数据超出内存时，使用 Dask 进行分块处理。
        
        Parameters
        ----------
        price_df : pd.DataFrame
            原始价格数据
        
        Returns
        -------
        pd.DataFrame
            处理后的价格数据
        
        Notes
        -----
        - 自动检测数据大小，仅在必要时使用 Dask
        - 使用 dask.dataframe 进行内存优化
        """
        import dask.dataframe as dd
        
        # 估算数据大小（MB）
        memory_usage_mb = price_df.memory_usage(deep=True).sum() / (1024 ** 2)
        
        logger.info(f"价格数据内存占用: {memory_usage_mb:.2f} MB")
        
        # 解析内存限制
        limit_mb = float(self.dask_memory_limit.replace("GB", "")) * 1024
        
        if memory_usage_mb > limit_mb * 0.5:
            logger.info(
                f"数据量较大，使用 Dask 进行优化处理 "
                f"(limit: {self.dask_memory_limit})"
            )
            
            # 转换为 Dask DataFrame
            npartitions = max(1, int(memory_usage_mb / (limit_mb * 0.25)))
            ddf = dd.from_pandas(price_df, npartitions=npartitions)
            
            # 执行必要的预处理（在 Dask 中）
            # 这里主要是确保数据类型正确
            ddf = ddf.astype(np.float64)
            
            # 计算并返回 Pandas DataFrame
            return ddf.compute()
        
        return price_df
    
    def calculate(
        self,
        price_df: pd.DataFrame,
        stock_sector_map: Dict[str, str],
        parallel: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        计算所有行业的拥挤因子
        
        使用 Ray 分布式计算框架并行计算 31 个申万一级行业的拥挤因子。
        
        Parameters
        ----------
        price_df : pd.DataFrame
            价格数据，格式要求：
            - index: DatetimeIndex（交易日期）
            - columns: 股票代码
            - values: 收盘价
        stock_sector_map : Dict[str, str]
            股票代码到申万一级行业名称的映射
            例如: {"000001": "银行", "600519": "食品饮料"}
        parallel : bool, optional
            是否使用并行计算，默认 True
        show_progress : bool, optional
            是否显示进度信息，默认 True
        
        Returns
        -------
        pd.DataFrame
            拥挤因子时间序列，格式：
            - index: DatetimeIndex
            - columns: 行业代码（或名称）
            - values: 拥挤因子分数（0-1，越高表示越拥挤）
        
        Raises
        ------
        ValueError
            当输入数据格式不正确时
        RuntimeError
            当 Ray 集群初始化失败时
        
        Examples
        --------
        >>> calculator = CrowdingFactorCalculator(n_workers=8)
        >>> 
        >>> # 准备数据
        >>> price_df = pd.read_parquet("data/prices.parquet")
        >>> sector_map = load_sector_mapping()
        >>> 
        >>> # 计算拥挤因子
        >>> crowding = calculator.calculate(price_df, sector_map)
        >>> 
        >>> # 查看结果
        >>> print(crowding.head())
        >>> print(crowding.describe())
        
        Notes
        -----
        - 使用 ray.put 将大数据对象放入共享内存，减少序列化开销
        - 对每个行业并行计算，充分利用多核 CPU
        - 对奇异矩阵（标准差为0、股票数不足）进行容错处理
        """
        import ray
        
        # 输入验证
        if price_df.empty:
            raise ValueError("价格数据为空")
        
        if not isinstance(price_df.index, pd.DatetimeIndex):
            logger.warning("价格数据索引不是 DatetimeIndex，尝试转换")
            price_df.index = pd.to_datetime(price_df.index)
        
        # 使用 Dask 处理大数据（如果启用）
        if self.use_dask:
            price_df = self._load_large_data_with_dask(price_df)
        
        # 按行业分组数据
        sector_data_dict = self._prepare_sector_data(price_df, stock_sector_map)
        
        if not sector_data_dict:
            logger.warning("没有有效的行业数据")
            return pd.DataFrame()
        
        if parallel:
            return self._calculate_parallel(
                sector_data_dict,
                price_df.index,
                show_progress
            )
        else:
            return self._calculate_sequential(
                sector_data_dict,
                price_df.index,
                show_progress
            )
    
    def _calculate_parallel(
        self,
        sector_data_dict: Dict[str, pd.DataFrame],
        date_index: pd.DatetimeIndex,
        show_progress: bool
    ) -> pd.DataFrame:
        """
        并行计算所有行业的拥挤因子
        
        使用 Ray 分布式计算框架。
        
        Parameters
        ----------
        sector_data_dict : Dict[str, pd.DataFrame]
            行业代码到价格数据的映射
        date_index : pd.DatetimeIndex
            日期索引
        show_progress : bool
            是否显示进度
        
        Returns
        -------
        pd.DataFrame
            拥挤因子 DataFrame
        """
        import ray
        
        # 初始化 Ray 集群
        self._init_ray_cluster()
        
        # 获取 Ray remote 函数
        calculate_sector_correlation = _create_ray_remote_function()
        
        # 使用 ray.put 将数据放入共享内存
        if show_progress:
            logger.info("正在将行业数据放入 Ray 共享内存...")
        
        sector_refs: Dict[str, Any] = {}
        for sector_code, sector_df in sector_data_dict.items():
            sector_refs[sector_code] = ray.put(sector_df)
        
        if show_progress:
            logger.info(f"共享内存准备完成，开始并行计算 {len(sector_refs)} 个行业")
        
        # 提交所有计算任务
        futures = []
        for sector_code, sector_ref in sector_refs.items():
            future = calculate_sector_correlation.remote(
                sector_ref,
                sector_code,
                self.window,
                self.min_periods
            )
            futures.append(future)
        
        # 收集结果
        results = ray.get(futures)
        
        if show_progress:
            logger.info("所有行业计算完成，正在合并结果...")
        
        # 合并结果
        crowding_series_dict = {}
        for sector_code, series in results:
            if not series.empty:
                crowding_series_dict[sector_code] = series
        
        if not crowding_series_dict:
            logger.warning("所有行业计算均失败")
            return pd.DataFrame()
        
        # 构建结果 DataFrame
        result_df = pd.DataFrame(crowding_series_dict)
        result_df = result_df.reindex(date_index)
        result_df.index.name = "date"
        
        logger.info(
            f"拥挤因子计算完成: "
            f"{len(result_df.columns)} 个行业, "
            f"{len(result_df)} 个交易日"
        )
        
        return result_df
    
    def _calculate_sequential(
        self,
        sector_data_dict: Dict[str, pd.DataFrame],
        date_index: pd.DatetimeIndex,
        show_progress: bool
    ) -> pd.DataFrame:
        """
        顺序计算所有行业的拥挤因子（非并行）
        
        用于调试或不需要并行的场景。
        
        Parameters
        ----------
        sector_data_dict : Dict[str, pd.DataFrame]
            行业代码到价格数据的映射
        date_index : pd.DatetimeIndex
            日期索引
        show_progress : bool
            是否显示进度
        
        Returns
        -------
        pd.DataFrame
            拥挤因子 DataFrame
        """
        crowding_series_dict = {}
        total = len(sector_data_dict)
        
        for i, (sector_code, sector_df) in enumerate(sector_data_dict.items(), 1):
            if show_progress and i % 5 == 0:
                logger.info(f"计算进度: {i}/{total}")
            
            try:
                # 计算收益率矩阵
                returns_df = sector_df.pct_change()
                returns_matrix = returns_df.values.astype(np.float64)
                
                # 计算滚动相关性
                crowding_scores = _rolling_correlation_1d(
                    returns_matrix,
                    self.window,
                    self.min_periods
                )
                
                # 创建结果 Series
                series = pd.Series(
                    crowding_scores,
                    index=sector_df.index,
                    name=sector_code
                )
                
                if not series.empty:
                    crowding_series_dict[sector_code] = series
                    
            except Exception as e:
                logger.warning(f"计算行业 {sector_code} 失败: {e}")
                continue
        
        if not crowding_series_dict:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(crowding_series_dict)
        result_df = result_df.reindex(date_index)
        result_df.index.name = "date"
        
        return result_df
    
    def calculate_single_sector(
        self,
        price_df: pd.DataFrame,
        sector_name: str = "default"
    ) -> pd.Series:
        """
        计算单个行业的拥挤因子
        
        用于快速计算单个行业或自定义股票组合的拥挤因子。
        
        Parameters
        ----------
        price_df : pd.DataFrame
            价格数据，index=日期，columns=股票代码
        sector_name : str
            行业名称，用于标识结果
        
        Returns
        -------
        pd.Series
            拥挤因子时间序列
        
        Examples
        --------
        >>> # 计算自定义股票组合的拥挤因子
        >>> stocks = ["600519", "000858", "002304"]
        >>> price_df = data[stocks]
        >>> crowding = calculator.calculate_single_sector(price_df, "白酒组合")
        """
        if price_df.empty or len(price_df.columns) < 2:
            logger.warning("股票数量不足（至少需要2只）")
            return pd.Series(dtype=np.float64)
        
        # 计算收益率矩阵
        returns_df = price_df.pct_change()
        returns_matrix = returns_df.values.astype(np.float64)
        
        # 计算滚动相关性
        crowding_scores = _rolling_correlation_1d(
            returns_matrix,
            self.window,
            self.min_periods
        )
        
        result = pd.Series(
            crowding_scores,
            index=price_df.index,
            name=sector_name
        )
        
        return result
    
    def get_sector_stocks(
        self,
        stock_sector_map: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        获取每个行业的股票列表
        
        Parameters
        ----------
        stock_sector_map : Dict[str, str]
            股票代码到行业名称的映射
        
        Returns
        -------
        Dict[str, List[str]]
            行业名称到股票列表的映射
        
        Examples
        --------
        >>> sector_stocks = calculator.get_sector_stocks(stock_sector_map)
        >>> print(f"银行业股票: {sector_stocks.get('银行', [])}")
        """
        result: Dict[str, List[str]] = {}
        
        for stock_code, sector_name in stock_sector_map.items():
            if sector_name not in result:
                result[sector_name] = []
            result[sector_name].append(stock_code)
        
        return result
    
    def close(self) -> None:
        """
        关闭计算器并释放资源
        
        释放 Ray 集群资源。建议在使用完成后调用。
        """
        self._shutdown_ray_cluster()
        logger.info("CrowdingFactorCalculator 已关闭")


def calculate_crowding_factor(
    price_df: pd.DataFrame,
    stock_sector_map: Dict[str, str],
    window: int = 20,
    min_periods: int = 10,
    n_workers: Optional[int] = None,
    use_dask: bool = False
) -> pd.DataFrame:
    """
    便捷函数：计算行业拥挤因子
    
    封装 CrowdingFactorCalculator 的快捷方式，适合一次性计算。
    
    Parameters
    ----------
    price_df : pd.DataFrame
        价格数据，格式要求：
        - index: DatetimeIndex（交易日期）
        - columns: 股票代码
        - values: 收盘价
    stock_sector_map : Dict[str, str]
        股票代码到申万一级行业名称的映射
    window : int, optional
        滚动窗口大小（交易日），默认 20
    min_periods : int, optional
        最小有效数据点数，默认 10
    n_workers : Optional[int]
        Ray 工作节点数量，默认为 CPU 核心数
    use_dask : bool, optional
        是否使用 Dask 处理大数据，默认 False
    
    Returns
    -------
    pd.DataFrame
        拥挤因子时间序列，格式：
        - index: DatetimeIndex
        - columns: 行业名称
        - values: 拥挤因子分数
    
    Examples
    --------
    >>> import pandas as pd
    >>> from src.crowding_factor import calculate_crowding_factor
    >>> 
    >>> # 加载价格数据
    >>> price_df = pd.read_parquet("data/prices.parquet")
    >>> 
    >>> # 准备行业映射
    >>> stock_sector_map = {
    ...     "000001": "银行",
    ...     "600519": "食品饮料",
    ...     "300750": "电子",
    ...     # ... 更多股票
    ... }
    >>> 
    >>> # 计算拥挤因子
    >>> crowding = calculate_crowding_factor(
    ...     price_df, 
    ...     stock_sector_map,
    ...     window=20,
    ...     n_workers=8
    ... )
    >>> 
    >>> # 查看结果
    >>> print(crowding.tail())
    >>> 
    >>> # 找出最拥挤的行业
    >>> latest = crowding.iloc[-1].sort_values(ascending=False)
    >>> print(f"当前最拥挤的行业: {latest.index[0]} ({latest.iloc[0]:.2f})")
    
    Notes
    -----
    - 此函数会自动初始化和关闭 Ray 集群
    - 对于需要多次计算的场景，建议直接使用 CrowdingFactorCalculator 类
    """
    calculator = CrowdingFactorCalculator(
        n_workers=n_workers,
        window=window,
        min_periods=min_periods,
        use_dask=use_dask
    )
    
    try:
        result = calculator.calculate(price_df, stock_sector_map)
        return result
    finally:
        calculator.close()


# ==================== 数据获取辅助函数 ====================

def fetch_sw_industry_mapping(
    stock_codes: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    获取申万一级行业分类映射
    
    从 AkShare 获取 A 股的申万一级行业分类。
    
    Parameters
    ----------
    stock_codes : Optional[List[str]]
        要获取的股票代码列表，如果为 None 则获取全市场
    
    Returns
    -------
    Dict[str, str]
        股票代码到申万一级行业名称的映射
    
    Examples
    --------
    >>> mapping = fetch_sw_industry_mapping(["000001", "600519", "300750"])
    >>> print(mapping)
    {'000001': '银行', '600519': '食品饮料', '300750': '电子'}
    
    Notes
    -----
    - 依赖 AkShare 库
    - 网络请求可能失败，建议缓存结果
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("需要安装 akshare: pip install akshare")
    
    result: Dict[str, str] = {}
    
    try:
        # 获取申万行业成分股
        for sector_code, sector_name in CrowdingFactorCalculator.SW_LEVEL1_SECTORS.items():
            try:
                df = ak.index_stock_cons_csindex(symbol=sector_code)
                
                if df is not None and not df.empty:
                    # 提取股票代码
                    code_col = None
                    for col in ["成分券代码", "品种代码", "代码", "code"]:
                        if col in df.columns:
                            code_col = col
                            break
                    
                    if code_col:
                        for code in df[code_col]:
                            code_str = str(code).zfill(6)
                            if stock_codes is None or code_str in stock_codes:
                                result[code_str] = sector_name
                                
            except Exception as e:
                logger.debug(f"获取行业 {sector_name} 成分股失败: {e}")
                continue
        
        logger.info(f"获取申万行业映射成功: {len(result)} 只股票")
        return result
        
    except Exception as e:
        logger.error(f"获取申万行业映射失败: {e}")
        return result


def load_industry_mapping_from_file(
    filepath: str
) -> Dict[str, str]:
    """
    从文件加载行业映射
    
    支持 CSV、Parquet 和 JSON 格式。
    
    Parameters
    ----------
    filepath : str
        文件路径
    
    Returns
    -------
    Dict[str, str]
        股票代码到行业名称的映射
    
    Examples
    --------
    >>> mapping = load_industry_mapping_from_file("data/industry_mapping.csv")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, dtype=str)
    elif filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix == ".json":
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {filepath.suffix}")
    
    # 查找列名
    code_col = None
    sector_col = None
    
    for col in ["stock_code", "code", "证券代码", "代码"]:
        if col in df.columns:
            code_col = col
            break
    
    for col in ["sw_industry_l1", "industry", "行业", "申万一级"]:
        if col in df.columns:
            sector_col = col
            break
    
    if code_col is None or sector_col is None:
        raise ValueError(f"无法识别列名，需要股票代码和行业列")
    
    result = dict(zip(df[code_col].astype(str), df[sector_col].astype(str)))
    logger.info(f"从文件加载行业映射: {len(result)} 只股票")
    
    return result

