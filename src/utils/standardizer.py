"""
数据标准化工具模块

提供数据源列名统一映射和单位转换功能。
"""

from typing import Dict, List
import logging

import pandas as pd


class DataStandardizer:
    """
    数据源列名标准化器
    
    将 Tushare Pro 返回的 DataFrame 统一映射为标准格式。
    支持 OHLCV + adj_factor 等常用字段的标准化。
    
    Attributes
    ----------
    STANDARD_COLUMNS : List[str]
        标准列名列表
    TUSHARE_MAPPING : Dict[str, str]
        Tushare 列名到标准列名的映射
    
    Examples
    --------
    >>> standardizer = DataStandardizer()
    >>> 
    >>> # 标准化 Tushare 数据
    >>> df_tushare = ts.pro_api().daily(ts_code='000001.SZ')
    >>> df_std = standardizer.standardize(df_tushare, source='tushare')
    """
    
    # 标准列名（输出格式）
    STANDARD_COLUMNS: List[str] = [
        'date', 'open', 'high', 'low', 'close', 
        'volume', 'amount', 'adj_factor', 'pct_change', 'turnover'
    ]
    
    # Tushare Pro 列名映射
    TUSHARE_MAPPING: Dict[str, str] = {
        'trade_date': 'date',
        'ts_code': 'symbol',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'vol': 'volume',          # Tushare 用 vol 表示成交量（手）
        'amount': 'amount',       # 成交额（千元）
        'adj_factor': 'adj_factor',
        'pct_chg': 'pct_change',  # 涨跌幅
        'change': 'change',       # 涨跌额
        'turnover_rate': 'turnover',  # 换手率
        'pre_close': 'pre_close',
    }
    
    # 数值单位转换因子
    TUSHARE_VOLUME_FACTOR: float = 100.0    # Tushare vol 单位是手，转换为股需 * 100
    TUSHARE_AMOUNT_FACTOR: float = 1000.0   # Tushare amount 单位是千元，转换为元需 * 1000
    
    def __init__(self) -> None:
        """初始化数据标准化器"""
        self._logger = logging.getLogger(__name__)
    
    def standardize(
        self,
        df: pd.DataFrame,
        source: str,
        convert_units: bool = True,
        set_datetime_index: bool = True
    ) -> pd.DataFrame:
        """
        统一标准化入口
        
        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        source : str
            数据源，目前仅支持 'tushare'
        convert_units : bool, optional
            是否转换单位（如 Tushare 的成交量从手转换为股），默认 True
        set_datetime_index : bool, optional
            是否将日期列设置为 DatetimeIndex，默认 True
        
        Returns
        -------
        pd.DataFrame
            标准化后的数据，包含统一的列名和格式
        
        Raises
        ------
        ValueError
            当数据源不支持时
        
        Examples
        --------
        >>> standardizer = DataStandardizer()
        >>> df_std = standardizer.standardize(raw_df, source='tushare')
        >>> print(df_std.columns.tolist())
        ['open', 'high', 'low', 'close', 'volume', 'amount', ...]
        """
        if df.empty:
            self._logger.warning("输入数据为空")
            return df
        
        df = df.copy()
        
        # 根据数据源选择映射
        if source.lower() == 'tushare':
            df = self._standardize_tushare(df, convert_units)
        else:
            raise ValueError(f"不支持的数据源: {source}，目前仅支持 'tushare'")
        
        # 设置日期索引
        if set_datetime_index and 'date' in df.columns:
            df = self._set_datetime_index(df)
        
        return df
    
    def _standardize_tushare(
        self,
        df: pd.DataFrame,
        convert_units: bool = True
    ) -> pd.DataFrame:
        """
        标准化 Tushare Pro 数据
        
        Parameters
        ----------
        df : pd.DataFrame
            Tushare 原始数据
        convert_units : bool
            是否转换单位
        
        Returns
        -------
        pd.DataFrame
            标准化后的数据
        """
        # 重命名列
        df = df.rename(columns=self.TUSHARE_MAPPING)
        
        # 转换日期格式（Tushare 日期格式为 YYYYMMDD）
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        
        # 单位转换
        if convert_units:
            # 成交量：手 -> 股
            if 'volume' in df.columns:
                df['volume'] = df['volume'] * self.TUSHARE_VOLUME_FACTOR
            
            # 成交额：千元 -> 元
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * self.TUSHARE_AMOUNT_FACTOR
        
        return df
    
    def _set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        设置日期时间索引
        
        Parameters
        ----------
        df : pd.DataFrame
            数据
        
        Returns
        -------
        pd.DataFrame
            设置索引后的数据
        """
        if 'date' in df.columns:
            df = df.set_index('date')
            df = df.sort_index()
            df.index.name = 'date'
        return df
    
    def get_required_columns(self, source: str) -> List[str]:
        """
        获取指定数据源的必需列名
        
        Parameters
        ----------
        source : str
            数据源，目前仅支持 'tushare'
        
        Returns
        -------
        List[str]
            必需列名列表
        """
        if source.lower() == 'tushare':
            return ['trade_date', 'open', 'high', 'low', 'close', 'vol']
        else:
            raise ValueError(f"不支持的数据源: {source}，目前仅支持 'tushare'")
    
    def validate_columns(self, df: pd.DataFrame, source: str) -> bool:
        """
        验证数据是否包含必需列
        
        Parameters
        ----------
        df : pd.DataFrame
            数据
        source : str
            数据源
        
        Returns
        -------
        bool
            是否通过验证
        """
        required = self.get_required_columns(source)
        missing = set(required) - set(df.columns)
        
        if missing:
            self._logger.warning(f"数据缺少必需列: {missing}")
            return False
        
        return True

