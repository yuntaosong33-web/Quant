"""
A股市场数据清洗器

提供针对A股市场数据的专业清洗功能，包括停牌处理、涨跌停识别等。
所有方法采用Pandas向量化操作，确保高性能处理大规模数据。
"""

from typing import Optional, Dict
from pathlib import Path
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AShareDataCleaner:
    """
    A股市场数据清洗器
    
    提供针对A股市场数据的专业清洗功能，包括停牌处理、涨跌停识别等。
    
    Attributes
    ----------
    output_dir : str
        清洗后数据的输出目录
    compression : str
        Parquet文件压缩算法
    
    Examples
    --------
    >>> cleaner = AShareDataCleaner(output_dir="data/processed")
    >>> cleaned_df = cleaner.clean_market_data(raw_df)
    >>> print(cleaned_df["is_limit"].sum())  # 涨跌停天数
    """
    
    # A股涨跌停幅度常量
    LIMIT_PCT_MAIN = 0.10      # 主板涨跌停幅度 10%
    LIMIT_PCT_GEM_STAR = 0.20  # 创业板/科创板涨跌停幅度 20%
    LIMIT_PCT_ST = 0.05        # ST股票涨跌停幅度 5%
    
    def __init__(
        self,
        output_dir: str = "data/processed",
        compression: str = "snappy"
    ) -> None:
        """
        初始化数据清洗器
        
        Parameters
        ----------
        output_dir : str, optional
            清洗后数据的输出目录，默认为 'data/processed'
        compression : str, optional
            Parquet压缩算法，默认为 'snappy'
        """
        self.output_dir = output_dir
        self.compression = compression
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"A股数据清洗器初始化完成，输出目录: {output_dir}")
    
    def clean_market_data(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        清洗A股市场数据
        
        执行完整的数据清洗流程，包括：
        1. 日期索引转换
        2. 停牌日识别与处理
        3. 一字涨跌停识别
        4. 数据保存
        
        Parameters
        ----------
        df : pd.DataFrame
            原始市场数据
        symbol : Optional[str]
            股票代码，用于保存文件命名和判断涨跌停幅度
        save : bool, optional
            是否保存清洗后的数据，默认为True
        
        Returns
        -------
        pd.DataFrame
            清洗后的数据，包含新增列：is_suspended, is_limit, is_limit_up, is_limit_down
        """
        df = df.copy()
        
        df = self._convert_datetime_index(df)
        df = self._handle_suspended_days(df)
        df = self._identify_limit_days(df, symbol)
        
        if save and symbol:
            self._save_to_parquet(df, symbol)
        
        logger.info(
            f"数据清洗完成: 总记录={len(df)}, "
            f"停牌={df['is_suspended'].sum()}, "
            f"一字板={df['is_limit'].sum()}"
        )
        
        return df
    
    def _convert_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """将日期列转换为DatetimeIndex"""
        date_columns = ["日期", "date", "Date", "DATE", "trade_date"]
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"未找到日期列，需要以下列之一: {date_columns}")
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        
        df = df.sort_index()
        df.index.name = "date"
        
        return df
    
    def _handle_suspended_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别停牌日并填充NaN"""
        volume_cols = ["volume", "成交量", "vol", "Volume"]
        volume_col = None
        
        for col in volume_cols:
            if col in df.columns:
                volume_col = col
                break
        
        if volume_col is None:
            logger.warning("未找到成交量列，跳过停牌日处理")
            df["is_suspended"] = False
            return df
        
        is_suspended = (df[volume_col] == 0) | df[volume_col].isna()
        df["is_suspended"] = is_suspended
        
        ohlc_cols = ["open", "high", "low", "close"]
        ohlc_chinese = ["开盘", "最高", "最低", "收盘"]
        
        existing_ohlc = [col for col in ohlc_cols + ohlc_chinese if col in df.columns]
        
        if existing_ohlc:
            df.loc[is_suspended, existing_ohlc] = np.nan
        
        suspended_count = is_suspended.sum()
        if suspended_count > 0:
            logger.info(f"识别到 {suspended_count} 个停牌日，已填充NaN")
        
        return df
    
    def _identify_limit_days(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """识别一字涨跌停"""
        limit_pct = self._get_limit_pct(symbol)
        
        open_col = "open" if "open" in df.columns else "开盘"
        high_col = "high" if "high" in df.columns else "最高"
        low_col = "low" if "low" in df.columns else "最低"
        close_col = "close" if "close" in df.columns else "收盘"
        
        required = [open_col, high_col, low_col, close_col]
        if not all(col in df.columns for col in required):
            logger.warning("缺少OHLC列，无法识别一字涨跌停")
            df["is_limit"] = False
            df["is_limit_up"] = False
            df["is_limit_down"] = False
            return df
        
        # 向量化判断一字板
        price_equal = (
            np.isclose(df[open_col], df[close_col], rtol=1e-6) &
            np.isclose(df[high_col], df[close_col], rtol=1e-6) &
            np.isclose(df[low_col], df[close_col], rtol=1e-6)
        )
        
        # 获取涨跌幅
        pct_cols = ["pct_change", "涨跌幅", "pctChg", "change_pct"]
        pct_col = None
        for col in pct_cols:
            if col in df.columns:
                pct_col = col
                break
        
        if pct_col:
            pct_change = df[pct_col]
            if pct_change.abs().max() > 1:
                pct_change = pct_change / 100
            
            is_limit_up = price_equal & (pct_change >= (limit_pct - 0.005))
            is_limit_down = price_equal & (pct_change <= -(limit_pct - 0.005))
        else:
            prev_close = df[close_col].shift(1)
            pct_change = (df[close_col] - prev_close) / prev_close
            
            is_limit_up = price_equal & (pct_change >= (limit_pct - 0.005))
            is_limit_down = price_equal & (pct_change <= -(limit_pct - 0.005))
        
        if "is_suspended" in df.columns:
            is_limit_up = is_limit_up & ~df["is_suspended"]
            is_limit_down = is_limit_down & ~df["is_suspended"]
        
        df["is_limit_up"] = is_limit_up.fillna(False).astype(bool)
        df["is_limit_down"] = is_limit_down.fillna(False).astype(bool)
        df["is_limit"] = df["is_limit_up"] | df["is_limit_down"]
        
        limit_up_count = df["is_limit_up"].sum()
        limit_down_count = df["is_limit_down"].sum()
        
        if limit_up_count > 0 or limit_down_count > 0:
            logger.info(f"识别到一字涨停 {limit_up_count} 天, 一字跌停 {limit_down_count} 天")
        
        return df
    
    def _get_limit_pct(self, symbol: Optional[str] = None) -> float:
        """根据股票代码判断涨跌停幅度"""
        if symbol is None:
            return self.LIMIT_PCT_MAIN
        
        code = symbol.lstrip("shszSHSZ.")
        
        if code.startswith(("300", "301")):
            return self.LIMIT_PCT_GEM_STAR
        
        if code.startswith(("688", "689")):
            return self.LIMIT_PCT_GEM_STAR
        
        if code.startswith(("8", "4")) and len(code) == 6:
            return 0.30  # 北交所30%
        
        return self.LIMIT_PCT_MAIN
    
    def _save_to_parquet(self, df: pd.DataFrame, symbol: str) -> str:
        """保存清洗后的数据为Parquet格式"""
        filepath = Path(self.output_dir) / f"{symbol}_cleaned.parquet"
        
        df.to_parquet(filepath, compression=self.compression, index=True)
        
        logger.info(f"清洗后数据已保存: {filepath}")
        return str(filepath)
    
    def batch_clean(
        self,
        data_dict: Dict[str, pd.DataFrame],
        parallel: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """批量清洗多只股票数据"""
        result = {}
        total = len(data_dict)
        
        for i, (symbol, df) in enumerate(data_dict.items(), 1):
            logger.info(f"清洗 {symbol} ({i}/{total})")
            try:
                cleaned = self.clean_market_data(df, symbol=symbol)
                result[symbol] = cleaned
            except Exception as e:
                logger.error(f"清洗 {symbol} 失败: {e}")
                continue
        
        logger.info(f"批量清洗完成: {len(result)}/{total}")
        return result


__all__ = ['AShareDataCleaner']

