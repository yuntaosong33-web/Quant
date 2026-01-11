"""
因子标准化处理函数

提供 Z-Score 标准化、财务数据滞后处理等工具函数。
"""

from typing import Optional, List, Union
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def z_score_normalize(
    data: pd.DataFrame,
    factor_cols: Union[str, List[str]],
    date_col: str,
    industry_col: Optional[str] = None,
    industry_neutral: bool = True
) -> pd.DataFrame:
    """
    独立的 Z-Score 标准化函数（支持行业中性化）
    
    对每一天的截面数据进行 Z-Score 标准化。
    若指定行业列，则在行业内部进行标准化（行业中性化）。
    
    Parameters
    ----------
    data : pd.DataFrame
        包含因子数据的 DataFrame
    factor_cols : Union[str, List[str]]
        需要标准化的因子列名
    date_col : str
        日期列名
    industry_col : Optional[str]
        行业分类列名（申万一级行业），默认 None
    industry_neutral : bool
        是否进行行业中性化，默认 True
    
    Returns
    -------
    pd.DataFrame
        标准化后的数据框，新增 '{factor}_zscore' 列
    """
    if isinstance(factor_cols, str):
        factor_cols = [factor_cols]
    
    result = data
    unique_dates = result[date_col].unique() if date_col in result.columns else [None]
    n_dates = len(unique_dates)
    
    logger.debug(f"Z-Score 标准化: {len(factor_cols)} 个因子, {n_dates} 个日期")
    
    for col in factor_cols:
        if col not in result.columns:
            logger.debug(f"因子列 {col} 不存在，跳过")
            continue
        
        zscore_col = f'{col}_zscore'
        
        try:
            if industry_neutral and industry_col is not None and industry_col in result.columns:
                zscore_values = np.zeros(len(result))
                for date_val in unique_dates:
                    date_mask = result[date_col] == date_val
                    for ind_val in result.loc[date_mask, industry_col].unique():
                        mask = date_mask & (result[industry_col] == ind_val)
                        if mask.sum() > 1:
                            vals = result.loc[mask, col].values
                            valid_count = np.sum(~np.isnan(vals))
                            if valid_count >= 2:
                                mean_val = np.nanmean(vals)
                                std_val = np.nanstd(vals)
                                if std_val > 1e-10:
                                    zscore_values[mask] = (vals - mean_val) / std_val
                result[zscore_col] = zscore_values
            else:
                zscore_values = np.zeros(len(result))
                for date_val in unique_dates:
                    mask = result[date_col] == date_val
                    if mask.sum() > 1:
                        vals = result.loc[mask, col].values
                        valid_count = np.sum(~np.isnan(vals))
                        if valid_count >= 2:
                            mean_val = np.nanmean(vals)
                            std_val = np.nanstd(vals)
                            if std_val > 1e-10:
                                zscore_values[mask] = (vals - mean_val) / std_val
                result[zscore_col] = zscore_values
            
            result[zscore_col] = result[zscore_col].fillna(0.0)
            
        except Exception as e:
            logger.warning(f"因子 {col} Z-Score 标准化失败: {e}，设为 0")
            result[zscore_col] = 0.0
    
    return result


def lag_fundamental_data(
    data: pd.DataFrame,
    lag_days: int = 60,
    date_col: Optional[str] = None,
    stock_col: str = "stock_code",
    exclude_market_value: bool = True
) -> pd.DataFrame:
    """
    对财务/基本面数据进行滞后处理，避免前视偏差（Look-Ahead Bias）
    
    A股财报发布时间规则决定了财务数据的实际可获得时间：
    - 年报（Q4）：次年4月30日前发布（滞后约4个月）
    - 半年报（Q2）：8月31日前发布（滞后约2个月）
    - 一季报（Q1）：4月30日前发布（滞后约1个月）
    - 三季报（Q3）：10月31日前发布（滞后约1个月）
    
    Parameters
    ----------
    data : pd.DataFrame
        财务/基本面数据
    lag_days : int, optional
        滞后的自然日天数，默认 60（约3个月）
    date_col : Optional[str]
        日期列名
    stock_col : str
        股票代码列名，默认 'stock_code'
    exclude_market_value : bool
        是否将市值列单独处理，默认 True
    
    Returns
    -------
    pd.DataFrame
        滞后后的财务数据
    """
    if data.empty:
        logger.warning("输入数据为空，直接返回")
        return data
    
    df = data.copy()
    lag_delta = pd.Timedelta(days=lag_days)
    
    # 市值类列名
    MARKET_VALUE_COLS = ['circ_mv', 'total_mv', '流通市值', '总市值']
    MV_LAG_DAYS = 1
    
    # 自动识别日期列
    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index + lag_delta
            logger.info(f"DatetimeIndex 已向后偏移 {lag_days} 天")
            return df
        
        for col in ['date', 'trade_date', 'report_date']:
            if col in df.columns:
                date_col = col
                break
    
    if date_col is None:
        logger.warning("未找到日期列，无法进行滞后处理")
        return df
    
    # 分离市值列
    if exclude_market_value:
        mv_cols_in_data = [c for c in MARKET_VALUE_COLS if c in df.columns]
        if mv_cols_in_data:
            # 市值数据仅滞后1天
            mv_data = df[[stock_col, date_col] + mv_cols_in_data].copy() if stock_col in df.columns else None
            if mv_data is not None:
                mv_data[date_col] = pd.to_datetime(mv_data[date_col]) + pd.Timedelta(days=MV_LAG_DAYS)
                df = df.drop(columns=mv_cols_in_data)
    
    # 对其他财务数据进行滞后
    df[date_col] = pd.to_datetime(df[date_col]) + lag_delta
    logger.info(f"财务数据日期列 {date_col} 已向后偏移 {lag_days} 天")
    
    # 合并市值数据
    if exclude_market_value and 'mv_data' in dir() and mv_data is not None:
        merge_keys = [stock_col, date_col] if stock_col in df.columns else [date_col]
        df = df.merge(mv_data, on=merge_keys, how='left')
        logger.info(f"市值数据已合并（滞后 {MV_LAG_DAYS} 天）")
    
    return df


def safe_zscore(x: pd.Series) -> pd.Series:
    """
    安全的 Z-Score 计算，处理标准差为 0 的情况
    
    Parameters
    ----------
    x : pd.Series
        输入序列
    
    Returns
    -------
    pd.Series
        Z-Score 标准化后的序列
    """
    mean_val = x.mean()
    std_val = x.std()
    
    if std_val == 0 or pd.isna(std_val):
        return pd.Series(0.0, index=x.index)
    
    return (x - mean_val) / std_val


__all__ = [
    'z_score_normalize',
    'lag_fundamental_data',
    'safe_zscore',
]

