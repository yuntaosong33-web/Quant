"""
因子有效性监控 (Factor IC Analysis)

提供因子 IC (Information Coefficient) 计算和前瞻收益计算功能。
"""

from typing import List, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_factor_ic(
    data: pd.DataFrame,
    factor_cols: List[str],
    return_col: str = 'forward_return_5d',
    date_col: str = 'date',
    stock_col: str = 'stock_code',
    log_results: bool = True
) -> pd.DataFrame:
    """
    计算因子 IC (Information Coefficient)
    
    IC 是因子值与未来收益的秩相关系数（Spearman），用于评估因子预测能力。
    
    Parameters
    ----------
    data : pd.DataFrame
        包含因子数据和收益率的 DataFrame
    factor_cols : List[str]
        需要评估的因子列名列表
    return_col : str
        收益率列名，默认 'forward_return_5d'
    date_col : str
        日期列名，默认 'date'
    stock_col : str
        股票代码列名，默认 'stock_code'
    log_results : bool
        是否记录日志，默认 True
    
    Returns
    -------
    pd.DataFrame
        因子 IC 统计结果，包含以下列:
        - factor: 因子名称
        - ic_mean: IC 均值
        - ic_std: IC 标准差
        - ic_ir: IC_IR (IC均值/IC标准差)
        - ic_positive_ratio: 正 IC 比例
        - t_stat: t 统计量
        - status: 有效性状态 ('有效', '边缘', '失效')
    
    Notes
    -----
    - IC > 0.03 通常被认为是有效因子
    - IC_IR > 0.5 表示因子预测能力稳定
    - 正 IC 比例 > 60% 表示因子方向稳定
    """
    if data.empty:
        logger.warning("输入数据为空，无法计算 IC")
        return pd.DataFrame()
    
    # 检查收益率列是否存在
    if return_col not in data.columns:
        if 'close' in data.columns:
            logger.info(f"'{return_col}' 列不存在，尝试从 close 计算 5 日前瞻收益")
            data = data.copy()
            if stock_col in data.columns:
                data[return_col] = data.groupby(stock_col)['close'].transform(
                    lambda x: x.shift(-5) / x - 1
                )
            else:
                data[return_col] = data['close'].shift(-5) / data['close'] - 1
        else:
            logger.warning(f"无法计算 IC: 缺少 '{return_col}' 列且无法自动生成")
            return pd.DataFrame()
    
    results = []
    
    def _safe_spearman_ic(x: pd.DataFrame, factor: str, ret: str) -> float:
        """
        安全计算 Spearman 相关，避免常数输入触发警告并污染结果。
        """
        if factor not in x.columns or ret not in x.columns:
            return float('nan')

        a = x[factor]
        b = x[ret]
        mask = a.notna() & b.notna()
        if mask.sum() <= 5:
            return float('nan')

        a = a[mask]
        b = b[mask]

        # 常数序列（或全相同 rank）会导致 Spearman 不定义
        if a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
            return float('nan')

        return float(a.corr(b, method='spearman'))

    for col in factor_cols:
        if col not in data.columns:
            logger.debug(f"因子列 '{col}' 不存在，跳过")
            continue
        
        try:
            # 按日期计算每日 IC（Spearman 秩相关）
            if date_col in data.columns:
                ic_by_date = data.groupby(date_col).apply(
                    lambda x: _safe_spearman_ic(x, col, return_col),
                    include_groups=False
                )
            else:
                ic_value = _safe_spearman_ic(data, col, return_col)
                ic_by_date = pd.Series([ic_value])
            
            ic_by_date = ic_by_date.dropna()
            
            if len(ic_by_date) == 0:
                logger.warning(f"因子 '{col}' 无有效 IC 数据")
                continue
            
            # 计算统计量
            ic_mean = ic_by_date.mean()
            ic_std = ic_by_date.std()
            ic_ir = ic_mean / (ic_std + 1e-8)
            ic_positive_ratio = (ic_by_date > 0).mean()
            
            # t 统计量
            n = len(ic_by_date)
            t_stat = ic_mean / (ic_std / np.sqrt(n) + 1e-8) if n > 1 else 0
            
            # 有效性判断
            if abs(ic_mean) >= 0.03 and ic_ir >= 0.5:
                status = "有效"
            elif abs(ic_mean) >= 0.02 or ic_ir >= 0.3:
                status = "边缘"
            else:
                status = "失效"
            
            results.append({
                'factor': col,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'ic_positive_ratio': ic_positive_ratio,
                't_stat': t_stat,
                'n_periods': n,
                'status': status
            })
            
        except Exception as e:
            logger.warning(f"计算因子 '{col}' IC 失败: {e}")
            continue
    
    if not results:
        logger.warning("没有成功计算任何因子的 IC")
        return pd.DataFrame()
    
    ic_df = pd.DataFrame(results)
    
    # 日志输出
    if log_results:
        logger.info("=" * 50)
        logger.info("因子 IC 监控报告")
        logger.info("=" * 50)
        for _, row in ic_df.iterrows():
            status_icon = "✅" if row['status'] == "有效" else ("⚠️" if row['status'] == "边缘" else "❌")
            logger.info(
                f"{status_icon} {row['factor']}: "
                f"IC={row['ic_mean']:.4f}, IC_IR={row['ic_ir']:.2f}, "
                f"正IC率={row['ic_positive_ratio']:.1%} [{row['status']}]"
            )
        logger.info("=" * 50)
    
    return ic_df


def calculate_forward_returns(
    data: pd.DataFrame,
    periods: List[int] = None,
    stock_col: str = 'stock_code',
    price_col: str = 'close',
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    计算多期前瞻收益
    
    Parameters
    ----------
    data : pd.DataFrame
        价格数据
    periods : List[int]
        收益计算期数列表，默认 [1, 5, 10, 20]
    stock_col : str
        股票代码列名
    price_col : str
        价格列名
    
    Returns
    -------
    pd.DataFrame
        添加了前瞻收益列的数据框
    """
    if periods is None:
        periods = [1, 5, 10, 20]
    
    result = data.copy()

    # 关键：前瞻收益必须按日期排序，否则 shift(-N) 会错位，导致 IC 失真/全 NaN
    if stock_col in result.columns:
        _date_col = date_col
        if _date_col is None:
            _date_col = 'trade_date' if 'trade_date' in result.columns else ('date' if 'date' in result.columns else None)

        if _date_col is not None and _date_col in result.columns:
            try:
                result[_date_col] = pd.to_datetime(result[_date_col])
                result = result.sort_values([stock_col, _date_col], kind='mergesort')
            except Exception:
                # 排序失败则降级（不抛错），但可能影响结果精度
                pass
    
    for period in periods:
        col_name = f'forward_return_{period}d'
        
        if stock_col in result.columns:
            result[col_name] = result.groupby(stock_col)[price_col].transform(
                lambda x: x.shift(-period) / x - 1
            )
        else:
            result[col_name] = result[price_col].shift(-period) / result[price_col] - 1
        
        result[col_name] = result[col_name].replace([np.inf, -np.inf], np.nan)
        logger.debug(f"前瞻收益 {col_name} 计算完成")
    
    return result


__all__ = [
    'calculate_factor_ic',
    'calculate_forward_returns',
]

