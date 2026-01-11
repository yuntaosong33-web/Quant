"""
PyPortfolioOpt 投资组合优化工具模块

提供协方差矩阵估计、预期收益率计算和权重优化功能。
"""

from typing import Optional, Dict, Any, List
import logging

import pandas as pd


def calculate_shrinkage_covariance(
    prices: pd.DataFrame,
    shrinkage_type: str = "ledoit_wolf"
) -> pd.DataFrame:
    """
    计算压缩协方差矩阵
    
    使用 PyPortfolioOpt 的 CovarianceShrinkage 计算更稳健的协方差矩阵。
    
    Parameters
    ----------
    prices : pd.DataFrame
        价格数据，索引为日期，列为股票代码
    shrinkage_type : str, optional
        压缩方法，可选：
        - 'ledoit_wolf': Ledoit-Wolf 压缩（默认）
        - 'oracle_approximating': Oracle Approximating Shrinkage
        - 'exponential': 指数加权协方差
    
    Returns
    -------
    pd.DataFrame
        压缩后的协方差矩阵
    
    Examples
    --------
    >>> cov_matrix = calculate_shrinkage_covariance(prices_df)
    >>> print(cov_matrix.shape)
    
    Notes
    -----
    Ledoit-Wolf 压缩通过将样本协方差矩阵向结构化目标收缩，
    可以显著提高协方差估计的稳定性，特别是在样本量较小时。
    """
    try:
        from pypfopt import risk_models
    except ImportError:
        raise ImportError(
            "请安装 pypfopt: pip install pyportfolioopt"
        )
    
    if shrinkage_type == "ledoit_wolf":
        cov_matrix = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    elif shrinkage_type == "oracle_approximating":
        cov_matrix = risk_models.CovarianceShrinkage(prices).oracle_approximating()
    elif shrinkage_type == "exponential":
        cov_matrix = risk_models.exp_cov(prices)
    else:
        raise ValueError(f"不支持的压缩方法: {shrinkage_type}")
    
    logging.debug(f"协方差矩阵计算完成，形状: {cov_matrix.shape}")
    return cov_matrix


def calculate_expected_returns_mean(
    prices: pd.DataFrame,
    method: str = "mean_historical",
    frequency: int = 252
) -> pd.Series:
    """
    计算预期收益率
    
    Parameters
    ----------
    prices : pd.DataFrame
        价格数据，索引为日期，列为股票代码
    method : str, optional
        计算方法，可选：
        - 'mean_historical': 历史平均收益率（默认）
        - 'ema_historical': 指数加权平均收益率
        - 'capm': CAPM 模型（需要市场收益率）
    frequency : int, optional
        年化频率，默认252（日频）
    
    Returns
    -------
    pd.Series
        预期年化收益率
    """
    try:
        from pypfopt import expected_returns
    except ImportError:
        raise ImportError(
            "请安装 pypfopt: pip install pyportfolioopt"
        )
    
    if method == "mean_historical":
        mu = expected_returns.mean_historical_return(prices, frequency=frequency)
    elif method == "ema_historical":
        mu = expected_returns.ema_historical_return(prices, frequency=frequency)
    else:
        raise ValueError(f"不支持的计算方法: {method}")
    
    logging.debug(f"预期收益率计算完成，股票数: {len(mu)}")
    return mu


def optimize_weights(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.02,
    max_weight: float = 0.05,
    min_weight: float = 0.0,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
    sector_mapper: Optional[Dict[str, str]] = None,
    sector_lower: Optional[Dict[str, float]] = None,
    sector_upper: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    使用 PyPortfolioOpt 优化投资组合权重
    
    支持最大夏普比率、最小波动率等多种优化目标。
    
    Parameters
    ----------
    expected_returns : pd.Series
        预期收益率，索引为股票代码
    cov_matrix : pd.DataFrame
        协方差矩阵
    objective : str, optional
        优化目标，可选：
        - 'max_sharpe': 最大夏普比率（默认）
        - 'min_volatility': 最小波动率
        - 'efficient_return': 给定目标收益的最小风险
        - 'efficient_risk': 给定目标波动率的最大收益
    risk_free_rate : float, optional
        无风险利率，默认0.02（2%）
    max_weight : float, optional
        单只股票最大权重，默认0.05（5%）
    min_weight : float, optional
        单只股票最小权重，默认0.0
    target_return : Optional[float]
        目标收益率（用于 efficient_return）
    target_volatility : Optional[float]
        目标波动率（用于 efficient_risk）
    sector_mapper : Optional[Dict[str, str]]
        股票到行业的映射字典
    sector_lower : Optional[Dict[str, float]]
        行业权重下限
    sector_upper : Optional[Dict[str, float]]
        行业权重上限
    
    Returns
    -------
    Dict[str, Any]
        优化结果，包含：
        - 'weights': 优化后的权重字典
        - 'clean_weights': 清理后的权重（去除接近零的权重）
        - 'performance': (预期收益, 波动率, 夏普比率)
        - 'n_assets': 有效资产数量
    
    Examples
    --------
    >>> result = optimize_weights(
    ...     expected_returns, cov_matrix,
    ...     objective='max_sharpe',
    ...     max_weight=0.05
    ... )
    >>> print(result['clean_weights'])
    >>> print(f"夏普比率: {result['performance'][2]:.2f}")
    
    Notes
    -----
    - 权重之和约束为 1（全仓投资）
    - 使用 clean_weights 方法去除小于阈值的权重
    - 支持行业权重约束
    """
    try:
        from pypfopt import EfficientFrontier
    except ImportError:
        raise ImportError(
            "请安装 pypfopt: pip install pyportfolioopt"
        )
    
    # 确保索引一致
    common_assets = expected_returns.index.intersection(cov_matrix.index)
    if len(common_assets) < len(expected_returns):
        logging.warning(
            f"预期收益率和协方差矩阵的资产不完全匹配，"
            f"使用共同的 {len(common_assets)} 个资产"
        )
    
    mu = expected_returns.loc[common_assets]
    S = cov_matrix.loc[common_assets, common_assets]
    
    # 创建有效边界优化器
    ef = EfficientFrontier(
        mu, S,
        weight_bounds=(min_weight, max_weight)
    )
    
    # 添加行业约束（如果提供）
    if sector_mapper is not None:
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
    
    # 执行优化
    try:
        if objective == "max_sharpe":
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif objective == "min_volatility":
            weights = ef.min_volatility()
        elif objective == "efficient_return":
            if target_return is None:
                raise ValueError("efficient_return 需要指定 target_return")
            weights = ef.efficient_return(target_return=target_return)
        elif objective == "efficient_risk":
            if target_volatility is None:
                raise ValueError("efficient_risk 需要指定 target_volatility")
            weights = ef.efficient_risk(target_volatility=target_volatility)
        else:
            raise ValueError(f"不支持的优化目标: {objective}")
    except Exception as e:
        logging.error(f"优化失败: {e}")
        # 失败时返回等权重
        n_assets = len(common_assets)
        equal_weight = 1.0 / n_assets
        weights = {asset: equal_weight for asset in common_assets}
        return {
            'weights': weights,
            'clean_weights': weights,
            'performance': (0, 0, 0),
            'n_assets': n_assets,
            'status': 'failed',
            'error': str(e)
        }
    
    # 清理权重（去除接近零的权重）
    clean_weights = ef.clean_weights(cutoff=1e-4, rounding=4)
    
    # 计算组合绩效
    performance = ef.portfolio_performance(
        verbose=False,
        risk_free_rate=risk_free_rate
    )
    
    # 计算有效资产数量
    n_effective = sum(1 for w in clean_weights.values() if w > 1e-4)
    
    logging.info(
        f"优化完成 [{objective}]: "
        f"预期收益 {performance[0]:.2%}, "
        f"波动率 {performance[1]:.2%}, "
        f"夏普比率 {performance[2]:.2f}, "
        f"有效资产 {n_effective}"
    )
    
    return {
        'weights': dict(weights),
        'clean_weights': clean_weights,
        'performance': performance,
        'n_assets': n_effective,
        'status': 'success'
    }


def optimize_portfolio_from_prices(
    prices: pd.DataFrame,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.02,
    max_weight: float = 0.05,
    min_weight: float = 0.0,
    shrinkage_type: str = "ledoit_wolf",
    returns_method: str = "mean_historical"
) -> Dict[str, Any]:
    """
    从价格数据直接优化投资组合（便捷函数）
    
    整合预期收益率计算、协方差矩阵估计和优化过程。
    
    Parameters
    ----------
    prices : pd.DataFrame
        价格数据，索引为日期，列为股票代码
    objective : str, optional
        优化目标，默认 'max_sharpe'
    risk_free_rate : float, optional
        无风险利率，默认0.02
    max_weight : float, optional
        单只股票最大权重，默认0.05
    min_weight : float, optional
        单只股票最小权重，默认0.0
    shrinkage_type : str, optional
        协方差压缩方法，默认 'ledoit_wolf'
    returns_method : str, optional
        预期收益率计算方法，默认 'mean_historical'
    
    Returns
    -------
    Dict[str, Any]
        优化结果
    
    Examples
    --------
    >>> result = optimize_portfolio_from_prices(
    ...     prices_df,
    ...     objective='max_sharpe',
    ...     max_weight=0.05
    ... )
    >>> weights = result['clean_weights']
    """
    # 计算预期收益率
    mu = calculate_expected_returns_mean(prices, method=returns_method)
    
    # 计算压缩协方差矩阵
    S = calculate_shrinkage_covariance(prices, shrinkage_type=shrinkage_type)
    
    # 优化权重
    result = optimize_weights(
        mu, S,
        objective=objective,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        min_weight=min_weight
    )
    
    # 添加额外信息
    result['expected_returns'] = mu
    result['cov_matrix'] = S
    
    return result


class PortfolioWeightOptimizer:
    """
    投资组合权重优化器
    
    封装 PyPortfolioOpt 的功能，提供灵活的权重优化接口。
    
    Parameters
    ----------
    risk_free_rate : float
        无风险利率，默认0.02
    max_weight : float
        单只股票最大权重，默认0.05
    min_weight : float
        单只股票最小权重，默认0.0
    
    Attributes
    ----------
    expected_returns : pd.Series
        预期收益率
    cov_matrix : pd.DataFrame
        协方差矩阵
    weights : Dict[str, float]
        优化后的权重
    
    Examples
    --------
    >>> optimizer = PortfolioWeightOptimizer(max_weight=0.05)
    >>> optimizer.fit(prices_df)
    >>> weights = optimizer.optimize('max_sharpe')
    >>> print(optimizer.get_portfolio_stats())
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 0.05,
        min_weight: float = 0.0,
        shrinkage_type: str = "ledoit_wolf"
    ) -> None:
        """
        初始化优化器
        
        Parameters
        ----------
        risk_free_rate : float
            无风险利率
        max_weight : float
            单只股票最大权重
        min_weight : float
            单只股票最小权重
        shrinkage_type : str
            协方差压缩方法
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.shrinkage_type = shrinkage_type
        
        self.expected_returns: Optional[pd.Series] = None
        self.cov_matrix: Optional[pd.DataFrame] = None
        self.weights: Optional[Dict[str, float]] = None
        self._performance: Optional[tuple] = None
    
    def fit(
        self,
        prices: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> "PortfolioWeightOptimizer":
        """
        拟合优化器
        
        可以从价格数据自动计算，或直接提供预期收益率和协方差矩阵。
        
        Parameters
        ----------
        prices : Optional[pd.DataFrame]
            价格数据
        expected_returns : Optional[pd.Series]
            预期收益率（如果提供则优先使用）
        cov_matrix : Optional[pd.DataFrame]
            协方差矩阵（如果提供则优先使用）
        
        Returns
        -------
        self
            返回自身以支持链式调用
        """
        if expected_returns is not None:
            self.expected_returns = expected_returns
        elif prices is not None:
            self.expected_returns = calculate_expected_returns_mean(prices)
        else:
            raise ValueError("必须提供 prices 或 expected_returns")
        
        if cov_matrix is not None:
            self.cov_matrix = cov_matrix
        elif prices is not None:
            self.cov_matrix = calculate_shrinkage_covariance(
                prices, self.shrinkage_type
            )
        else:
            raise ValueError("必须提供 prices 或 cov_matrix")
        
        logging.info(f"优化器拟合完成，资产数: {len(self.expected_returns)}")
        return self
    
    def optimize(
        self,
        objective: str = "max_sharpe",
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        执行优化
        
        Parameters
        ----------
        objective : str
            优化目标
        target_return : Optional[float]
            目标收益率
        target_volatility : Optional[float]
            目标波动率
        
        Returns
        -------
        Dict[str, float]
            清理后的权重字典
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("请先调用 fit 方法")
        
        result = optimize_weights(
            self.expected_returns,
            self.cov_matrix,
            objective=objective,
            risk_free_rate=self.risk_free_rate,
            max_weight=self.max_weight,
            min_weight=self.min_weight,
            target_return=target_return,
            target_volatility=target_volatility
        )
        
        self.weights = result['clean_weights']
        self._performance = result['performance']
        
        return self.weights
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """
        获取组合统计信息
        
        Returns
        -------
        Dict[str, float]
            包含预期收益、波动率、夏普比率的字典
        """
        if self._performance is None:
            raise ValueError("请先调用 optimize 方法")
        
        return {
            'expected_return': self._performance[0],
            'volatility': self._performance[1],
            'sharpe_ratio': self._performance[2]
        }
    
    def get_weights_series(self) -> pd.Series:
        """
        获取权重 Series
        
        Returns
        -------
        pd.Series
            权重序列，索引为股票代码
        """
        if self.weights is None:
            raise ValueError("请先调用 optimize 方法")
        
        return pd.Series(self.weights).sort_values(ascending=False)
    
    def get_effective_assets(self, threshold: float = 1e-4) -> List[str]:
        """
        获取有效资产列表
        
        Parameters
        ----------
        threshold : float
            权重阈值
        
        Returns
        -------
        List[str]
            有效资产列表
        """
        if self.weights is None:
            raise ValueError("请先调用 optimize 方法")
        
        return [asset for asset, weight in self.weights.items() if weight > threshold]

