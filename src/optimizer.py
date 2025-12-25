"""
投资组合优化模块

基于PyPortfolioOpt实现均值-方差优化、风险平价等组合优化算法。
支持多种协方差估计方法和约束条件。
"""

from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass
import logging
import warnings

import pandas as pd
import numpy as np

# PyPortfolioOpt 导入
try:
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt import objective_functions
    from pypfopt.exceptions import OptimizationError
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    warnings.warn("pypfopt未安装，组合优化功能不可用。请运行: pip install pypfopt")

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    组合优化结果数据类
    
    Attributes
    ----------
    weights : Dict[str, float]
        资产权重字典
    expected_return : float
        预期收益率
    volatility : float
        预期波动率
    sharpe_ratio : float
        夏普比率
    success : bool
        优化是否成功
    message : str
        优化状态消息
    """
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str


def optimize_max_sharpe(
    mu: Union[pd.Series, np.ndarray],
    prices: pd.DataFrame,
    max_weight: float = 0.10,
    min_weight: float = 0.0,
    risk_free_rate: float = 0.03,
    cov_method: str = "ledoit_wolf",
    l2_gamma: float = 0.0
) -> OptimizationResult:
    """
    最大夏普比率组合优化
    
    使用均值-方差优化框架求解最大夏普比率的投资组合权重。
    采用Ledoit-Wolf压缩协方差矩阵估计以提高稳健性。
    
    Parameters
    ----------
    mu : Union[pd.Series, np.ndarray]
        预期收益向量，索引或顺序需与prices的列对应
    prices : pd.DataFrame
        历史价格数据，行索引为日期，列为资产代码
    max_weight : float, optional
        单只资产最大权重，默认0.10 (10%)
    min_weight : float, optional
        单只资产最小权重，默认0.0
    risk_free_rate : float, optional
        无风险利率，默认0.03 (3%)
    cov_method : str, optional
        协方差估计方法，可选:
        - 'ledoit_wolf': Ledoit-Wolf压缩估计（默认，推荐）
        - 'sample': 样本协方差
        - 'semicovariance': 半协方差（下行风险）
        - 'exp_cov': 指数加权协方差
    l2_gamma : float, optional
        L2正则化系数，用于防止过拟合，默认0.0
    
    Returns
    -------
    OptimizationResult
        优化结果，包含权重、预期收益、波动率和夏普比率
    
    Raises
    ------
    ValueError
        当输入数据无效时
    
    Examples
    --------
    >>> import pandas as pd
    >>> from src.optimizer import optimize_max_sharpe
    >>> 
    >>> # 准备数据
    >>> prices = pd.DataFrame(...)  # 历史价格
    >>> mu = expected_returns.mean_historical_return(prices)
    >>> 
    >>> # 优化
    >>> result = optimize_max_sharpe(mu, prices, max_weight=0.10)
    >>> 
    >>> if result.success:
    ...     print(f"夏普比率: {result.sharpe_ratio:.2f}")
    ...     print(f"权重: {result.weights}")
    
    Notes
    -----
    Ledoit-Wolf压缩协方差估计器通过将样本协方差向结构化目标（通常是单位矩阵的倍数）
    收缩来减少估计误差，特别适合于资产数量较多而样本量有限的情况。
    
    约束条件:
    - 单只资产权重 ∈ [min_weight, max_weight]
    - 所有权重之和 = 1 (全仓)
    """
    # 检查依赖
    if not PYPFOPT_AVAILABLE:
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message="pypfopt未安装，请运行: pip install pypfopt"
        )
    
    # 输入验证
    if prices.empty:
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message="价格数据为空"
        )
    
    n_assets = len(prices.columns)
    if n_assets < 2:
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message="至少需要2个资产进行优化"
        )
    
    # 转换mu为Series（如果是ndarray）
    if isinstance(mu, np.ndarray):
        if len(mu) != n_assets:
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                success=False,
                message=f"预期收益向量长度({len(mu)})与资产数量({n_assets})不匹配"
            )
        mu = pd.Series(mu, index=prices.columns)
    
    logger.info(
        f"开始组合优化: {n_assets}只资产, "
        f"max_weight={max_weight:.0%}, "
        f"cov_method={cov_method}"
    )
    
    try:
        # Step 1: 计算协方差矩阵
        cov_matrix = _calculate_covariance(prices, method=cov_method)
        
        # Step 2: 构建有效前沿优化器
        ef = EfficientFrontier(
            expected_returns=mu,
            cov_matrix=cov_matrix,
            weight_bounds=(min_weight, max_weight),  # 权重约束
        )
        
        # Step 3: 添加L2正则化（可选，防止极端权重）
        if l2_gamma > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
        
        # Step 4: 求解最大夏普比率
        raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        
        # Step 5: 清理权重（移除过小的权重）
        cleaned_weights = ef.clean_weights(cutoff=1e-4, rounding=4)
        
        # Step 6: 计算组合绩效指标
        expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
            verbose=False,
            risk_free_rate=risk_free_rate
        )
        
        # 验证权重和为1
        weight_sum = sum(cleaned_weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-3):
            logger.warning(f"权重和为 {weight_sum:.4f}，进行归一化")
            cleaned_weights = {
                k: v / weight_sum for k, v in cleaned_weights.items()
            }
        
        # 过滤掉零权重资产
        non_zero_weights = {k: v for k, v in cleaned_weights.items() if v > 1e-6}
        
        logger.info(
            f"优化成功: "
            f"预期收益={expected_return:.2%}, "
            f"波动率={volatility:.2%}, "
            f"夏普比率={sharpe_ratio:.2f}, "
            f"持仓数量={len(non_zero_weights)}"
        )
        
        return OptimizationResult(
            weights=cleaned_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            success=True,
            message="优化成功"
        )
        
    except OptimizationError as e:
        error_msg = f"优化求解失败: {str(e)}"
        logger.error(error_msg)
        
        # 尝试降级策略：等权重
        return _fallback_equal_weight(
            prices=prices,
            mu=mu,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            error_message=error_msg
        )
        
    except Exception as e:
        error_msg = f"优化过程异常: {str(e)}"
        logger.error(error_msg)
        
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message=error_msg
        )


def _calculate_covariance(
    prices: pd.DataFrame,
    method: str = "ledoit_wolf"
) -> pd.DataFrame:
    """
    计算协方差矩阵
    
    Parameters
    ----------
    prices : pd.DataFrame
        历史价格数据
    method : str
        协方差估计方法
    
    Returns
    -------
    pd.DataFrame
        协方差矩阵
    """
    if method == "ledoit_wolf":
        # Ledoit-Wolf 压缩协方差估计
        # 将样本协方差向结构化目标收缩，减少估计误差
        cov_matrix = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        logger.debug("使用Ledoit-Wolf压缩协方差估计")
        
    elif method == "sample":
        # 样本协方差
        cov_matrix = risk_models.sample_cov(prices)
        logger.debug("使用样本协方差")
        
    elif method == "semicovariance":
        # 半协方差（只考虑下行风险）
        cov_matrix = risk_models.semicovariance(prices)
        logger.debug("使用半协方差（下行风险）")
        
    elif method == "exp_cov":
        # 指数加权协方差
        cov_matrix = risk_models.exp_cov(prices)
        logger.debug("使用指数加权协方差")
        
    else:
        raise ValueError(f"不支持的协方差方法: {method}")
    
    return cov_matrix


def _fallback_equal_weight(
    prices: pd.DataFrame,
    mu: pd.Series,
    risk_free_rate: float,
    max_weight: float,
    error_message: str
) -> OptimizationResult:
    """
    降级到等权重策略
    
    当优化失败时，返回等权重组合作为备选方案。
    
    Parameters
    ----------
    prices : pd.DataFrame
        价格数据
    mu : pd.Series
        预期收益
    risk_free_rate : float
        无风险利率
    max_weight : float
        最大权重约束
    error_message : str
        原始错误信息
    
    Returns
    -------
    OptimizationResult
        等权重组合结果
    """
    logger.warning("优化失败，降级到等权重策略")
    
    n_assets = len(prices.columns)
    
    # 考虑max_weight约束计算等权重
    if max_weight * n_assets >= 1.0:
        # 可以使用等权重
        weight = 1.0 / n_assets
    else:
        # max_weight约束导致无法全仓，按max_weight分配
        weight = max_weight
    
    weights = {col: weight for col in prices.columns}
    
    # 归一化（确保权重和为1）
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}
    
    # 计算等权重组合的绩效
    returns = prices.pct_change().dropna()
    portfolio_return = (returns.mean() * pd.Series(weights)).sum() * 252
    portfolio_vol = np.sqrt(
        np.dot(
            list(weights.values()),
            np.dot(returns.cov() * 252, list(weights.values()))
        )
    )
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    
    return OptimizationResult(
        weights=weights,
        expected_return=portfolio_return,
        volatility=portfolio_vol,
        sharpe_ratio=sharpe,
        success=False,  # 标记为降级结果
        message=f"优化失败，降级到等权重。原因: {error_message}"
    )


def optimize_min_volatility(
    prices: pd.DataFrame,
    max_weight: float = 0.10,
    min_weight: float = 0.0,
    cov_method: str = "ledoit_wolf"
) -> OptimizationResult:
    """
    最小波动率组合优化
    
    求解全局最小方差组合。
    
    Parameters
    ----------
    prices : pd.DataFrame
        历史价格数据
    max_weight : float, optional
        单只资产最大权重，默认0.10
    min_weight : float, optional
        单只资产最小权重，默认0.0
    cov_method : str, optional
        协方差估计方法
    
    Returns
    -------
    OptimizationResult
        优化结果
    """
    if not PYPFOPT_AVAILABLE:
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message="pypfopt未安装"
        )
    
    try:
        # 计算协方差矩阵
        cov_matrix = _calculate_covariance(prices, method=cov_method)
        
        # 使用历史均值作为预期收益（仅用于计算绩效指标）
        mu = expected_returns.mean_historical_return(prices)
        
        # 构建优化器
        ef = EfficientFrontier(
            expected_returns=mu,
            cov_matrix=cov_matrix,
            weight_bounds=(min_weight, max_weight),
        )
        
        # 最小波动率
        ef.min_volatility()
        cleaned_weights = ef.clean_weights(cutoff=1e-4, rounding=4)
        
        # 计算绩效
        expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
            verbose=False
        )
        
        logger.info(f"最小波动率优化成功: 波动率={volatility:.2%}")
        
        return OptimizationResult(
            weights=cleaned_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            success=True,
            message="最小波动率优化成功"
        )
        
    except Exception as e:
        logger.error(f"最小波动率优化失败: {e}")
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message=str(e)
        )


def optimize_risk_parity(
    prices: pd.DataFrame,
    cov_method: str = "ledoit_wolf"
) -> OptimizationResult:
    """
    风险平价组合优化
    
    使每个资产对组合总风险的贡献相等。
    
    Parameters
    ----------
    prices : pd.DataFrame
        历史价格数据
    cov_method : str, optional
        协方差估计方法
    
    Returns
    -------
    OptimizationResult
        优化结果
    
    Notes
    -----
    风险平价策略不需要预期收益估计，只依赖协方差矩阵。
    """
    if not PYPFOPT_AVAILABLE:
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message="pypfopt未安装"
        )
    
    try:
        from pypfopt import HRPOpt
        
        # 计算收益率
        returns = prices.pct_change().dropna()
        
        # HRP (Hierarchical Risk Parity) 分层风险平价
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        cleaned_weights = hrp.clean_weights(cutoff=1e-4, rounding=4)
        
        # 计算绩效
        expected_return, volatility, sharpe_ratio = hrp.portfolio_performance(
            verbose=False
        )
        
        logger.info(f"风险平价优化成功: {len([w for w in cleaned_weights.values() if w > 0])}只资产")
        
        return OptimizationResult(
            weights=cleaned_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            success=True,
            message="风险平价优化成功"
        )
        
    except Exception as e:
        logger.error(f"风险平价优化失败: {e}")
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            success=False,
            message=str(e)
        )


def calculate_expected_returns(
    prices: pd.DataFrame,
    method: str = "mean_historical",
    **kwargs
) -> pd.Series:
    """
    计算预期收益
    
    Parameters
    ----------
    prices : pd.DataFrame
        历史价格数据
    method : str, optional
        计算方法，可选:
        - 'mean_historical': 历史均值（默认）
        - 'ema_historical': 指数加权历史均值
        - 'capm': CAPM模型
    **kwargs
        额外参数传递给具体方法
    
    Returns
    -------
    pd.Series
        预期收益向量
    
    Examples
    --------
    >>> mu = calculate_expected_returns(prices, method='ema_historical', span=252)
    """
    if not PYPFOPT_AVAILABLE:
        # 简单实现：年化历史均值
        returns = prices.pct_change().dropna()
        return returns.mean() * 252
    
    if method == "mean_historical":
        return expected_returns.mean_historical_return(prices, **kwargs)
    elif method == "ema_historical":
        return expected_returns.ema_historical_return(prices, **kwargs)
    elif method == "capm":
        return expected_returns.capm_return(prices, **kwargs)
    else:
        raise ValueError(f"不支持的方法: {method}")


class PortfolioOptimizer:
    """
    投资组合优化器类
    
    封装多种优化策略，提供统一接口。
    
    Attributes
    ----------
    prices : pd.DataFrame
        历史价格数据
    max_weight : float
        单只资产最大权重
    risk_free_rate : float
        无风险利率
    
    Examples
    --------
    >>> optimizer = PortfolioOptimizer(prices, max_weight=0.10)
    >>> result = optimizer.optimize(method='max_sharpe')
    >>> print(result.weights)
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        max_weight: float = 0.10,
        min_weight: float = 0.0,
        risk_free_rate: float = 0.03,
        cov_method: str = "ledoit_wolf"
    ) -> None:
        """
        初始化优化器
        
        Parameters
        ----------
        prices : pd.DataFrame
            历史价格数据
        max_weight : float, optional
            单只资产最大权重
        min_weight : float, optional
            单只资产最小权重
        risk_free_rate : float, optional
            无风险利率
        cov_method : str, optional
            协方差估计方法
        """
        self.prices = prices
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free_rate = risk_free_rate
        self.cov_method = cov_method
        
        # 预计算
        self._mu = None
        self._cov = None
        
        logger.info(
            f"PortfolioOptimizer初始化: "
            f"{len(prices.columns)}只资产, "
            f"max_weight={max_weight:.0%}"
        )
    
    @property
    def expected_returns(self) -> pd.Series:
        """获取预期收益向量"""
        if self._mu is None:
            self._mu = calculate_expected_returns(self.prices)
        return self._mu
    
    @expected_returns.setter
    def expected_returns(self, value: pd.Series) -> None:
        """设置预期收益向量"""
        self._mu = value
    
    def optimize(
        self,
        method: str = "max_sharpe",
        mu: Optional[pd.Series] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        执行组合优化
        
        Parameters
        ----------
        method : str, optional
            优化方法，可选:
            - 'max_sharpe': 最大夏普比率（默认）
            - 'min_volatility': 最小波动率
            - 'risk_parity': 风险平价
        mu : Optional[pd.Series]
            自定义预期收益向量，如果为None则使用历史均值
        **kwargs
            传递给具体优化方法的参数
        
        Returns
        -------
        OptimizationResult
            优化结果
        """
        # 使用自定义或默认预期收益
        if mu is not None:
            self._mu = mu
        
        if method == "max_sharpe":
            return optimize_max_sharpe(
                mu=self.expected_returns,
                prices=self.prices,
                max_weight=self.max_weight,
                min_weight=self.min_weight,
                risk_free_rate=self.risk_free_rate,
                cov_method=self.cov_method,
                **kwargs
            )
        elif method == "min_volatility":
            return optimize_min_volatility(
                prices=self.prices,
                max_weight=self.max_weight,
                min_weight=self.min_weight,
                cov_method=self.cov_method
            )
        elif method == "risk_parity":
            return optimize_risk_parity(
                prices=self.prices,
                cov_method=self.cov_method
            )
        else:
            raise ValueError(f"不支持的优化方法: {method}")
    
    def efficient_frontier(
        self,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        计算有效前沿
        
        Parameters
        ----------
        n_points : int, optional
            有效前沿上的点数
        
        Returns
        -------
        pd.DataFrame
            有效前沿数据，包含收益率、波动率和夏普比率
        """
        if not PYPFOPT_AVAILABLE:
            logger.warning("pypfopt未安装，无法计算有效前沿")
            return pd.DataFrame()
        
        try:
            cov_matrix = _calculate_covariance(self.prices, self.cov_method)
            
            results = []
            
            # 计算最小和最大可达收益
            ef_min = EfficientFrontier(
                self.expected_returns, 
                cov_matrix,
                weight_bounds=(self.min_weight, self.max_weight)
            )
            ef_min.min_volatility()
            min_ret, _, _ = ef_min.portfolio_performance()
            
            ef_max = EfficientFrontier(
                self.expected_returns, 
                cov_matrix,
                weight_bounds=(self.min_weight, self.max_weight)
            )
            ef_max.max_sharpe()
            max_ret, _, _ = ef_max.portfolio_performance()
            
            # 在收益范围内采样
            target_returns = np.linspace(min_ret, max_ret * 1.2, n_points)
            
            for target in target_returns:
                try:
                    ef = EfficientFrontier(
                        self.expected_returns,
                        cov_matrix,
                        weight_bounds=(self.min_weight, self.max_weight)
                    )
                    ef.efficient_return(target)
                    ret, vol, sharpe = ef.portfolio_performance(
                        risk_free_rate=self.risk_free_rate
                    )
                    results.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe': sharpe
                    })
                except Exception:
                    continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"计算有效前沿失败: {e}")
            return pd.DataFrame()
    
    def plot_efficient_frontier(
        self,
        show_assets: bool = True,
        show_optimal: bool = True,
        figsize: tuple = (10, 7),
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制有效前沿图
        
        Parameters
        ----------
        show_assets : bool, optional
            是否显示单个资产
        show_optimal : bool, optional
            是否标注最优组合
        figsize : tuple, optional
            图表大小
        save_path : Optional[str]
            保存路径
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("请安装matplotlib")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 计算有效前沿
        frontier = self.efficient_frontier()
        if frontier.empty:
            logger.warning("无法绘制有效前沿")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制有效前沿
        ax.plot(
            frontier['volatility'] * 100,
            frontier['return'] * 100,
            'b-',
            linewidth=2,
            label='有效前沿'
        )
        
        # 绘制单个资产
        if show_assets:
            returns = self.prices.pct_change().dropna()
            asset_returns = returns.mean() * 252 * 100
            asset_vols = returns.std() * np.sqrt(252) * 100
            
            ax.scatter(
                asset_vols,
                asset_returns,
                c='gray',
                s=30,
                alpha=0.6,
                label='单个资产'
            )
        
        # 标注最优组合
        if show_optimal:
            result = self.optimize(method='max_sharpe')
            if result.success:
                ax.scatter(
                    result.volatility * 100,
                    result.expected_return * 100,
                    c='red',
                    s=200,
                    marker='*',
                    label=f'最大夏普 (SR={result.sharpe_ratio:.2f})'
                )
        
        ax.set_xlabel('波动率 (%)', fontsize=12)
        ax.set_ylabel('预期收益率 (%)', fontsize=12)
        ax.set_title('投资组合有效前沿', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        else:
            plt.show()

