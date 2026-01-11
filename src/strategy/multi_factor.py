"""
多因子选股策略模块

本模块实现 MultiFactorStrategy，基于价值、质量、动量和市值因子的综合打分进行选股。
支持情绪分析加成、自适应权重、拥挤度轮动等高级功能。
"""
from typing import Optional, List, Dict, Any, Tuple, Union
import logging
import pandas as pd
import numpy as np

from .base import BaseStrategy, SignalType, TradeSignal

# LLMCircuitBreakerError fallback
try:
    from src.llm_client import LLMCircuitBreakerError
except ImportError:
    try:
        from llm_client import LLMCircuitBreakerError
    except ImportError:
        class LLMCircuitBreakerError(RuntimeError):
            """LLM 熔断器触发异常（回退定义）"""
            pass

logger = logging.getLogger(__name__)


class MultiFactorStrategy(BaseStrategy):
    """
    多因子选股策略
    
    基于价值、质量和动量因子的综合打分进行选股。
    
    打分公式 (默认): 
    Final_Score = Quality_Weight * Quality_Z + Momentum_Weight * Momentum_Z + Size_Weight * Size_Z
    (+ Sentiment_Score * Sentiment_Weight if enabled)
    
    Parameters
    ----------
    name : str
        策略名称
    config : Optional[Dict[str, Any]]
        配置参数，包含：
        - value_weight: 价值因子权重 (默认 0.0)
        - quality_weight: 质量因子权重 (默认 0.3)
        - momentum_weight: 动量因子权重 (默认 0.7)
        - size_weight: 市值因子权重 (默认 0.0)
        - top_n: 选取股票数量
        - momentum_col: 动量因子列名 (默认 sharpe_20_zscore)
    
    Attributes
    ----------
    value_weight : float
        价值因子权重
    quality_weight : float
        质量因子权重
    momentum_weight : float
        动量因子权重
    top_n : int
        选取股票数量
    min_listing_days : int
        最小上市天数
    
    Examples
    --------
    >>> config = {
    ...     "value_weight": 0.4,
    ...     "quality_weight": 0.4,
    ...     "momentum_weight": 0.2,
    ...     "top_n": 30
    ... }
    >>> strategy = MultiFactorStrategy("Multi-Factor", config)
    >>> target_positions = strategy.generate_target_positions(factor_data)
    """
    
    # ===== 小资金实盘优化：流动性与可交易性过滤常量 =====
    MIN_DAILY_AMOUNT = 50_000_000  # 5000万
    LIMIT_UP_THRESHOLD = 0.095
    LIMIT_DOWN_THRESHOLD = -0.095
    ST_KEYWORDS = ('ST', '*ST', '退', 'S', 'PT')
    TURNOVER_OVERHEAT_THRESHOLD = 5.0
    
    def __init__(
        self,
        name: str = "Multi-Factor Strategy",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """初始化多因子策略"""
        super().__init__(name, config)
        
        # 因子权重配置
        self.value_weight: float = self.config.get("value_weight", 0.0)
        self.quality_weight: float = self.config.get("quality_weight", 0.3)
        self.momentum_weight: float = self.config.get("momentum_weight", 0.7)
        self.size_weight: float = self.config.get("size_weight", 0.0)
        self.sentiment_weight: float = self.config.get("sentiment_weight", 0.2)
        
        # 选股参数配置
        self.top_n: int = self.config.get("top_n", 5)
        MAX_POSITIONS_LIMIT = 8
        if self.top_n > MAX_POSITIONS_LIMIT:
            logger.warning(f"top_n ({self.top_n}) 超限，调整为 {MAX_POSITIONS_LIMIT}")
            self.top_n = MAX_POSITIONS_LIMIT

        self.min_listing_days: int = self.config.get("min_listing_days", 126)
        
        # 板块过滤配置
        self._exclude_chinext: bool = self.config.get("exclude_chinext", False)
        self._exclude_star: bool = self.config.get("exclude_star", False)
        
        # 因子列名配置
        self.value_col: str = self.config.get("value_col", "value_zscore")
        self.quality_col: str = self.config.get("quality_col", "turnover_5d_zscore")
        self.momentum_col: str = self.config.get("momentum_col", "sharpe_20_zscore")
        self.size_col: str = self.config.get("size_col", "small_cap_zscore")
        
        # 日期和股票列名配置
        self.date_col: str = self.config.get("date_col", "date")
        self.stock_col: str = self.config.get("stock_col", "stock_code")
        
        # 调仓频率配置
        self.rebalance_frequency: str = self.config.get("rebalance_frequency", "monthly")
        self.rebalance_buffer: float = self.config.get("rebalance_buffer", 0.05)
        self.holding_bonus: float = self.config.get("holding_bonus", 0.0)
        
        # 大盘风控参数
        market_risk_config = self.config.get("market_risk", {})
        self._market_risk_enabled: bool = market_risk_config.get("enabled", True)
        self._market_risk_ma_period: int = market_risk_config.get("ma_period", 60)
        self._market_risk_drop_threshold: float = market_risk_config.get("drop_threshold", 0.05)
        self._market_risk_drop_lookback: int = market_risk_config.get("drop_lookback", 20)
        
        if self.rebalance_frequency not in ("monthly", "weekly"):
            self.rebalance_frequency = "monthly"
        
        # 验证权重
        weight_sum = self.value_weight + self.quality_weight + self.momentum_weight + self.size_weight
        if abs(weight_sum - 1.0) > 1e-6:
            if not any(w < 0 for w in [self.value_weight, self.quality_weight, self.momentum_weight, self.size_weight]):
                logger.warning(f"因子权重之和为 {weight_sum}，建议为 1.0")
        
        # LLM 情绪分析配置
        self._llm_config = self.config.get("llm", {})
        self._enable_sentiment_filter: bool = self._llm_config.get("enable_sentiment_filter", False)
        self._sentiment_threshold: float = self._llm_config.get("sentiment_threshold", -0.5)
        self._min_confidence: float = self._llm_config.get("min_confidence", 0.7)
        self._sentiment_buffer_multiplier: int = self._llm_config.get("sentiment_buffer_multiplier", 3)
        self._sentiment_engine = None
        
        # 过热熔断阈值
        self.turnover_threshold: float = self.config.get("turnover_threshold", 50.0)
        self.volatility_threshold: float = self.config.get("volatility_threshold", 5.0)

        # 可配置过滤阈值（偏进攻时可放宽/关闭）
        self.min_daily_amount: float = float(self.config.get("min_daily_amount", self.MIN_DAILY_AMOUNT))
        self.max_price: Optional[float] = self.config.get("max_price", 100.0)
        self.max_rsi: Optional[float] = self.config.get("max_rsi", 80.0)
        self.min_efficiency: Optional[float] = self.config.get("min_efficiency", 0.3)
        self.overheat_check_col: str = self.config.get("overheat_check_col", self.quality_col)
        self.min_circ_mv: Optional[float] = self.config.get("min_circ_mv", None)

        # 趋势过滤（提高“看起来更合理”的持仓：高波动但要求趋势向上）
        trend_cfg = self.config.get("trend_filter", {})
        self._trend_filter_enabled: bool = bool(trend_cfg.get("enabled", False))
        self._require_positive_return_20: bool = bool(trend_cfg.get("require_positive_return_20", True))

        # 行业分散与黑名单（抑制集中押注）
        industry_cfg = self.config.get("industry_constraints", {})
        self._industry_constraints_enabled: bool = bool(industry_cfg.get("enabled", False))
        self._industry_col: str = str(industry_cfg.get("industry_col", "industry"))
        self._max_per_industry: int = int(industry_cfg.get("max_per_industry", 2))
        self._excluded_industries: List[str] = [str(x) for x in industry_cfg.get("excluded_industries", [])]
        
        # 拥挤度板块轮动配置
        self._enable_crowding_rotation: bool = self.config.get("enable_crowding_rotation", False)
        self._crowding_exit_threshold: float = self.config.get("crowding_exit_threshold", 0.95)
        self._crowding_entry_threshold: float = self.config.get("crowding_entry_threshold", 0.50)
        self._crowding_calculator = None
        self._crowding_cache: Dict[str, pd.DataFrame] = {}
        
        # Alpha 因子开关
        self._enable_alpha_factors: bool = self.config.get("enable_alpha_factors", True)
        
        # 市场状态自适应权重配置
        self._market_regime_config = self.config.get("market_regime", {})
        self._enable_adaptive_weights: bool = self._market_regime_config.get("enabled", False)
        self._current_market_regime: str = "sideways"
        self._current_market_volatility: Optional[float] = None
        self._current_position_scale: float = 1.0

        # 因子方向（用于负 IC 自动反向）
        # 约定：+1 表示“因子越大越好”；-1 表示“因子越小越好（反向使用）”
        self._factor_directions: Dict[str, int] = {
            'value': 1,
            'quality': 1,
            'momentum': 1,
            'size': 1,
        }
        
        # 初始化情绪分析引擎
        if self._enable_sentiment_filter:
            try:
                from src.features import SentimentEngine
                self._sentiment_engine = SentimentEngine(self._llm_config)
                logger.info(f"情绪分析过滤已启用: threshold={self._sentiment_threshold}")
            except ImportError:
                logger.warning("无法导入 SentimentEngine，情绪分析过滤未启用")
                self._enable_sentiment_filter = False
            except Exception as e:
                logger.warning(f"初始化 SentimentEngine 失败: {e}")
                self._enable_sentiment_filter = False
        
        logger.info(
            f"多因子策略初始化: 价值权重={self.value_weight}, "
            f"质量权重={self.quality_weight}, 动量权重={self.momentum_weight}, "
            f"市值权重={self.size_weight}, Top N={self.top_n}"
        )
    
    # ==================== 市场状态识别与自适应权重 ====================
    
    def identify_market_regime(
        self,
        index_data: pd.DataFrame,
        date: Optional[pd.Timestamp] = None
    ) -> str:
        """识别市场状态（牛市/熊市/震荡市）"""
        if index_data.empty:
            return 'sideways'
        
        try:
            if not isinstance(index_data.index, pd.DatetimeIndex):
                if 'date' in index_data.columns:
                    index_data = index_data.set_index('date')
                elif 'trade_date' in index_data.columns:
                    index_data = index_data.set_index('trade_date')
            index_data = index_data.sort_index()
            
            ma_period = self._market_regime_config.get('trend_ma_period', 60)
            vol_period = self._market_regime_config.get('volatility_period', 20)
            high_vol_threshold = self._market_regime_config.get('high_volatility_threshold', 0.25)
            low_vol_threshold = self._market_regime_config.get('low_volatility_threshold', 0.15)
            
            if date is not None:
                valid_dates = index_data.index[index_data.index <= date]
                if len(valid_dates) == 0:
                    return 'sideways'
                index_data = index_data.loc[:valid_dates[-1]]
            
            latest_close = index_data['close'].iloc[-1]
            ma_value = index_data['close'].rolling(ma_period, min_periods=ma_period//2).mean().iloc[-1]
            
            if pd.isna(ma_value):
                return 'sideways'
            
            trend_up = latest_close > ma_value
            
            returns = index_data['close'].pct_change()
            volatility = returns.rolling(vol_period, min_periods=vol_period//2).std().iloc[-1] * np.sqrt(252)
            self._current_market_volatility = float(volatility) if pd.notna(volatility) else None
            
            if pd.isna(volatility):
                return 'sideways'
            
            if trend_up and volatility < high_vol_threshold:
                regime = 'bull'
            elif not trend_up and volatility > low_vol_threshold:
                regime = 'bear'
            else:
                regime = 'sideways'
            
            self._current_market_regime = regime
            logger.info(f"市场状态识别: {regime.upper()} (趋势={'上升' if trend_up else '下降'}, 波动率={volatility:.1%})")
            return regime
            
        except Exception as e:
            logger.warning(f"市场状态识别失败: {e}")
            return 'sideways'
    
    def get_adaptive_weights(self, market_regime: Optional[str] = None) -> Dict[str, float]:
        """根据市场状态返回因子权重"""
        if market_regime is None:
            market_regime = self._current_market_regime

        # 允许从配置覆盖（推荐在 strategy_config.yaml 中显式配置，避免硬编码与当前行情不一致）
        user_regime_weights = self._market_regime_config.get("regime_weights")
        if isinstance(user_regime_weights, dict) and user_regime_weights:
            fallback = user_regime_weights.get("sideways", {})
            return user_regime_weights.get(market_regime, fallback)

        # 默认值（保守基线，避免过度进攻）
        regime_weights = {
            'bull': {
                'momentum_weight': 0.20,
                'sentiment_weight': 0.15,
                'size_weight': -0.20,
                'quality_weight': 0.55,
                'value_weight': 0.10
            },
            'sideways': {
                'momentum_weight': 0.15,
                'sentiment_weight': 0.10,
                'size_weight': -0.10,
                'quality_weight': 0.45,
                'value_weight': 0.30
            },
            'bear': {
                'momentum_weight': 0.05,
                'sentiment_weight': 0.00,
                'size_weight': -0.05,
                'quality_weight': 0.35,
                'value_weight': 0.65
            }
        }
        return regime_weights.get(market_regime, regime_weights['sideways'])
    
    def apply_adaptive_weights(self, index_data: pd.DataFrame, date: Optional[pd.Timestamp] = None) -> None:
        """应用自适应权重"""
        if not self._enable_adaptive_weights:
            return
        
        regime = self.identify_market_regime(index_data, date)
        weights = self.get_adaptive_weights(regime)
        
        self.momentum_weight = weights['momentum_weight']
        self.sentiment_weight = weights['sentiment_weight']
        self.size_weight = weights['size_weight']
        self.quality_weight = weights['quality_weight']
        self.value_weight = weights['value_weight']

        # 动态仓位系数（高波动自动降仓）
        pos_scale_cfg = self._market_regime_config.get("position_scale", {})
        base_scale = float(pos_scale_cfg.get(regime, 1.0)) if isinstance(pos_scale_cfg, dict) else 1.0

        hv_scale = 1.0
        if self._current_market_volatility is not None:
            high_vol_threshold = float(self._market_regime_config.get('high_volatility_threshold', 0.25))
            if self._current_market_volatility >= high_vol_threshold:
                hv_scale = float(self._market_regime_config.get("high_volatility_position_scale", 0.75))

        self._current_position_scale = float(np.clip(base_scale * hv_scale, 0.0, 1.0))

        logger.info(f"自适应权重已应用 ({regime}), position_scale={self._current_position_scale:.0%}")

    def get_position_scale(self) -> float:
        """
        获取当前动态仓位系数（0~1）。

        Returns
        -------
        float
            当前仓位系数，值越小表示越保守（留更多现金）。
        """
        return float(self._current_position_scale)

    def apply_factor_direction_from_ic(
        self,
        ic_results: pd.DataFrame,
        abs_ic_threshold: float = 0.02,
        ir_threshold: float = 0.3,
        positive_ratio_threshold: float = 0.55
    ) -> Dict[str, int]:
        """
        基于 IC 结果自动校准因子方向（负 IC 反向使用）。

        Parameters
        ----------
        ic_results : pd.DataFrame
            因子 IC 统计结果（来自 calculate_factor_ic），至少包含:
            - factor, ic_mean, ic_ir, ic_positive_ratio
        abs_ic_threshold : float
            触发方向校准的 |IC| 阈值。
        ir_threshold : float
            触发方向校准的稳定性阈值（|IC_IR|）。
        positive_ratio_threshold : float
            方向稳定性辅助阈值（正 IC 占比）。

        Returns
        -------
        Dict[str, int]
            实际被更新的方向映射（key 为组件名 value/quality/momentum/size）。
        """
        if ic_results is None or ic_results.empty:
            return {}

        factor_to_component: Dict[str, str] = {
            self.value_col: 'value',
            self.quality_col: 'quality',
            self.momentum_col: 'momentum',
            self.size_col: 'size',
        }

        updated: Dict[str, int] = {}
        for _, row in ic_results.iterrows():
            factor = str(row.get("factor", ""))
            if factor not in factor_to_component:
                continue

            ic_mean = float(row.get("ic_mean", 0.0))
            ic_ir = float(row.get("ic_ir", 0.0))
            pos_ratio = float(row.get("ic_positive_ratio", row.get("positive_ratio", 0.5)))

            if abs(ic_mean) < abs_ic_threshold or abs(ic_ir) < ir_threshold:
                continue

            component = factor_to_component[factor]
            # 使用 IC 符号作为方向：IC<0 => 反向
            direction = 1 if ic_mean >= 0 else -1

            # 用正 IC 占比做轻约束：若方向不稳定则不改
            if direction == 1 and pos_ratio < positive_ratio_threshold:
                continue
            if direction == -1 and pos_ratio > (1.0 - positive_ratio_threshold):
                continue

            old = self._factor_directions.get(component, 1)
            if old != direction:
                self._factor_directions[component] = direction
                updated[component] = direction
                logger.warning(f"因子方向校准: {component} ({factor}) direction {old} -> {direction}")

        return updated
    
    def apply_factor_circuit_breaker(
        self,
        ic_results: pd.DataFrame,
        ic_threshold: float = 0.01,
        ir_threshold: float = 0.3
    ) -> Dict[str, float]:
        """因子失效熔断：自动将失效因子权重降为 0"""
        if ic_results is None or ic_results.empty:
            return {}
        
        factor_weight_mapping = {
            'momentum_composite': 'momentum_weight', 'momentum': 'momentum_weight',
            'small_cap': 'size_weight', 'size': 'size_weight',
            'turnover_5d': 'quality_weight', 'quality_composite': 'quality_weight', 'quality': 'quality_weight',
            'value_composite': 'value_weight', 'value': 'value_weight',
            'sentiment': 'sentiment_weight',
        }
        
        breaker_triggered = {}
        
        for _, row in ic_results.iterrows():
            factor_name = row['factor']
            ic_mean = row.get('ic_mean', 0)
            ic_ir = row.get('ic_ir', 0)
            abs_ic = abs(ic_mean)
            abs_ir = abs(ic_ir)
            base_name = factor_name.replace('_zscore', '')
            
            weight_attr = None
            for prefix, attr in factor_weight_mapping.items():
                if base_name.startswith(prefix) or base_name == prefix:
                    weight_attr = attr
                    break
            
            if abs_ic < ic_threshold and abs_ir < ir_threshold:
                if weight_attr is not None and hasattr(self, weight_attr):
                    old_weight = getattr(self, weight_attr)
                    if old_weight > 0:
                        setattr(self, weight_attr, 0.0)
                        breaker_triggered[factor_name] = old_weight
                        logger.warning(f"因子熔断触发: {factor_name} 权重 {old_weight:.0%} -> 0%")
        
        if breaker_triggered:
            remaining = self.value_weight + self.quality_weight + self.momentum_weight + self.size_weight
            if remaining > 0 and remaining < 1.0:
                scale = 1.0 / remaining
                self.value_weight *= scale
                self.quality_weight *= scale
                self.momentum_weight *= scale
                self.size_weight *= scale
        
        return breaker_triggered
    
    def check_position_drift(
        self,
        current_positions: Dict[str, float],
        target_positions: Dict[str, float],
        max_drift: float = 0.15
    ) -> Tuple[bool, float, Dict[str, float]]:
        """检测持仓累计偏移"""
        current_total = sum(current_positions.values()) if current_positions else 0
        target_total = sum(target_positions.values()) if target_positions else 0
        
        if current_total == 0 and target_total == 0:
            return False, 0.0, {}
        if current_total == 0 or target_total == 0:
            return True, 1.0, {}
        
        all_stocks = set(current_positions.keys()) | set(target_positions.keys())
        drift_details = {}
        total_drift = 0.0
        
        for stock in all_stocks:
            current_weight = current_positions.get(stock, 0) / current_total
            target_weight = target_positions.get(stock, 0) / target_total
            drift = abs(current_weight - target_weight)
            drift_details[stock] = {'current_weight': current_weight, 'target_weight': target_weight, 'drift': drift}
            total_drift += drift
        
        normalized_drift = total_drift / 2
        force_rebalance = normalized_drift > max_drift
        
        if force_rebalance:
            logger.warning(f"持仓偏移超限: {normalized_drift:.1%} > {max_drift:.1%}")
        
        return force_rebalance, normalized_drift, drift_details
    
    def calculate_dynamic_slippage(
        self,
        order_value: float,
        daily_amount: float,
        avg_spread: float = 0.001,
        impact_coefficient: float = 0.1
    ) -> float:
        """计算动态滑点"""
        if daily_amount <= 0:
            return 0.005
        
        volume_ratio = order_value / daily_amount
        market_impact = impact_coefficient * np.sqrt(volume_ratio)
        total_slippage = avg_spread / 2 + market_impact
        max_slippage = self.config.get("slippage", {}).get("max_slippage", 0.02)
        return min(total_slippage, max_slippage)
    
    def estimate_trading_cost(
        self,
        order_value: float,
        daily_amount: float,
        is_buy: bool = True,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.001,
        min_commission: float = 5.0
    ) -> Dict[str, float]:
        """估算完整交易成本"""
        commission = max(order_value * commission_rate, min_commission)
        stamp = order_value * stamp_duty if not is_buy else 0
        
        slippage_config = self.config.get("slippage", {})
        if slippage_config.get("mode", "dynamic") == "dynamic":
            slippage_rate = self.calculate_dynamic_slippage(order_value, daily_amount)
        else:
            slippage_rate = slippage_config.get("fixed_slippage", 0.001)
        
        slippage_cost = order_value * slippage_rate
        total = commission + stamp + slippage_cost
        
        return {
            'commission': commission, 'stamp_duty': stamp, 'slippage': slippage_cost,
            'slippage_rate': slippage_rate, 'total': total,
            'total_rate': total / order_value if order_value > 0 else 0
        }
    
    def calculate_total_score(
        self,
        data: pd.DataFrame,
        sentiment_scores: Optional[pd.Series] = None,
        return_components: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, Dict[str, pd.Series]]]:
        """
        计算综合因子得分
        
        Parameters
        ----------
        data : pd.DataFrame
            当日横截面数据（建议已完成 Z-Score 标准化）
        sentiment_scores : Optional[pd.Series]
            情绪分数（index=stock_code），可选
        return_components : bool
            是否返回分项贡献（用于报告可解释性）
        
        Returns
        -------
        Union[pd.Series, Tuple[pd.Series, Dict[str, pd.Series]]]
            - return_components=False: total_score
            - return_components=True: (total_score, components)
        """
        total_score = pd.Series(0.0, index=data.index)
        score_components: Dict[str, pd.Series] = {}

        # 可选：用“绝对权重和”归一化，避免出现负权重时总权重缩小导致信号幅度偏低
        normalize_cfg = self.config.get("score_normalization", {})
        normalize_by_abs: bool = bool(normalize_cfg.get("normalize_by_abs_weight_sum", True))
        base_weights = np.array([self.value_weight, self.quality_weight, self.momentum_weight, self.size_weight], dtype=float)
        abs_sum = float(np.abs(base_weights).sum())
        scale = 1.0
        if normalize_by_abs and abs_sum > 1e-12:
            scale = 1.0 / abs_sum
        
        # 价值因子
        if self.value_col in data.columns and self.value_weight != 0:
            direction = self._factor_directions.get('value', 1)
            value_contribution = (self.value_weight * scale) * (direction * data[self.value_col].fillna(0))
            total_score += value_contribution
            score_components['value'] = value_contribution
        
        # 质量因子（含换手率过热惩罚）
        is_turnover_factor = 'turnover' in self.quality_col.lower()
        if self.quality_col in data.columns and self.quality_weight != 0:
            direction = self._factor_directions.get('quality', 1)
            raw_quality = direction * data[self.quality_col].fillna(0)
            if is_turnover_factor:
                TURNOVER_PENALTY_THRESHOLD = 3.5
                quality_score = np.where(
                    raw_quality > TURNOVER_PENALTY_THRESHOLD,
                    TURNOVER_PENALTY_THRESHOLD - (raw_quality - TURNOVER_PENALTY_THRESHOLD) * 0.5,
                    raw_quality
                )
            else:
                quality_score = raw_quality
            quality_contribution = (self.quality_weight * scale) * pd.Series(quality_score, index=data.index)
            total_score += quality_contribution
            score_components['quality'] = quality_contribution
        
        # 动量因子
        if self.momentum_col in data.columns and self.momentum_weight != 0:
            direction = self._factor_directions.get('momentum', 1)
            momentum_contribution = (self.momentum_weight * scale) * (direction * data[self.momentum_col].fillna(0))
            total_score += momentum_contribution
            score_components['momentum'] = momentum_contribution
        
        # 市值因子
        if self.size_col in data.columns and self.size_weight != 0:
            direction = self._factor_directions.get('size', 1)
            size_contribution = (self.size_weight * scale) * (direction * data[self.size_col].fillna(0))
            total_score += size_contribution
            score_components['size'] = size_contribution
        
        # 情绪因子加成
        if sentiment_scores is not None and self.sentiment_weight > 0:
            stock_col = self.stock_col if self.stock_col in data.columns else 'symbol'
            if stock_col in data.columns:
                aligned_sentiment = data[stock_col].map(sentiment_scores).fillna(0)
            else:
                if isinstance(data.index, pd.MultiIndex):
                    stock_codes = data.index.get_level_values(-1)
                else:
                    stock_codes = data.index
                aligned_sentiment = stock_codes.to_series().map(sentiment_scores).fillna(0)
                aligned_sentiment.index = data.index
            
            mean_sent = aligned_sentiment.mean()
            std_sent = aligned_sentiment.std()
            
            if std_sent > 1e-8:
                scaled_sentiment = (aligned_sentiment - mean_sent) / std_sent
            else:
                scaled_sentiment = aligned_sentiment * 3.0
            
            sentiment_contribution = self.sentiment_weight * scaled_sentiment
            total_score += sentiment_contribution
            score_components['sentiment'] = sentiment_contribution
        
        if return_components:
            return total_score, score_components
        return total_score
    
    def get_month_end_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """获取每月最后一个交易日"""
        dates_series = pd.Series(dates, index=dates)
        month_end_dates = dates_series.groupby([dates_series.index.year, dates_series.index.month]).last()
        return pd.DatetimeIndex(month_end_dates.values)
    
    def get_week_end_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """获取每周最后一个交易日"""
        dates_series = pd.Series(dates, index=dates)
        week_end_dates = dates_series.groupby(
            [dates_series.index.isocalendar().year, dates_series.index.isocalendar().week]
        ).last()
        return pd.DatetimeIndex(week_end_dates.values)
    
    def get_rebalance_dates(self, dates: pd.DatetimeIndex, frequency: str = "monthly") -> pd.DatetimeIndex:
        """根据频率获取调仓日期"""
        if frequency == "monthly":
            return self.get_month_end_dates(dates)
        elif frequency == "weekly":
            return self.get_week_end_dates(dates)
        else:
            raise ValueError(f"不支持的调仓频率: {frequency}")
    
    def filter_stocks(self, data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """根据条件过滤股票（小资金实盘优化）"""
        # 获取当日数据
        if self.date_col in data.columns:
            day_data = data[data[self.date_col] == date].copy()
        elif isinstance(data.index, pd.DatetimeIndex):
            day_data = data.loc[data.index == date].copy()
        elif isinstance(data.index, pd.MultiIndex):
            if date in data.index.get_level_values(0):
                day_data = data.loc[date].copy()
            else:
                day_data = pd.DataFrame()
        else:
            return pd.DataFrame()
        
        if day_data.empty:
            return day_data
        
        stock_col = self.stock_col if self.stock_col in day_data.columns else 'symbol'
        initial_count = len(day_data)
        filter_stats = {}
        
        # 过滤涨跌停
        if 'is_limit' in day_data.columns:
            before = len(day_data)
            day_data = day_data[~day_data['is_limit'].fillna(False)]
            filter_stats['is_limit'] = before - len(day_data)
        
        # 过滤一字涨停
        if 'high' in day_data.columns and 'low' in day_data.columns:
            pct_col = next((c for c in ['pct_change', 'pctChg', 'change_pct'] if c in day_data.columns), None)
            if pct_col:
                threshold = self.LIMIT_UP_THRESHOLD if day_data[pct_col].abs().max() < 1 else self.LIMIT_UP_THRESHOLD * 100
                is_one_word = (day_data['high'] == day_data['low']) & (day_data[pct_col] >= threshold)
                before = len(day_data)
                day_data = day_data[~is_one_word.fillna(False)]
                filter_stats['一字涨停'] = before - len(day_data)
        
        # 过滤流动性不足
        if 'amount' in day_data.columns:
            before = len(day_data)
            amount_values = day_data['amount'].copy()
            amount_in_yuan = amount_values * 10000 if amount_values.max() < 1_000_000 else amount_values
            day_data = day_data[amount_in_yuan >= float(self.min_daily_amount)]
            filter_stats['流动性不足'] = before - len(day_data)

        # 过滤微盘股（用流通市值下限控制“过于小盘”的暴露）
        if self.min_circ_mv is not None and 'circ_mv' in day_data.columns:
            try:
                before = len(day_data)
                circ_mv = pd.to_numeric(day_data['circ_mv'], errors='coerce')
                day_data = day_data[circ_mv >= float(self.min_circ_mv)]
                filter_stats['流通市值过小'] = before - len(day_data)
            except Exception:
                pass
        
        # 过滤ST股
        name_col = next((c for c in ['name', 'stock_name', '股票名称', 'sec_name'] if c in day_data.columns), None)
        if name_col:
            before = len(day_data)
            st_mask = day_data[name_col].astype(str).apply(lambda x: any(kw in x for kw in self.ST_KEYWORDS))
            day_data = day_data[~st_mask]
            filter_stats['ST/退市'] = before - len(day_data)
        
        # 过滤高价股
        price_col = 'close' if 'close' in day_data.columns else next((c for c in ['price', 'close_price'] if c in day_data.columns), None)
        if price_col and self.max_price is not None:
            before = len(day_data)
            day_data = day_data[day_data[price_col] <= float(self.max_price)]
            filter_stats['高价股'] = before - len(day_data)
        
        # 过滤次新股
        before = len(day_data)
        if 'days_listed' in day_data.columns:
            day_data = day_data[day_data['days_listed'] >= self.min_listing_days]
        elif 'listing_days' in day_data.columns:
            day_data = day_data[day_data['listing_days'] >= self.min_listing_days]
        elif 'list_date' in day_data.columns:
            list_dates = pd.to_datetime(day_data['list_date'])
            listing_days = (date - list_dates).dt.days
            day_data = day_data[listing_days >= self.min_listing_days]
        filter_stats['次新股'] = before - len(day_data)
        
        # 过滤创业板
        if self._exclude_chinext:
            if stock_col in day_data.columns:
                before = len(day_data)
                chinext_mask = day_data[stock_col].astype(str).str[:3].isin(['300', '301'])
                day_data = day_data[~chinext_mask]
                filter_stats['创业板'] = before - len(day_data)
        
        # 过滤科创板
        if self._exclude_star:
            if stock_col in day_data.columns:
                before = len(day_data)
                star_mask = day_data[stock_col].astype(str).str[:3] == '688'
                day_data = day_data[~star_mask]
                filter_stats['科创板'] = before - len(day_data)
        
        # 过滤RSI过热
        if 'rsi_20' in day_data.columns and self.max_rsi is not None:
            before = len(day_data)
            day_data = day_data[day_data['rsi_20'] <= float(self.max_rsi)]
            filter_stats['RSI过热'] = before - len(day_data)
        
        # 过滤震荡股
        if 'efficiency_20' in day_data.columns and self.min_efficiency is not None:
            before = len(day_data)
            day_data = day_data[day_data['efficiency_20'] >= float(self.min_efficiency)]
            filter_stats['震荡股'] = before - len(day_data)

        # 趋势过滤：要求 20 日收益为正（偏动量进攻）
        if self._trend_filter_enabled and self._require_positive_return_20 and 'return_20' in day_data.columns:
            before = len(day_data)
            day_data = day_data[pd.to_numeric(day_data['return_20'], errors='coerce') > 0]
            filter_stats['趋势过滤(return_20>0)'] = before - len(day_data)
        
        # 过热熔断
        check_col = self.overheat_check_col
        if check_col in day_data.columns:
            before = len(day_data)
            overheat_mask = day_data[check_col] > self.turnover_threshold
            if 'score' in day_data.columns or 'sentiment_score' in day_data.columns:
                sentiment_col = 'score' if 'score' in day_data.columns else 'sentiment_score'
                overheat_mask = overheat_mask & (day_data[sentiment_col].fillna(0) < 0.8)
            day_data = day_data[~overheat_mask]
            filter_stats['过热熔断'] = before - len(day_data)
        
        total_filtered = initial_count - len(day_data)
        if total_filtered > 0:
            filter_detail = ", ".join(f"{k}:{v}" for k, v in filter_stats.items() if v > 0)
            logger.debug(f"日期 {date.strftime('%Y-%m-%d')}: 过滤 {total_filtered} 只 ({filter_detail})")
        
        return day_data

    def _apply_industry_constraints(
        self,
        ranked_df: pd.DataFrame,
        stock_col: str,
        n: int
    ) -> List[str]:
        """
        对已按分数排序的候选集合应用行业分散/黑名单。

        Parameters
        ----------
        ranked_df : pd.DataFrame
            已按分数降序排列，且包含 stock_col 与 industry_col（可选）
        stock_col : str
            股票代码列名
        n : int
            目标数量

        Returns
        -------
        List[str]
            应用约束后的股票列表
        """
        if not self._industry_constraints_enabled:
            return ranked_df[stock_col].astype(str).tolist()[:n]

        if self._max_per_industry <= 0:
            return ranked_df[stock_col].astype(str).tolist()[:n]

        if self._industry_col not in ranked_df.columns:
            return ranked_df[stock_col].astype(str).tolist()[:n]

        counts: Dict[str, int] = {}
        selected: List[str] = []

        for _, row in ranked_df.iterrows():
            code = str(row.get(stock_col, "")).strip()
            if not code:
                continue

            ind = row.get(self._industry_col, None)
            industry = str(ind).strip() if ind is not None and str(ind).strip() != "" else "UNKNOWN"

            if industry in self._excluded_industries:
                continue

            cnt = counts.get(industry, 0)
            if cnt >= self._max_per_industry:
                continue

            selected.append(code)
            counts[industry] = cnt + 1
            if len(selected) >= n:
                break

        return selected
    
    def _apply_sentiment_filter(self, candidates: List[str], date: pd.Timestamp) -> List[str]:
        """对预选候选股票应用 LLM 情绪分析过滤（Fail-Closed 策略）"""
        if not candidates:
            return []
        
        if not self._enable_sentiment_filter or self._sentiment_engine is None:
            return candidates
        
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        VETO_THRESHOLD = -0.5
        
        try:
            sentiment_df = self._sentiment_engine.calculate_sentiment(candidates, date_str)
            
            if sentiment_df.empty:
                logger.critical(f"情绪分析返回空结果 ({date_str}), Fail-Closed")
                return []
            
            filtered_candidates: List[str] = []
            for _, row in sentiment_df.iterrows():
                stock_code = row["stock_code"]
                score = row["score"]
                confidence = row["confidence"]
                
                if score < VETO_THRESHOLD:
                    logger.warning(f"风控剔除: {stock_code} 情绪分 {score:.2f}")
                    continue
                
                if confidence < self._min_confidence:
                    continue
                
                filtered_candidates.append(stock_code)
            
            return filtered_candidates
        
        except LLMCircuitBreakerError:
            logger.critical("LLM Circuit Breaker Triggered!")
            raise
        
        except Exception as e:
            logger.critical(f"LLM 情绪分析失败 ({date_str}): {e}. Fail-Closed")
            return []
    
    def select_top_stocks(
        self,
        data: pd.DataFrame,
        n: Optional[int] = None,
        date: Optional[pd.Timestamp] = None,
        use_sentiment_scoring: bool = True
    ) -> List[str]:
        """两阶段选股：技术面初筛 + 情绪面加成"""
        n = n or self.top_n
        if data.empty:
            return []
        
        stock_col = self.stock_col if self.stock_col in data.columns else 'symbol'
        data = data.copy()
        
        # 数据预处理
        date_col_name = None
        if self.date_col in data.columns:
            date_col_name = self.date_col
        elif 'trade_date' in data.columns:
            date_col_name = 'trade_date'
        elif isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            date_col_name = 'index' if 'index' in data.columns else data.columns[0]
        elif isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()
            date_col_name = data.columns[0]
        
        if stock_col in data.columns and date_col_name is not None:
            data = data.sort_values(date_col_name, ascending=False)
            data = data.drop_duplicates(subset=[stock_col], keep='first')
        
        # 第一阶段：技术面初筛
        data['base_score'] = self.calculate_total_score(data, sentiment_scores=None)
        
        # 持股惯性加分
        if self.holding_bonus > 0 and 'is_holding' in data.columns:
            holding_mask = data['is_holding'] == True
            if holding_mask.sum() > 0:
                data.loc[holding_mask, 'base_score'] += self.holding_bonus
        
        valid_data = data.dropna(subset=['base_score'])
        if valid_data.empty:
            return []
        
        should_use_sentiment = (
            use_sentiment_scoring and self._enable_sentiment_filter 
            and self._sentiment_engine is not None and self.sentiment_weight > 0
        )
        
        if not should_use_sentiment:
            if stock_col not in valid_data.columns:
                # 兜底：无显式股票列时直接按 index 取
                if isinstance(valid_data.index, pd.MultiIndex):
                    top_stocks = valid_data.nlargest(n, 'base_score').index.get_level_values(-1).astype(str).tolist()
                else:
                    top_stocks = valid_data.nlargest(n, 'base_score').index.astype(str).tolist()
                return list(dict.fromkeys(top_stocks))[:n]

            ranked = valid_data.sort_values('base_score', ascending=False, kind='mergesort')
            ranked = ranked.drop_duplicates(subset=[stock_col], keep='first')
            top_stocks = self._apply_industry_constraints(ranked_df=ranked, stock_col=stock_col, n=n)
            return list(dict.fromkeys(top_stocks))[:n]
        
        # 第二阶段：情绪面加成
        buffer_n = n * self._sentiment_buffer_multiplier
        
        if stock_col not in valid_data.columns:
            if isinstance(valid_data.index, pd.MultiIndex):
                pre_candidates = valid_data.nlargest(buffer_n, 'base_score').index.get_level_values(-1).tolist()
            else:
                pre_candidates = valid_data.nlargest(buffer_n, 'base_score').index.tolist()
        else:
            pre_candidates = valid_data.nlargest(buffer_n, 'base_score')[stock_col].tolist()
        
        if date is None:
            if self.date_col in data.columns:
                date = pd.to_datetime(data[self.date_col]).max()
            elif isinstance(data.index, pd.DatetimeIndex):
                date = data.index.max()
            else:
                date = pd.Timestamp.now()
        
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        VETO_THRESHOLD = -0.5
        vetoed_stocks: List[str] = []
        sentiment_scores: Optional[pd.Series] = None
        
        try:
            sentiment_df = self._sentiment_engine.calculate_sentiment(pre_candidates, date_str)
            
            if not sentiment_df.empty:
                for _, row in sentiment_df.iterrows():
                    if row["score"] < VETO_THRESHOLD:
                        vetoed_stocks.append(row["stock_code"])
                
                pre_candidates = [s for s in pre_candidates if s not in vetoed_stocks]
                
                # 情绪分用于“加分项”：用置信度做衰减，且低置信度直接置零，避免噪声放大
                score_series = pd.Series(sentiment_df['score'].values, index=sentiment_df['stock_code'].values).astype(float)
                conf_series = pd.Series(
                    sentiment_df['confidence'].values, index=sentiment_df['stock_code'].values
                ).astype(float) if 'confidence' in sentiment_df.columns else pd.Series(1.0, index=score_series.index)

                conf_series = conf_series.clip(lower=0.0, upper=1.0)
                conf_series = conf_series.where(conf_series >= self._min_confidence, 0.0)
                sentiment_scores = (score_series.clip(lower=0.0) * conf_series).astype(float)
        
        except LLMCircuitBreakerError:
            raise
        except Exception as e:
            logger.warning(f"情绪分析失败 ({date_str}): {e}, 降级为纯技术面")
            sentiment_scores = None
        
        # 第三阶段：最终排名
        if stock_col in valid_data.columns:
            candidate_mask = valid_data[stock_col].isin(pre_candidates)
        else:
            if isinstance(valid_data.index, pd.MultiIndex):
                candidate_mask = valid_data.index.get_level_values(-1).isin(pre_candidates)
            else:
                candidate_mask = valid_data.index.isin(pre_candidates)
        
        candidate_data = valid_data[candidate_mask].copy()
        candidate_data['final_score'] = self.calculate_total_score(candidate_data, sentiment_scores=sentiment_scores)
        
        if stock_col not in candidate_data.columns:
            if isinstance(candidate_data.index, pd.MultiIndex):
                top_stocks = candidate_data.nlargest(n, 'final_score').index.get_level_values(-1).astype(str).tolist()
            else:
                top_stocks = candidate_data.nlargest(n, 'final_score').index.astype(str).tolist()
            return list(dict.fromkeys(top_stocks))[:n]

        ranked = candidate_data.sort_values('final_score', ascending=False, kind='mergesort')
        ranked = ranked.drop_duplicates(subset=[stock_col], keep='first')
        top_stocks = self._apply_industry_constraints(ranked_df=ranked, stock_col=stock_col, n=n)
        return list(dict.fromkeys(top_stocks))[:n]
    
    def generate_target_positions(
        self,
        data: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """生成目标持仓矩阵"""
        # 确定日期列
        if self.date_col in data.columns:
            dates_array = pd.to_datetime(data[self.date_col].unique())
        elif isinstance(data.index, pd.DatetimeIndex):
            dates_array = data.index.unique()
        elif isinstance(data.index, pd.MultiIndex):
            dates_array = data.index.get_level_values(0).unique()
        else:
            raise ValueError("无法确定日期列")
        
        # 确定股票列
        stock_col = self.stock_col if self.stock_col in data.columns else 'symbol'
        if stock_col in data.columns:
            all_stocks = data[stock_col].unique()
        elif isinstance(data.index, pd.MultiIndex):
            all_stocks = data.index.get_level_values(-1).unique()
        else:
            raise ValueError("无法确定股票代码列")
        
        all_dates = pd.DatetimeIndex(sorted(dates_array))
        if start_date is not None:
            all_dates = all_dates[all_dates >= start_date]
        if end_date is not None:
            all_dates = all_dates[all_dates <= end_date]
        
        if all_dates.empty:
            return pd.DataFrame()
        
        rebalance_dates = set(self.get_rebalance_dates(all_dates, self.rebalance_frequency))
        
        # 大盘风控
        market_risk_series = pd.Series(False, index=all_dates)
        if benchmark_data is not None and not benchmark_data.empty:
            try:
                index_df = benchmark_data.copy()
                if not isinstance(index_df.index, pd.DatetimeIndex):
                    if 'date' in index_df.columns:
                        index_df = index_df.set_index('date')
                    else:
                        index_df.index = pd.to_datetime(index_df.index)
                index_df = index_df.sort_index()
                
                index_df['ma60'] = index_df['close'].rolling(window=60).mean()
                index_df['drop_20d'] = (index_df['close'] - index_df['close'].shift(20)) / index_df['close'].shift(20)
                
                aligned_index = index_df.reindex(all_dates, method='ffill')
                market_risk_series = (
                    (aligned_index['close'] < aligned_index['ma60']) | 
                    (aligned_index['drop_20d'] < -0.05)
                ).fillna(False)
            except Exception as e:
                logger.warning(f"大盘风控计算失败: {e}")
        
        # 初始化持仓矩阵
        target_positions = pd.DataFrame(False, index=all_dates, columns=sorted(all_stocks), dtype=bool)
        target_positions.index.name = 'date'
        target_positions.columns.name = 'symbol'
        
        current_holdings: List[str] = []
        
        for date in all_dates:
            is_risk_triggered = market_risk_series.loc[date]
            
            if date in rebalance_dates:
                filtered_data = self.filter_stocks(data, date)
                if not filtered_data.empty:
                    try:
                        current_holdings = self.select_top_stocks(filtered_data, n=self.top_n, date=date)
                    except LLMCircuitBreakerError:
                        raise
                else:
                    current_holdings = []
            
            if is_risk_triggered:
                continue
            
            for stock in current_holdings:
                if stock in target_positions.columns:
                    target_positions.loc[date, stock] = True
        
        return target_positions
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号（兼容基类接口）"""
        has_factors = all(col in data.columns for col in [self.value_col, self.quality_col, self.momentum_col])
        
        if not has_factors:
            return pd.Series(0, index=data.index)
        
        total_score = self.calculate_total_score(data)
        signals = pd.Series(0, index=data.index)
        
        high_threshold = total_score.quantile(0.9)
        signals[total_score >= high_threshold] = 1
        
        low_threshold = total_score.quantile(0.1)
        signals[total_score <= low_threshold] = -1
        
        return signals
    
    def calculate_position_size(self, signal: TradeSignal, portfolio_value: float) -> float:
        """计算仓位大小（等权重分配）"""
        base_size = portfolio_value / self.top_n
        return base_size * signal.strength
    
    def optimize_weights(
        self,
        prices: pd.DataFrame,
        selected_stocks: List[str],
        objective: str = "max_sharpe",
        risk_free_rate: float = 0.02,
        max_weight: Optional[float] = None,
        min_weight: float = 0.0,
        lookback_days: int = 252
    ) -> Dict[str, float]:
        """使用 PyPortfolioOpt 优化投资组合权重"""
        try:
            from pypfopt import EfficientFrontier, risk_models, expected_returns
        except ImportError:
            logger.warning("未安装 pypfopt，使用等权重分配")
            return self._equal_weights(selected_stocks)
        
        if max_weight is None:
            max_weight = min(0.05, 1.0 / len(selected_stocks))
        
        available_stocks = [s for s in selected_stocks if s in prices.columns]
        if len(available_stocks) < 2:
            return self._equal_weights(selected_stocks)
        
        stock_prices = prices[available_stocks].tail(lookback_days).dropna(axis=1)
        if stock_prices.shape[1] < 2:
            return self._equal_weights(selected_stocks)
        
        try:
            if objective == "equal_weight":
                return self._equal_weights(selected_stocks)
            
            mu = expected_returns.mean_historical_return(stock_prices)
            S = risk_models.CovarianceShrinkage(stock_prices).ledoit_wolf()
            
            ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
            
            if objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == "min_volatility":
                ef.min_volatility()
            else:
                raise ValueError(f"不支持的优化目标: {objective}")
            
            return ef.clean_weights(cutoff=1e-4, rounding=4)
        except Exception as e:
            logger.warning(f"权重优化失败: {e}")
            return self._equal_weights(available_stocks)
    
    def _equal_weights(self, stocks: List[str]) -> Dict[str, float]:
        """生成等权重分配"""
        if not stocks:
            return {}
        weight = 1.0 / len(stocks)
        return {stock: weight for stock in stocks}
    
    def generate_target_weights(
        self,
        factor_data: pd.DataFrame,
        prices: pd.DataFrame,
        objective: str = "equal_weight",
        risk_free_rate: float = 0.02,
        max_weight: Optional[float] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        current_holdings_weights: Optional[Dict[str, float]] = None,
        rebalance_threshold: Optional[float] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """生成带权重的目标持仓矩阵（含懒惰再平衡 + 大盘风控）"""
        buffer_threshold = rebalance_threshold if rebalance_threshold is not None else self.rebalance_buffer
        
        if max_weight is None:
            max_weight = min(0.25, 1.0 / self.top_n)
        
        # 确定日期和股票列
        if self.date_col in factor_data.columns:
            dates_array = pd.to_datetime(factor_data[self.date_col].unique())
        elif isinstance(factor_data.index, pd.DatetimeIndex):
            dates_array = factor_data.index.unique()
        elif isinstance(factor_data.index, pd.MultiIndex):
            dates_array = factor_data.index.get_level_values(0).unique()
        else:
            raise ValueError("无法确定日期列")
        
        stock_col = self.stock_col if self.stock_col in factor_data.columns else 'symbol'
        if stock_col in factor_data.columns:
            all_stocks = factor_data[stock_col].unique()
        elif isinstance(factor_data.index, pd.MultiIndex):
            all_stocks = factor_data.index.get_level_values(-1).unique()
        else:
            raise ValueError("无法确定股票代码列")
        
        all_dates = pd.DatetimeIndex(sorted(dates_array))
        if start_date is not None:
            all_dates = all_dates[all_dates >= start_date]
        if end_date is not None:
            all_dates = all_dates[all_dates <= end_date]
        
        if all_dates.empty:
            return pd.DataFrame()
        
        rebalance_dates = set(self.get_rebalance_dates(all_dates, self.rebalance_frequency))
        
        # 大盘风控
        market_risk_series = pd.Series(False, index=all_dates)
        if self._market_risk_enabled and benchmark_data is not None and not benchmark_data.empty:
            try:
                index_df = benchmark_data.copy()
                if not isinstance(index_df.index, pd.DatetimeIndex):
                    if 'date' in index_df.columns:
                        index_df = index_df.set_index('date')
                    else:
                        index_df.index = pd.to_datetime(index_df.index)
                index_df = index_df.sort_index()
                
                ma_period = self._market_risk_ma_period
                index_df['ma60'] = index_df['close'].rolling(window=ma_period).mean()
                
                drop_lookback = self._market_risk_drop_lookback
                index_df['drop_20d'] = (index_df['close'] - index_df['close'].shift(drop_lookback)) / index_df['close'].shift(drop_lookback)
                
                aligned_index = index_df.reindex(all_dates, method='ffill')
                drop_threshold = -self._market_risk_drop_threshold
                
                market_risk_series = (
                    (aligned_index['close'] < aligned_index['ma60']) & 
                    (aligned_index['drop_20d'] < drop_threshold)
                ).fillna(False)
            except Exception as e:
                logger.warning(f"大盘风控计算失败: {e}")
        
        # 初始化权重矩阵
        target_weights = pd.DataFrame(0.0, index=all_dates, columns=sorted(all_stocks), dtype=float)
        target_weights.index.name = 'date'
        target_weights.columns.name = 'symbol'
        
        current_weights: Dict[str, float] = current_holdings_weights.copy() if current_holdings_weights else {}
        
        for date in all_dates:
            is_risk_triggered = market_risk_series.loc[date]
            
            if is_risk_triggered:
                current_weights = {}
                continue
            
            if date in rebalance_dates:
                # 回测与实盘一致：调仓日按当日市场状态应用自适应权重与动态仓位系数
                try:
                    if hasattr(self, 'apply_adaptive_weights') and benchmark_data is not None:
                        self.apply_adaptive_weights(index_data=benchmark_data, date=date)
                except Exception as e:
                    logger.warning(f"回测自适应权重应用失败（忽略并降级）: {e}")

                filtered_data = self.filter_stocks(factor_data, date)
                
                if not filtered_data.empty:
                    # 添加持股惯性标记
                    current_holding_set = set(current_weights.keys()) if current_weights else set()
                    filtered_data = filtered_data.copy()
                    if stock_col in filtered_data.columns:
                        filtered_data['is_holding'] = filtered_data[stock_col].isin(current_holding_set)
                    
                    # 选股
                    buffer_n = self.top_n * self._sentiment_buffer_multiplier
                    pre_candidates = self.select_top_stocks(filtered_data, n=buffer_n)
                    
                    if self._enable_sentiment_filter and self._sentiment_engine is not None:
                        try:
                            final_candidates = self._apply_sentiment_filter(pre_candidates, date)
                        except LLMCircuitBreakerError:
                            raise
                    else:
                        final_candidates = pre_candidates
                    
                    selected_stocks = final_candidates[:self.top_n]
                    
                    if selected_stocks:
                        # 懒惰再平衡
                        selected_set = set(selected_stocks)
                        continuing_stocks = current_holding_set & selected_set
                        stocks_to_sell = current_holding_set - selected_set
                        stocks_to_buy = selected_set - current_holding_set
                        
                        if not current_holding_set:
                            final_weights = self._equal_weights(selected_stocks)
                        else:
                            final_weights_dict: Dict[str, float] = {}
                            
                            for stock in continuing_stocks:
                                final_weights_dict[stock] = current_weights[stock]
                            
                            released_weight = sum(current_weights.get(s, 0.0) for s in stocks_to_sell)
                            
                            if stocks_to_buy:
                                if released_weight > 0:
                                    weight_per_new = released_weight / len(stocks_to_buy)
                                    for stock in stocks_to_buy:
                                        final_weights_dict[stock] = weight_per_new
                                else:
                                    n_total = len(continuing_stocks) + len(stocks_to_buy)
                                    target_weight = 1.0 / n_total
                                    final_weights_dict = {s: target_weight for s in continuing_stocks | stocks_to_buy}
                            
                            # 归一化
                            weight_sum = sum(final_weights_dict.values())
                            if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
                                final_weights_dict = {k: v / weight_sum for k, v in final_weights_dict.items()}
                            
                            final_weights = final_weights_dict
                        
                        current_weights = final_weights
                else:
                    current_weights = {}
            
            # 动态仓位系数：留现金以降低高波动/弱市阶段回撤
            position_scale = self.get_position_scale() if hasattr(self, "get_position_scale") else 1.0
            for stock, weight in current_weights.items():
                if stock in target_weights.columns:
                    target_weights.loc[date, stock] = weight * position_scale
        
        return target_weights
    
    def get_rebalance_summary(self, target_positions: pd.DataFrame) -> pd.DataFrame:
        """获取调仓汇总信息"""
        if target_positions.empty:
            return pd.DataFrame()
        
        position_change = target_positions.astype(int).diff()
        rebalance_dates = self.get_month_end_dates(target_positions.index)
        
        summary_records = []
        for date in rebalance_dates:
            if date not in position_change.index:
                continue
            
            day_change = position_change.loc[date]
            buy_stocks = day_change[day_change == 1].index.tolist()
            sell_stocks = day_change[day_change == -1].index.tolist()
            hold_stocks = target_positions.loc[date][target_positions.loc[date]].index.tolist()
            
            summary_records.append({
                'date': date,
                'buy_count': len(buy_stocks),
                'sell_count': len(sell_stocks),
                'hold_count': len(hold_stocks),
                'buy_stocks': buy_stocks,
                'sell_stocks': sell_stocks,
            })
        
        return pd.DataFrame(summary_records)
    
    # ==================== 拥挤度板块轮动方法 ====================
    
    def calculate_sector_crowding(
        self,
        price_data: pd.DataFrame,
        stock_sector_map: Dict[str, str],
        window: int = 20
    ) -> pd.DataFrame:
        """计算行业拥挤度因子"""
        if not self._enable_crowding_rotation:
            return pd.DataFrame()
        
        try:
            from src.crowding_factor import CrowdingFactorCalculator
        except ImportError:
            try:
                from crowding_factor import CrowdingFactorCalculator
            except ImportError:
                logger.warning("无法导入 CrowdingFactorCalculator")
                return pd.DataFrame()
        
        if self._crowding_calculator is None:
            self._crowding_calculator = CrowdingFactorCalculator(window=window, min_periods=10)
        
        try:
            return self._crowding_calculator.calculate(price_data, stock_sector_map)
        except Exception as e:
            logger.warning(f"行业拥挤度计算失败: {e}")
            return pd.DataFrame()
    
    def apply_crowding_rotation(
        self,
        candidates: List[str],
        crowding_data: pd.DataFrame,
        stock_sector_map: Dict[str, str],
        date: pd.Timestamp,
        current_holdings: Optional[List[str]] = None
    ) -> List[str]:
        """应用拥挤度轮动策略调整候选股票"""
        if not self._enable_crowding_rotation:
            return candidates
        
        if crowding_data.empty or date not in crowding_data.index:
            return candidates
        
        day_crowding = crowding_data.loc[date]
        crowding_percentile = day_crowding.rank(pct=True)
        
        adjusted_candidates = []
        for stock in candidates:
            sector = stock_sector_map.get(stock)
            if sector is None:
                adjusted_candidates.append(stock)
                continue
            
            sector_percentile = crowding_percentile.get(sector, 0.5)
            if sector_percentile <= self._crowding_exit_threshold:
                adjusted_candidates.append(stock)
        
        return adjusted_candidates
    
    def get_low_crowding_sectors(self, crowding_data: pd.DataFrame, date: pd.Timestamp) -> List[str]:
        """获取低拥挤度的行业列表"""
        if crowding_data.empty or date not in crowding_data.index:
            return []
        
        day_crowding = crowding_data.loc[date]
        crowding_percentile = day_crowding.rank(pct=True)
        
        return crowding_percentile[crowding_percentile < self._crowding_entry_threshold].index.tolist()

