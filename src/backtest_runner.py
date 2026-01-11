"""
回测运行器模块

本模块提供策略回测的核心逻辑。
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import pandas as pd
import numpy as np

from .strategy import MultiFactorStrategy
from .features import calculate_factor_ic, calculate_forward_returns

# 延迟导入
BacktestEngine = None
DataLoader = None
TushareDataLoader = None

logger = logging.getLogger(__name__)

# 默认路径
DATA_RAW_PATH = Path("data/raw")
DATA_PROCESSED_PATH = Path("data/processed")
REPORTS_PATH = Path("reports")


def _lazy_import():
    """延迟导入重型模块"""
    global BacktestEngine, DataLoader, TushareDataLoader
    
    if BacktestEngine is None:
        try:
            from .backtest import BacktestEngine
        except ImportError:
            from backtest import BacktestEngine
    
    if DataLoader is None:
        try:
            from .data_loader import DataLoader
        except ImportError:
            from data_loader import DataLoader
    
    if TushareDataLoader is None:
        try:
            from .tushare import TushareDataLoader
        except ImportError:
            TushareDataLoader = None


def generate_factor_data(
    price_data_dict: Dict[str, pd.DataFrame],
    close_df: pd.DataFrame,
    strategy_config: Dict[str, Any],
    financial_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    生成回测用因子数据
    
    Parameters
    ----------
    price_data_dict : Dict[str, pd.DataFrame]
        各股票的价格数据
    close_df : pd.DataFrame
        收盘价矩阵
    strategy_config : Dict[str, Any]
        策略配置
    financial_data : Optional[pd.DataFrame]
        财务数据（含市值）
    
    Returns
    -------
    pd.DataFrame
        因子数据
    """
    factor_records = []
    
    # 计算技术因子
    for stock_code, df in price_data_dict.items():
        if df.empty or 'close' not in df.columns:
            continue
        
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/20, min_periods=20).mean()
        avg_loss = loss.ewm(alpha=1/20, min_periods=20).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi_20'] = 100 - (100 / (1 + rs))
        
        # 换手率
        if 'turn' in df.columns:
            df['turnover_5d'] = df['turn'].rolling(5).mean()
        else:
            df['turnover_5d'] = 0
        
        # 动量
        df['return_20'] = df['close'].pct_change(20)
        
        # 路径效率
        abs_changes = df['close'].diff().abs().rolling(20).sum()
        net_change = df['close'].diff(20).abs()
        df['efficiency_20'] = net_change / abs_changes.replace(0, np.nan)
        
        # 夏普比率
        returns = df['close'].pct_change()
        df['sharpe_20'] = (
            returns.rolling(20).mean() / returns.rolling(20).std().replace(0, np.nan)
        ) * np.sqrt(252)
        
        for date, row in df.iterrows():
            factor_records.append({
                'date': date,
                'stock_code': stock_code,
                'close': row.get('close', np.nan),
                'rsi_20': row.get('rsi_20', np.nan),
                'turnover_5d': row.get('turnover_5d', 0),
                'return_20': row.get('return_20', np.nan),
                'efficiency_20': row.get('efficiency_20', np.nan),
                'sharpe_20': row.get('sharpe_20', np.nan),
            })
    
    factor_data = pd.DataFrame(factor_records)
    
    if factor_data.empty:
        return factor_data
    
    # 合并财务数据
    if financial_data is not None and not financial_data.empty:
        if 'stock_code' in financial_data.columns:
            fin_cols = ['stock_code']
            for col in ['circ_mv', 'total_mv', 'pe_ttm', 'pb']:
                if col in financial_data.columns:
                    fin_cols.append(col)
            
            fin_df = financial_data[fin_cols].drop_duplicates(subset=['stock_code'])
            factor_data = factor_data.merge(fin_df, on='stock_code', how='left')
    
    # Z-Score 标准化（按日期横截面）
    def zscore_by_date(group, col):
        mean = group[col].mean()
        std = group[col].std()
        if std > 0:
            return (group[col] - mean) / std
        return 0
    
    zscore_mappings = {
        'rsi_20': 'rsi_20_zscore',
        'turnover_5d': 'turnover_5d_zscore',
        'return_20': 'momentum_zscore',
        'sharpe_20': 'sharpe_20_zscore',
        'efficiency_20': 'efficiency_20_zscore',
    }
    
    for src_col, dst_col in zscore_mappings.items():
        if src_col in factor_data.columns:
            factor_data[dst_col] = factor_data.groupby('date').apply(
                lambda g: zscore_by_date(g, src_col)
            ).reset_index(level=0, drop=True)
    
    # 小市值因子
    if 'circ_mv' in factor_data.columns:
        def small_cap_zscore(group):
            log_mv = np.log(group['circ_mv'].replace(0, np.nan))
            mean = log_mv.mean()
            std = log_mv.std()
            if std > 0:
                return -(log_mv - mean) / std
            return 0
        
        factor_data['small_cap_zscore'] = factor_data.groupby('date').apply(
            small_cap_zscore
        ).reset_index(level=0, drop=True)
    
    return factor_data


def run_backtest(
    start_date: str,
    end_date: str,
    config: Optional[Dict[str, Any]] = None,
    strategy_type: str = "multi_factor",
    no_llm: bool = False
) -> bool:
    """
    运行策略回测
    
    Parameters
    ----------
    start_date : str
        回测开始日期 (YYYY-MM-DD)
    end_date : str
        回测结束日期 (YYYY-MM-DD)
    config : Optional[Dict[str, Any]]
        回测配置参数
    strategy_type : str
        策略类型: 'multi_factor', 'ma_cross'
    no_llm : bool
        是否禁用 LLM 风控
    
    Returns
    -------
    bool
        回测是否成功
    """
    _lazy_import()
    
    logger.info("=" * 60)
    logger.info(f"开始回测: {start_date} ~ {end_date}")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        if config is None:
            config = {}
        
        if no_llm:
            config["llm"] = {}
        
        backtest_config = config.get("backtest", {})
        portfolio_config = config.get("portfolio", {})
        strategy_config = config.get("strategy", {})
        data_config = config.get("data", {})
        
        initial_capital = portfolio_config.get("total_capital", 300000)
        commission = config.get("trading", {}).get("commission_rate", 0.0003)
        slippage = config.get("trading", {}).get("slippage", 0.001)
        benchmark_code = backtest_config.get("benchmark", "000905")
        
        logger.info(f"回测配置: 初始资金=¥{initial_capital:,.0f}, 基准={benchmark_code}")
        
        # Step 1: 加载历史数据
        logger.info("Step 1/6: 加载历史 OHLCV 数据")
        
        data_loader = DataLoader(output_dir=str(DATA_RAW_PATH))
        stock_pool = data_config.get("stock_pool", "hs300")
        
        # 获取股票列表
        stock_list = []
        if TushareDataLoader is not None:
            try:
                ts_loader = TushareDataLoader()
                stock_list = ts_loader.fetch_index_constituents(index_code=stock_pool)
            except Exception as e:
                logger.warning(f"获取成分股失败: {e}")
        
        if not stock_list:
            stock_list = data_loader.get_hs300_constituents()
        
        if not stock_list:
            logger.warning("无法获取成分股列表，使用示例股票")
            stock_list = ["000001", "000002", "600519", "601318", "000858"]
        
        max_stocks = backtest_config.get("max_stocks", 100)
        stock_list = stock_list[:max_stocks]
        logger.info(f"股票池: {stock_pool}, 回测股票数量: {len(stock_list)}")
        
        # 下载历史数据
        price_data_dict: Dict[str, pd.DataFrame] = {}
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        
        # 优先从本地缓存加载
        local_cache_dir = Path("data/lake/daily")
        
        for i, stock in enumerate(stock_list):
            df = None
            try:
                cache_file = local_cache_dir / f"{stock}.parquet"
                if cache_file.exists():
                    try:
                        cached_df = pd.read_parquet(cache_file)
                        
                        if 'date' not in cached_df.columns:
                            if isinstance(cached_df.index, pd.DatetimeIndex):
                                cached_df = cached_df.reset_index()
                                cached_df.columns = ['date'] + list(cached_df.columns[1:])
                        
                        if not cached_df.empty and 'date' in cached_df.columns:
                            cached_df['date'] = pd.to_datetime(cached_df['date'])
                            req_start = pd.to_datetime(start_date)
                            req_end = pd.to_datetime(end_date)
                            
                            df = cached_df[
                                (cached_df['date'] >= req_start) & 
                                (cached_df['date'] <= req_end)
                            ].copy()
                            
                            if not df.empty:
                                df = df.set_index('date').sort_index()
                    except Exception:
                        pass
                
                if df is None or df.empty:
                    df = data_loader.fetch_daily_price(stock, start_fmt, end_fmt)
                
                if df is not None and not df.empty:
                    price_data_dict[stock] = df
                    
            except Exception as e:
                logger.debug(f"获取 {stock} 数据失败: {e}")
            
            if (i + 1) % 50 == 0:
                logger.info(f"数据加载进度: {i + 1}/{len(stock_list)}")
        
        if not price_data_dict:
            logger.error("未获取到任何历史数据")
            return False
        
        logger.info(f"成功加载 {len(price_data_dict)} 只股票的历史数据")
        
        # Step 2: 准备价格矩阵
        logger.info("Step 2/6: 准备价格矩阵")
        
        close_prices = {}
        for stock, df in price_data_dict.items():
            if 'close' in df.columns:
                close_prices[stock] = df['close']
        
        close_df = pd.DataFrame(close_prices)
        close_df.index = pd.to_datetime(close_df.index)
        close_df = close_df.sort_index().ffill().bfill()
        
        logger.info(f"价格矩阵: {close_df.shape[0]} 天 x {close_df.shape[1]} 只股票")
        
        # Step 3: 加载财务数据
        logger.info("Step 3/6: 加载财务数据")
        
        financial_data = None
        financial_files = sorted(DATA_RAW_PATH.glob("financial_*.parquet"), reverse=True)
        
        if financial_files:
            try:
                financial_data = pd.read_parquet(financial_files[0])
                if 'stock_code' in financial_data.columns:
                    financial_data = financial_data[
                        financial_data['stock_code'].isin(price_data_dict.keys())
                    ]
                logger.info(f"财务数据加载成功: {len(financial_data)} 条")
            except Exception as e:
                logger.warning(f"财务数据加载失败: {e}")
        
        # Step 4: 获取基准指数
        logger.info("Step 4/6: 获取基准指数数据")
        
        benchmark_data = None
        try:
            benchmark_data = data_loader.fetch_index_price(
                index_code=benchmark_code,
                start_date=start_date,
                end_date=end_date
            )
            if benchmark_data is not None and not benchmark_data.empty:
                logger.info(f"基准指数数据: {len(benchmark_data)} 条")
        except Exception as e:
            logger.warning(f"获取基准指数失败: {e}")
        
        # Step 5: 生成因子数据
        logger.info("Step 5/6: 生成因子数据")
        
        factor_data = generate_factor_data(
            price_data_dict=price_data_dict,
            close_df=close_df,
            strategy_config=strategy_config,
            financial_data=financial_data
        )
        
        if factor_data.empty:
            logger.error("因子数据生成失败")
            return False
        
        logger.info(f"因子数据: {len(factor_data)} 条记录")
        
        # Step 6: 执行回测
        logger.info("Step 6/6: 执行回测")
        
        # 初始化策略
        strategy = MultiFactorStrategy(
            name=strategy_config.get("name", "Multi-Factor Strategy"),
            config={
                "value_weight": strategy_config.get("value_weight", 0.0),
                "quality_weight": strategy_config.get("quality_weight", 0.3),
                "momentum_weight": strategy_config.get("momentum_weight", 0.7),
                "size_weight": strategy_config.get("size_weight", 0.0),
                "top_n": strategy_config.get("top_n", 5),
                "min_listing_days": strategy_config.get("min_listing_days", 126),
                "value_col": strategy_config.get("value_col", "value_zscore"),
                "quality_col": strategy_config.get("quality_col", "turnover_5d_zscore"),
                "momentum_col": strategy_config.get("momentum_col", "sharpe_20_zscore"),
                "size_col": strategy_config.get("size_col", "small_cap_zscore"),
                "rebalance_frequency": strategy_config.get("rebalance_frequency", "monthly"),
                "market_regime": strategy_config.get("market_regime", {}),
                "score_normalization": strategy_config.get("score_normalization", {}),
                "min_daily_amount": strategy_config.get("min_daily_amount", 50_000_000),
                "min_circ_mv": strategy_config.get("min_circ_mv", None),
                "max_price": strategy_config.get("max_price", 100.0),
                "max_rsi": strategy_config.get("max_rsi", 80.0),
                "min_efficiency": strategy_config.get("min_efficiency", 0.3),
                "overheat_check_col": strategy_config.get("overheat_check_col", strategy_config.get("quality_col", "turnover_5d_zscore")),
            }
        )

        # IC 监控（可选）：用于回测时的“方向校准/因子熔断”（与实盘对齐）
        try:
            ic_cfg = config.get("ic_monitor", {}) if config is not None else {}
            if ic_cfg.get("enabled", False) and factor_data is not None and not factor_data.empty:
                lookback_days = int(ic_cfg.get("lookback_days", 5))
                monitored_factors: List[str] = list(ic_cfg.get("monitored_factors", []))
                if monitored_factors:
                    ic_src = calculate_forward_returns(
                        data=factor_data,
                        periods=[lookback_days],
                        stock_col='stock_code' if 'stock_code' in factor_data.columns else 'symbol',
                        price_col='close'
                    )
                    ic_df = calculate_factor_ic(
                        data=ic_src,
                        factor_cols=monitored_factors,
                        return_col=f'forward_return_{lookback_days}d',
                        date_col='date' if 'date' in ic_src.columns else 'trade_date',
                        stock_col='stock_code' if 'stock_code' in ic_src.columns else 'symbol',
                        log_results=False
                    )

                    if ic_cfg.get("circuit_breaker_enabled", False) and not ic_df.empty:
                        strategy.apply_factor_circuit_breaker(
                            ic_results=ic_df,
                            ic_threshold=float(ic_cfg.get("circuit_breaker_ic_threshold", 0.005)),
                            ir_threshold=float(ic_cfg.get("circuit_breaker_ir_threshold", 0.2))
                        )

                    dir_cfg = ic_cfg.get("directional_adjustment", {})
                    if dir_cfg.get("enabled", True) and not ic_df.empty:
                        strategy.apply_factor_direction_from_ic(
                            ic_results=ic_df,
                            abs_ic_threshold=float(dir_cfg.get("abs_ic_threshold", 0.02)),
                            ir_threshold=float(dir_cfg.get("ir_threshold", 0.3)),
                            positive_ratio_threshold=float(dir_cfg.get("positive_ratio_threshold", 0.55))
                        )
        except Exception as e:
            logger.warning(f"回测 IC 监控/自适应失败（忽略并降级）: {e}")
        
        # 生成目标权重
        target_weights = strategy.generate_target_weights(
            factor_data=factor_data,
            prices=close_df,
            objective="equal_weight",
            benchmark_data=benchmark_data
        )
        
        if target_weights.empty:
            logger.error("目标权重生成失败")
            return False
        
        logger.info(f"目标权重矩阵: {target_weights.shape}")
        
        # 初始化回测引擎
        engine_config = {
            "initial_capital": initial_capital,
            "commission": commission,
            "slippage": slippage,
            "risk_free_rate": portfolio_config.get("risk_free_rate", 0.03),
        }
        engine = BacktestEngine(config=engine_config)
        
        # 执行回测
        results = engine.run(
            strategy=strategy,
            price_data=close_df,
            factor_data=factor_data,
            target_weights=target_weights,
            benchmark_data=benchmark_data
        )
        
        if results is None:
            logger.error("回测执行失败")
            return False
        
        # 输出结果
        logger.info("=" * 60)
        logger.info("回测完成，结果摘要：")
        logger.info(f"  总收益率: {results.get('total_return', 0):.2%}")
        logger.info(f"  年化收益: {results.get('annual_return', 0):.2%}")
        logger.info(f"  最大回撤: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"  夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  胜率: {results.get('win_rate', 0):.1%}")
        logger.info("=" * 60)
        
        # 保存结果
        results_path = REPORTS_PATH / f"backtest_{start_date}_{end_date}.json"
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                      for k, v in results.items()}, f, indent=2)
        
        logger.info(f"回测结果已保存至: {results_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

