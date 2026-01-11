"""
每日更新运行器模块

本模块提供每日数据更新、因子计算、调仓信号生成和报告输出的核心逻辑。
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json
import os

import pandas as pd
import numpy as np

from .strategy import MultiFactorStrategy
from .report_generator import ReportGenerator
from .features import calculate_factor_ic, calculate_forward_returns
from .utils.messaging import send_pushplus_msg

# 延迟导入，避免循环依赖
TushareDataLoader = None
AShareDataCleaner = None
DataLoader = None

logger = logging.getLogger(__name__)

# 默认路径
DATA_RAW_PATH = Path("data/raw")
DATA_PROCESSED_PATH = Path("data/processed")
REPORTS_PATH = Path("reports")


def _lazy_import():
    """延迟导入重型模块"""
    global TushareDataLoader, AShareDataCleaner, DataLoader
    
    if TushareDataLoader is None:
        try:
            from .tushare import TushareDataLoader
        except ImportError:
            TushareDataLoader = None
    
    if AShareDataCleaner is None:
        try:
            from .data_loader import AShareDataCleaner
        except ImportError:
            AShareDataCleaner = None


class DailyUpdateRunner:
    """
    每日更新运行器
    
    负责执行每日数据更新、因子计算、调仓信号生成和报告输出。
    
    Parameters
    ----------
    config : Optional[Dict[str, Any]]
        配置参数
    
    Attributes
    ----------
    config : Dict[str, Any]
        配置参数
    tushare_loader : TushareDataLoader
        Tushare 数据加载器
    strategy : MultiFactorStrategy
        多因子策略
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化每日更新运行器"""
        _lazy_import()
        
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # 确保目录存在
        DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
        
        # 状态变量
        self.today = pd.Timestamp.now().normalize()
        self.ohlcv_data: Optional[pd.DataFrame] = None
        self.financial_data: Optional[pd.DataFrame] = None
        self.industry_data: Optional[pd.DataFrame] = None
        self.factor_data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.current_positions: Dict[str, float] = {}
        self.target_positions: Dict[str, float] = {}
        
        # 报告生成器
        self.report_generator = ReportGenerator(self.config, REPORTS_PATH)
        
        # 加载当前持仓
        self.load_current_holdings()
        
        self.logger.info("DailyUpdateRunner 初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "data": {
                "stock_pool": "hs300",
                "start_date": "2020-01-01",
                "update_days": 5,
            },
            "strategy": {
                "name": "Multi-Factor Strategy",
                "value_weight": 0.0,
                "quality_weight": 0.3,
                "momentum_weight": 0.7,
                "top_n": 5,
                "min_listing_days": 126,
            },
            "portfolio": {
                "total_capital": 300000,
                "max_weight": 0.25,
                "risk_free_rate": 0.02,
            },
            "report": {
                "format": "markdown",
                "output_dir": "reports",
            },
        }
    
    def _init_components(self) -> None:
        """初始化各组件"""
        # Tushare 数据加载器
        tushare_config = self.config.get("tushare", {})
        api_token = tushare_config.get("api_token") or os.environ.get("TUSHARE_TOKEN", "")
        
        if not api_token:
            self.logger.warning("Tushare API Token 未配置，部分功能可能不可用")
        
        self.tushare_loader = TushareDataLoader(
            api_token=api_token,
            cache_dir=tushare_config.get("cache_dir", "data/tushare_cache")
        )
        
        self.data_cleaner = AShareDataCleaner()
        
        # 策略
        strategy_config = self.config.get("strategy", {})
        llm_config = self.config.get("llm", {})
        
        self.strategy = MultiFactorStrategy(
            name=strategy_config.get("name", "Multi-Factor Strategy"),
            config={
                "value_weight": strategy_config.get("value_weight", 0.0),
                "quality_weight": strategy_config.get("quality_weight", 0.3),
                "momentum_weight": strategy_config.get("momentum_weight", 0.7),
                "size_weight": strategy_config.get("size_weight", 0.0),
                "sentiment_weight": strategy_config.get("sentiment_weight", 0.0),
                "top_n": strategy_config.get("top_n", 5),
                "min_listing_days": strategy_config.get("min_listing_days", 126),
                "exclude_chinext": strategy_config.get("exclude_chinext", False),
                "exclude_star": strategy_config.get("exclude_star", False),
                "value_col": strategy_config.get("value_col", "value_zscore"),
                "quality_col": strategy_config.get("quality_col", "turnover_5d_zscore"),
                "momentum_col": strategy_config.get("momentum_col", "sharpe_20_zscore"),
                "size_col": strategy_config.get("size_col", "small_cap_zscore"),
                "rebalance_frequency": strategy_config.get("rebalance_frequency", "monthly"),
                "rebalance_buffer": strategy_config.get("rebalance_buffer", 0.05),
                "holding_bonus": strategy_config.get("holding_bonus", 0.0),
                "turnover_threshold": strategy_config.get("turnover_threshold", 50.0),
                "volatility_threshold": strategy_config.get("volatility_threshold", 5.0),
                "min_daily_amount": strategy_config.get("min_daily_amount", 50_000_000),
                "min_circ_mv": strategy_config.get("min_circ_mv", None),
                "max_price": strategy_config.get("max_price", 100.0),
                "max_rsi": strategy_config.get("max_rsi", 80.0),
                "min_efficiency": strategy_config.get("min_efficiency", 0.3),
                "overheat_check_col": strategy_config.get("overheat_check_col", strategy_config.get("quality_col", "turnover_5d_zscore")),
                "market_risk": self.config.get("risk", {}).get("market_risk", {}),
                "market_regime": strategy_config.get("market_regime", {}),
                "score_normalization": strategy_config.get("score_normalization", {}),
                "llm": llm_config,
            }
        )

    def _compute_ic_results(self) -> pd.DataFrame:
        """
        计算因子 IC 监控结果（用于报告与在线自适应）。

        Returns
        -------
        pd.DataFrame
            因子 IC 统计表；若未启用或数据不足则返回空表。
        """
        ic_cfg = self.config.get("ic_monitor", {})
        if not ic_cfg.get("enabled", False):
            return pd.DataFrame()

        if self.factor_data is None or self.factor_data.empty:
            return pd.DataFrame()

        try:
            sample_days = int(ic_cfg.get("sample_days", 30))
            lookback_days = int(ic_cfg.get("lookback_days", 5))
            monitored_factors: List[str] = list(ic_cfg.get("monitored_factors", []))
            if not monitored_factors:
                return pd.DataFrame()

            df = self.factor_data.copy()
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                unique_dates = pd.DatetimeIndex(sorted(df['trade_date'].unique()))
                if len(unique_dates) > sample_days:
                    start_dt = unique_dates[-sample_days]
                    df = df[df['trade_date'] >= start_dt]

            # 计算前瞻收益
            df = calculate_forward_returns(
                data=df,
                periods=[lookback_days],
                stock_col='stock_code' if 'stock_code' in df.columns else 'symbol',
                price_col='close'
            )
            ret_col = f'forward_return_{lookback_days}d'
            # 关键：剔除“未来收益不可得”的样本（最近 lookback_days 天必然是 NaN）
            if ret_col in df.columns:
                df = df[df[ret_col].notna()].copy()

            # 监控因子预筛选：剔除“几乎全缺失/全常数”的因子，避免刷屏与误导
            date_col = 'trade_date' if 'trade_date' in df.columns else ('date' if 'date' in df.columns else None)
            if date_col is None:
                return pd.DataFrame()

            valid_factors: List[str] = []
            skipped: List[str] = []
            min_valid_days = int(ic_cfg.get("min_valid_days", 8))
            min_non_null = int(ic_cfg.get("min_non_null_rows", 5000))

            for fac in monitored_factors:
                if fac not in df.columns:
                    skipped.append(f"{fac}(missing)")
                    continue
                s = df[fac]
                if int(s.notna().sum()) < min_non_null:
                    skipped.append(f"{fac}(few_non_null)")
                    continue
                if int(s.dropna().nunique()) <= 1:
                    skipped.append(f"{fac}(constant)")
                    continue

                # 至少有若干交易日横截面非“常数”，否则 Spearman 会大量 NaN
                try:
                    nunique_by_day = df.groupby(date_col)[fac].nunique(dropna=True)
                    if int((nunique_by_day > 1).sum()) < min_valid_days:
                        skipped.append(f"{fac}(few_valid_days)")
                        continue
                except Exception:
                    skipped.append(f"{fac}(group_fail)")
                    continue

                valid_factors.append(fac)

            if skipped:
                self.logger.info(f"IC监控预筛选剔除因子: {', '.join(skipped[:10])}" + (" ..." if len(skipped) > 10 else ""))
            if not valid_factors:
                return pd.DataFrame()

            # IC
            ic_df = calculate_factor_ic(
                data=df,
                factor_cols=valid_factors,
                return_col=ret_col,
                date_col=date_col,
                stock_col='stock_code' if 'stock_code' in df.columns else 'symbol',
                log_results=False
            )
            return ic_df
        except Exception as e:
            self.logger.warning(f"IC 监控计算失败（忽略并降级）: {e}")
            return pd.DataFrame()
    
    def _validate_and_fix_data_units(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        对数据进行单位一致性检查，并自动纠正。
        
        Tushare 数据的标准单位:
        - volume: 股 (手需要 * 100)
        - amount: 千元 (需要转换为元: * 1000)
        - total_mv/circ_mv: 万元 (需要转换为元: * 10000)
        
        统一输出单位:
        - volume: 股
        - amount: 元
        - total_mv/circ_mv: 元
        
        Parameters
        ----------
        df : pd.DataFrame
            待检查的数据
        data_type : str
            数据类型: "ohlcv" 或 "financial"
        
        Returns
        -------
        pd.DataFrame
            单位已统一的数据
        """
        if df.empty:
            return df
        
        df = df.copy()
        corrections = []
        
        if data_type == "ohlcv":
            # 检查并修正成交量单位 (预期为股，正常股票单日成交量应 > 10万股)
            if 'volume' in df.columns:
                median_vol = df['volume'].median()
                if median_vol < 10000:
                    # 可能是手，转换为股
                    df['volume'] = df['volume'] * 100
                    corrections.append(f"volume: 手→股 (*100)")
            
            # 检查并修正成交额单位 (Tushare 原始为千元，统一转换为元)
            if 'amount' in df.columns:
                median_amt = df['amount'].median()
                if median_amt < 1e8:
                    # 可能是千元，转换为元
                    df['amount'] = df['amount'] * 1000
                    corrections.append(f"amount: 千元→元 (*1000)")
                    
        elif data_type == "financial":
            # 检查并修正市值单位 (Tushare 原始为万元，统一转换为元)
            for col in ['total_mv', 'circ_mv']:
                if col in df.columns:
                    max_val = df[col].max()
                    # 万元单位下，千亿市值 = 1e7 万元
                    if max_val < 1e10:
                        # 可能是万元，转换为元
                        df[col] = df[col] * 10000
                        corrections.append(f"{col}: 万元→元 (*10000)")
        
        if corrections:
            self.logger.info(f"📐 数据单位自动修正 ({data_type}): {', '.join(corrections)}")
        
        return df
    
    def _validate_data_units(self, df: pd.DataFrame, data_type: str) -> None:
        """
        对数据进行单位一致性检查（仅检查不修改，用于日志记录）。
        
        实际修正请使用 _validate_and_fix_data_units 方法。
        """
        if df.empty:
            return
        
        if data_type == "ohlcv":
            if 'volume' in df.columns:
                median_vol = df['volume'].median()
                if median_vol < 1000:
                    self.logger.warning(
                        f"⚠️ OHLCV 'volume' 单位可能错误 (中位数 {median_vol:.0f})，"
                        f"预期为股，当前可能为手"
                    )
            
            if 'amount' in df.columns:
                median_amt = df['amount'].median()
                if median_amt > 1e12:
                    self.logger.warning(
                        f"⚠️ OHLCV 'amount' 单位异常 (中位数 {median_amt:.0f})，"
                        f"请确认单位一致性"
                    )
                    
        elif data_type == "financial":
            for col in ['total_mv', 'circ_mv']:
                if col in df.columns:
                    max_val = df[col].max()
                    if max_val > 1e15:
                        self.logger.warning(
                            f"⚠️ 财务数据 '{col}' 单位异常 "
                            f"(最大值 {max_val:.2e})，请确认单位一致性"
                        )
    
    def load_current_holdings(self) -> None:
        """加载当前持仓"""
        holdings_path = DATA_PROCESSED_PATH / "real_holdings.json"
        
        if holdings_path.exists():
            try:
                with open(holdings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.current_positions = data.get("positions", {})
                self.current_positions = {str(k): float(v) for k, v in self.current_positions.items()}
                
                self.logger.info(
                    f"已加载持仓数据: {len(self.current_positions)} 只股票, "
                    f"总市值 ¥{sum(self.current_positions.values()):,.0f}"
                )
            except Exception as e:
                self.logger.warning(f"加载持仓文件失败: {e}")
                self.current_positions = {}
        else:
            self.logger.info("持仓文件不存在，初始化为空持仓")
            self.current_positions = {}
    
    def save_current_holdings(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float]
    ) -> None:
        """保存当前持仓"""
        new_positions = self.current_positions.copy()
        
        for stock, amount in buy_orders.items():
            new_positions[stock] = new_positions.get(stock, 0) + amount
        
        for stock, amount in sell_orders.items():
            if stock in new_positions:
                new_positions[stock] -= amount
                if new_positions[stock] <= 0:
                    del new_positions[stock]
        
        holdings_data = {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "update_date": self.today.strftime("%Y-%m-%d"),
            "positions": new_positions,
            "total_value": sum(new_positions.values()),
            "num_stocks": len(new_positions),
        }
        
        holdings_path = DATA_PROCESSED_PATH / "real_holdings.json"
        
        try:
            with open(holdings_path, 'w', encoding='utf-8') as f:
                json.dump(holdings_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"持仓已更新: {len(new_positions)} 只股票")
            self.current_positions = new_positions
        except Exception as e:
            self.logger.error(f"保存持仓文件失败: {e}")
    
    def import_broker_holdings(
        self,
        csv_path: Optional[str] = None,
        positions: Optional[Dict[str, float]] = None,
        cash: float = 0.0
    ) -> bool:
        """
        从券商导入实际持仓（日终对账）
        
        支持两种方式：
        1. 从 CSV 文件导入（券商导出）
        2. 直接传入持仓字典
        
        Parameters
        ----------
        csv_path : Optional[str]
            券商导出的持仓 CSV 文件路径
            预期列：股票代码, 持仓市值 (或 证券代码, 市值)
        positions : Optional[Dict[str, float]]
            直接传入的持仓字典 {股票代码: 市值}
        cash : float
            可用现金
        
        Returns
        -------
        bool
            导入是否成功
        """
        self.logger.info("=" * 50)
        self.logger.info("开始日终对账：导入券商实际持仓")
        self.logger.info("=" * 50)
        
        imported_positions: Dict[str, float] = {}
        
        if csv_path:
            try:
                import_df = pd.read_csv(csv_path, encoding='utf-8')
                
                # 尝试识别列名
                stock_col = None
                amount_col = None
                
                # 常见的股票代码列名
                for col in ['股票代码', '证券代码', 'stock_code', 'symbol', '代码']:
                    if col in import_df.columns:
                        stock_col = col
                        break
                
                # 常见的市值列名
                for col in ['持仓市值', '市值', '参考市值', 'amount', 'value', '市值（元）']:
                    if col in import_df.columns:
                        amount_col = col
                        break
                
                if stock_col is None or amount_col is None:
                    self.logger.error(
                        f"无法识别 CSV 列名，请确保包含股票代码和市值列。"
                        f"当前列: {list(import_df.columns)}"
                    )
                    return False
                
                for _, row in import_df.iterrows():
                    stock = str(row[stock_col]).strip()
                    # 标准化股票代码（提取6位数字）
                    import re
                    match = re.search(r'\d{6}', stock)
                    if match:
                        stock = match.group()
                    
                    amount = float(row[amount_col])
                    if amount > 0:
                        imported_positions[stock] = amount
                
                self.logger.info(f"从 CSV 导入 {len(imported_positions)} 只股票持仓")
                
            except Exception as e:
                self.logger.error(f"CSV 导入失败: {e}")
                return False
        
        elif positions:
            imported_positions = positions.copy()
            self.logger.info(f"直接导入 {len(imported_positions)} 只股票持仓")
        
        else:
            self.logger.error("请提供 csv_path 或 positions 参数")
            return False
        
        # 计算与系统持仓的偏差
        self._log_holdings_diff(imported_positions, cash)
        
        # 保存新持仓
        holdings_data = {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "update_date": self.today.strftime("%Y-%m-%d"),
            "positions": imported_positions,
            "cash": cash,
            "total_value": sum(imported_positions.values()) + cash,
            "num_stocks": len(imported_positions),
            "source": "broker_import",
            "note": "从券商实际持仓导入（日终对账）"
        }
        
        holdings_path = DATA_PROCESSED_PATH / "real_holdings.json"
        
        try:
            with open(holdings_path, 'w', encoding='utf-8') as f:
                json.dump(holdings_data, f, ensure_ascii=False, indent=2)
            
            self.current_positions = imported_positions
            self.logger.info(f"✅ 日终对账完成: {len(imported_positions)} 只股票, 现金 ¥{cash:,.0f}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存持仓失败: {e}")
            return False
    
    def _log_holdings_diff(self, new_positions: Dict[str, float], new_cash: float) -> None:
        """记录持仓偏差（系统持仓 vs 券商实际）"""
        old_positions = self.current_positions
        
        all_stocks = set(old_positions.keys()) | set(new_positions.keys())
        
        diff_lines = []
        total_old = sum(old_positions.values())
        total_new = sum(new_positions.values())
        
        for stock in sorted(all_stocks):
            old_amt = old_positions.get(stock, 0)
            new_amt = new_positions.get(stock, 0)
            diff = new_amt - old_amt
            
            if abs(diff) > 100:  # 忽略小偏差
                if old_amt == 0:
                    diff_lines.append(f"  + {stock}: ¥{new_amt:,.0f} (新增)")
                elif new_amt == 0:
                    diff_lines.append(f"  - {stock}: ¥{old_amt:,.0f} (清仓)")
                else:
                    sign = '+' if diff > 0 else ''
                    diff_lines.append(f"  Δ {stock}: ¥{old_amt:,.0f} → ¥{new_amt:,.0f} ({sign}{diff:,.0f})")
        
        if diff_lines:
            self.logger.warning("持仓偏差检测（系统 vs 券商）:")
            for line in diff_lines:
                self.logger.warning(line)
            
            total_diff = total_new - total_old
            self.logger.warning(f"  总市值偏差: ¥{total_diff:+,.0f} ({total_diff/total_old*100:+.1f}%)" if total_old > 0 else f"  总市值: ¥{total_new:,.0f}")
        else:
            self.logger.info("持仓无偏差")
    
    def update_market_data(self) -> bool:
        """更新市场数据"""
        self.logger.info("开始更新市场数据...")
        
        try:
            # 检查今日缓存
            ohlcv_path = DATA_RAW_PATH / f"ohlcv_{self.today.strftime('%Y%m%d')}.parquet"
            if ohlcv_path.exists():
                try:
                    self.ohlcv_data = pd.read_parquet(ohlcv_path)
                    if not self.ohlcv_data.empty:
                        self.logger.info(f"使用缓存数据: {ohlcv_path.name}")
                        return True
                except Exception as e:
                    self.logger.warning(f"读取缓存失败: {e}")
            
            data_config = self.config.get("data", {})
            stock_pool = data_config.get("stock_pool", "hs300")
            
            end_date = self.today.strftime("%Y%m%d")
            update_days = data_config.get("update_days", 5)
            start_date = (self.today - timedelta(days=update_days * 2)).strftime("%Y%m%d")
            
            # 获取股票列表
            if stock_pool == "all":
                stock_list = self.tushare_loader.fetch_all_stocks()
            else:
                stock_list = self.tushare_loader.fetch_index_constituents(stock_pool)
            
            if not stock_list:
                self.logger.error(f"无法获取 {stock_pool} 股票列表")
                return False
            
            self.logger.info(f"股票池: {stock_pool}, 股票数量: {len(stock_list)}")
            
            # ===== 优化：使用按日期模式获取日线数据（大幅减少API调用）=====
            # 日更场景下，按日期获取更高效：每个交易日1次API调用
            # 而按股票获取：每只股票2次API调用（daily + adj_factor）
            fetch_mode = data_config.get("fetch_mode", "by_date")  # by_date / by_stock
            
            if fetch_mode == "by_date":
                self.logger.info(f"📊 使用【按日期】高效模式获取日线数据")
                self.ohlcv_data = self.tushare_loader.fetch_daily_range_optimized(
                    start_date, end_date, stock_list, show_progress=True
                )
            else:
                # 兼容旧模式
                self.logger.info(f"📊 使用【按股票】传统模式获取日线数据")
                self.ohlcv_data = self.tushare_loader.fetch_daily_data_batch(
                    stock_list, start_date, end_date
                )
            
            if self.ohlcv_data is None or self.ohlcv_data.empty:
                self.logger.error("未获取到任何 OHLCV 数据")
                return False
            
            self.logger.info(f"OHLCV 数据更新完成，共 {len(self.ohlcv_data)} 条记录")
            
            # 保存数据
            self.ohlcv_data.to_parquet(ohlcv_path)
            self._current_stock_list = stock_list
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新市场数据失败: {e}")
            return False
    
    def update_financial_data(self) -> bool:
        """更新财务数据"""
        self.logger.info("开始更新财务数据...")
        
        financial_path = DATA_RAW_PATH / f"financial_{self.today.strftime('%Y%m%d')}.parquet"
        if financial_path.exists():
            try:
                self.financial_data = pd.read_parquet(financial_path)
                if not self.financial_data.empty:
                    self.logger.info(f"使用缓存数据: {financial_path.name}")
                    return True
            except Exception as e:
                self.logger.warning(f"读取缓存失败: {e}")
        
        try:
            if self.ohlcv_data is None:
                return False
            
            # 获取唯一股票列表
            stocks = list(set(self.ohlcv_data['stock_code'].unique().tolist()))
            
            # ===== 优化：每日基础指标已经是高效模式（1次API调用获取全市场）=====
            # fetch_daily_basic 是按日期获取，非常高效
            basic_df = self.tushare_loader.fetch_daily_basic(stock_list=stocks)
            
            # 立即重置索引，避免后续操作的索引问题
            if basic_df is not None and not basic_df.empty:
                basic_df = basic_df.copy().reset_index(drop=True)
            
            # ===== 财务指标获取策略 =====
            # fina_indicator 需要逐只股票获取，但有7天缓存
            # 如果缓存命中率高，实际API调用会很少
            data_config = self.config.get("data", {})
            skip_fina_indicator = data_config.get("skip_fina_indicator", False)
            
            if skip_fina_indicator:
                # 跳过财务指标获取（仅使用 daily_basic 的估值数据）
                self.logger.info("跳过财务指标获取（使用 daily_basic 估值数据）")
                fina_df = pd.DataFrame()
            else:
                # 获取财务指标（有缓存保护，使用配置的批次参数）
                fina_batch_size = data_config.get("fina_batch_size", 300)
                fina_batch_sleep = data_config.get("fina_batch_sleep", 0.0)  # 默认不休息
                self.logger.info(f"📈 财务指标获取参数: batch_size={fina_batch_size}, batch_sleep={fina_batch_sleep}s")
                fina_df = self.tushare_loader.fetch_financial_batch(
                    stocks, 
                    show_progress=True,
                    batch_size=fina_batch_size,
                    batch_sleep=fina_batch_sleep
                )
                
                # 立即重置索引
                if fina_df is not None and not fina_df.empty:
                    fina_df = fina_df.copy().reset_index(drop=True)
            
            # 防御性处理：确保 DataFrame 有正确的结构
            if basic_df is None:
                basic_df = pd.DataFrame()
            if fina_df is None:
                fina_df = pd.DataFrame()
            
            # 重置索引，避免索引冲突 - 使用 RangeIndex 强制唯一索引
            if not basic_df.empty:
                basic_df = basic_df.copy()
                basic_df.index = pd.RangeIndex(len(basic_df))
            if not fina_df.empty:
                fina_df = fina_df.copy()
                fina_df.index = pd.RangeIndex(len(fina_df))
            
            if not basic_df.empty and not fina_df.empty:
                # 确保两边都去重
                basic_df_dedup = basic_df.drop_duplicates(subset=['stock_code'], keep='last')
                basic_df_dedup.index = pd.RangeIndex(len(basic_df_dedup))
                
                # 检查 fina_df 是否包含需要的列
                fina_cols = ['stock_code']
                if 'roe' in fina_df.columns:
                    fina_cols.append('roe')
                
                # 先复制再选择列，避免索引问题
                fina_subset = fina_df[fina_cols].copy()
                fina_subset.index = pd.RangeIndex(len(fina_subset))
                fina_df_dedup = fina_subset.drop_duplicates(subset=['stock_code'], keep='last')
                fina_df_dedup.index = pd.RangeIndex(len(fina_df_dedup))
                
                merged_df = basic_df_dedup.merge(
                    fina_df_dedup,
                    on='stock_code',
                    how='left'
                )
                # 删除重复的列名
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]
                self.logger.info(f"合并财务数据: basic={len(basic_df_dedup)}, fina={len(fina_df_dedup)}, merged={len(merged_df)}")
            elif not basic_df.empty:
                merged_df = basic_df.drop_duplicates(subset=['stock_code'], keep='last')
                merged_df.index = pd.RangeIndex(len(merged_df))
            elif not fina_df.empty:
                merged_df = fina_df.drop_duplicates(subset=['stock_code'], keep='last')
                merged_df.index = pd.RangeIndex(len(merged_df))
            else:
                return False
            
            self.financial_data = merged_df
            self.financial_data.to_parquet(financial_path)
            
            self.logger.info(f"财务数据更新完成: {len(self.financial_data)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"财务数据获取失败: {e}")
            return False
    
    def update_benchmark_data(self) -> bool:
        """更新基准指数数据（用于大盘风控）"""
        self.logger.info("开始更新基准指数数据...")
        
        try:
            risk_config = self.config.get("risk", {}).get("market_risk", {})
            benchmark_code = risk_config.get("benchmark", "000300")
            
            end_date = self.today.strftime("%Y%m%d")
            start_date = (self.today - timedelta(days=120)).strftime("%Y%m%d")
            
            self.benchmark_data = self.tushare_loader.fetch_index_daily(
                benchmark_code, start_date, end_date
            )
            
            if self.benchmark_data is not None and not self.benchmark_data.empty:
                self.logger.info(f"基准指数数据更新完成: {len(self.benchmark_data)} 条")
                return True
            else:
                self.logger.warning("未获取到基准指数数据，大盘风控可能不生效")
                return False
                
        except Exception as e:
            self.logger.warning(f"基准指数数据获取失败: {e}")
            return False
    
    def calculate_factors(self) -> bool:
        """计算因子数据"""
        self.logger.info("开始计算因子数据...")
        
        try:
            if self.ohlcv_data is None or self.ohlcv_data.empty:
                self.logger.error("OHLCV 数据为空，无法计算因子")
                return False
            
            # ========================================
            # 数据单位一致性检查与自动修正
            # ========================================
            self.ohlcv_data = self._validate_and_fix_data_units(self.ohlcv_data, "ohlcv")
            
            # 合并 OHLCV 和财务数据
            df = self.ohlcv_data.copy()
            
            if self.financial_data is not None and not self.financial_data.empty:
                self.financial_data = self._validate_and_fix_data_units(self.financial_data, "financial")
                
                # ========================================
                # 财务数据日期对齐（修复前视偏差）
                # ========================================
                # 财务数据通常是截面数据（某一天的快照），需要按日期对齐
                # 策略：使用 OHLCV 数据中每个 trade_date 对应的财务数据
                # 如果财务数据有 trade_date，则按 (stock_code, trade_date) 合并
                # 否则，只取最新财务数据广播到最新交易日
                
                fin_df = self.financial_data.copy()
                
                if 'trade_date' in fin_df.columns:
                    # 财务数据有日期，按 (stock_code, trade_date) 精确合并
                    df = df.merge(
                        fin_df,
                        on=['stock_code', 'trade_date'],
                        how='left',
                        suffixes=('', '_fin')
                    )
                    self.logger.info("财务数据已按 (stock_code, trade_date) 精确合并")
                else:
                    # 财务数据无日期，使用最新快照
                    # 只对 OHLCV 中最新交易日的数据合并财务
                    latest_trade_date = df['trade_date'].max()
                    
                    # 为财务数据添加标记，表示仅适用于最新日期
                    fin_df_dedup = fin_df.drop_duplicates(subset=['stock_code'], keep='last')
                    
                    # 只对最新日期的数据合并财务字段
                    df_latest = df[df['trade_date'] == latest_trade_date].copy()
                    df_historical = df[df['trade_date'] != latest_trade_date].copy()
                    
                    df_latest = df_latest.merge(
                        fin_df_dedup,
                        on='stock_code',
                        how='left',
                        suffixes=('', '_fin')
                    )
                    
                    # 历史数据不合并财务（避免前视偏差）
                    # 但需要确保列一致
                    for col in fin_df_dedup.columns:
                        if col != 'stock_code' and col not in df_historical.columns:
                            df_historical[col] = np.nan
                    
                    df = pd.concat([df_historical, df_latest], ignore_index=True)
                    self.logger.warning(
                        f"财务数据无 trade_date 列，仅合并到最新交易日 {latest_trade_date}，"
                        f"历史日期财务字段为 NaN（避免前视偏差）"
                    )
            
            # 计算技术因子（按股票分组）
            factor_dfs = []
            
            for stock_code, group in df.groupby('stock_code'):
                group = group.sort_values('trade_date')
                
                # RSI
                if 'close' in group.columns:
                    delta = group['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = (-delta).where(delta < 0, 0)
                    
                    avg_gain = gain.ewm(alpha=1/20, min_periods=20).mean()
                    avg_loss = loss.ewm(alpha=1/20, min_periods=20).mean()
                    rs = avg_gain / avg_loss.replace(0, np.nan)
                    group['rsi_20'] = 100 - (100 / (1 + rs))
                
                # 换手率5日均值（兼容 turnover_rate 和 turn 列名）
                turn_col = 'turn' if 'turn' in group.columns else ('turnover_rate' if 'turnover_rate' in group.columns else None)
                if turn_col:
                    group['turnover_5d'] = group[turn_col].rolling(5).mean()
                
                # 20日收益率（动量）
                if 'close' in group.columns:
                    group['return_20'] = group['close'].pct_change(20)
                    
                    # 路径效率
                    abs_changes = group['close'].diff().abs().rolling(20).sum()
                    net_change = group['close'].diff(20).abs()
                    group['efficiency_20'] = net_change / abs_changes.replace(0, np.nan)
                    
                    # 夏普比率（简化版）
                    returns = group['close'].pct_change()
                    group['sharpe_20'] = (
                        returns.rolling(20).mean() / returns.rolling(20).std().replace(0, np.nan)
                    ) * np.sqrt(252)
                    
                    # 波动率（年化，20日）
                    group['volatility_20'] = returns.rolling(20).std().replace(0, np.nan) * np.sqrt(252)
                
                # ===== Alpha 原子因子（用于 momentum_composite_zscore）=====
                # alpha_002: 价格振幅因子
                if {'high', 'low', 'close'}.issubset(group.columns):
                    group['alpha_002'] = (group['high'] - group['low']) / group['close'].replace(0, np.nan)
                    
                    # alpha_005: 尾盘强度因子
                    range_hl = (group['high'] - group['low']).replace(0, np.nan)
                    group['alpha_005'] = (group['close'] - group['low']) / range_hl
                
                # alpha_003: 量价背离因子（5日）
                if {'close', 'volume'}.issubset(group.columns):
                    price_change = group['close'].pct_change(5)
                    volume_change = group['volume'].pct_change(5)
                    group['alpha_003'] = price_change - volume_change
                
                factor_dfs.append(group)
            
            self.factor_data = pd.concat(factor_dfs, ignore_index=True)
            
            # ========================================
            # Z-Score 标准化 - 按交易日横截面（修复前视偏差）
            # ========================================
            zscore_cols = {
                'rsi_20': 'rsi_20_zscore',
                'turnover_5d': 'turnover_5d_zscore',
                'return_20': 'momentum_zscore',
                'sharpe_20': 'sharpe_20_zscore',
                'efficiency_20': 'efficiency_20_zscore',
                'volatility_20': 'volatility_20_zscore',
                'alpha_002': 'alpha_002_zscore',
                'alpha_003': 'alpha_003_zscore',
                'alpha_005': 'alpha_005_zscore',
                'pe_ttm': 'pe_ttm_zscore',
                'pb': 'pb_zscore',
            }
            
            def cross_sectional_zscore(group: pd.DataFrame, col: str) -> pd.Series:
                """按交易日横截面计算Z-Score（消除前视偏差）"""
                values = group[col]
                mean = values.mean()
                std = values.std()
                if std > 0 and not pd.isna(std):
                    return (values - mean) / std
                return pd.Series(0.0, index=group.index)
            
            for src_col, dst_col in zscore_cols.items():
                if src_col in self.factor_data.columns:
                    # 估值类字段：过滤非正值，避免 log / 比率异常污染横截面
                    if src_col in ('pe_ttm', 'pb'):
                        self.factor_data[src_col] = pd.to_numeric(self.factor_data[src_col], errors='coerce')
                        self.factor_data.loc[self.factor_data[src_col] <= 0, src_col] = np.nan
                    # 按 trade_date 分组，每个交易日内部横截面标准化
                    self.factor_data[dst_col] = self.factor_data.groupby('trade_date', group_keys=False).apply(
                        lambda g: cross_sectional_zscore(g, src_col), include_groups=False
                    )
                    self.logger.debug(f"横截面标准化完成: {src_col} -> {dst_col}")
            
            # 小市值因子（同样按交易日横截面）
            if 'circ_mv' in self.factor_data.columns:
                def small_cap_zscore(group: pd.DataFrame) -> pd.Series:
                    """小市值因子：对数市值的负Z-Score"""
                    log_mv = np.log(group['circ_mv'].replace(0, np.nan))
                    mean = log_mv.mean()
                    std = log_mv.std()
                    if std > 0 and not pd.isna(std):
                        return -(log_mv - mean) / std
                    return pd.Series(0.0, index=group.index)
                
                self.factor_data['small_cap_zscore'] = self.factor_data.groupby('trade_date', group_keys=False).apply(
                    small_cap_zscore, include_groups=False
                )
                self.logger.debug("横截面标准化完成: circ_mv -> small_cap_zscore")
            
            # ========================================
            # 复合因子（与 strategy_config.yaml 对齐）
            # ========================================
            # quality_composite_zscore = 50% turnover + 30% low_vol + 20% efficiency
            if {'turnover_5d_zscore', 'volatility_20_zscore', 'efficiency_20_zscore'}.issubset(self.factor_data.columns):
                low_vol_z = -self.factor_data['volatility_20_zscore'].fillna(0.0)
                quality_raw = (
                    0.5 * self.factor_data['turnover_5d_zscore'].fillna(0.0)
                    + 0.3 * low_vol_z
                    + 0.2 * self.factor_data['efficiency_20_zscore'].fillna(0.0)
                )
                self.factor_data['quality_composite_raw'] = quality_raw
                self.factor_data['quality_composite_zscore'] = self.factor_data.groupby('trade_date', group_keys=False).apply(
                    lambda g: cross_sectional_zscore(g, 'quality_composite_raw'), include_groups=False
                )
                self.logger.info("复合因子已生成: quality_composite_zscore")
            else:
                self.logger.warning("缺少复合质量因子所需列，未生成 quality_composite_zscore")
            
            # momentum_composite_zscore = alpha_002/003/005 + efficiency（等权）
            if {'alpha_002_zscore', 'alpha_003_zscore', 'alpha_005_zscore', 'efficiency_20_zscore'}.issubset(self.factor_data.columns):
                mom_raw = (
                    0.25 * self.factor_data['alpha_002_zscore'].fillna(0.0)
                    + 0.25 * self.factor_data['alpha_003_zscore'].fillna(0.0)
                    + 0.25 * self.factor_data['alpha_005_zscore'].fillna(0.0)
                    + 0.25 * self.factor_data['efficiency_20_zscore'].fillna(0.0)
                )
                self.factor_data['momentum_composite_raw'] = mom_raw
                self.factor_data['momentum_composite_zscore'] = self.factor_data.groupby('trade_date', group_keys=False).apply(
                    lambda g: cross_sectional_zscore(g, 'momentum_composite_raw'), include_groups=False
                )
                self.logger.info("复合因子已生成: momentum_composite_zscore")
            else:
                self.logger.warning("缺少复合动量因子所需列，未生成 momentum_composite_zscore")
            
            # value_composite_zscore = 估值（低PE、低PB更好）
            if {'pe_ttm_zscore', 'pb_zscore'}.issubset(self.factor_data.columns):
                # zscore越低代表估值越低，因此取负号变成“高分=更便宜”
                value_raw = -(0.5 * self.factor_data['pe_ttm_zscore'].fillna(0.0) + 0.5 * self.factor_data['pb_zscore'].fillna(0.0))
                self.factor_data['value_composite_raw'] = value_raw
                self.factor_data['value_composite_zscore'] = self.factor_data.groupby('trade_date', group_keys=False).apply(
                    lambda g: cross_sectional_zscore(g, 'value_composite_raw'), include_groups=False
                )
                self.logger.info("复合因子已生成: value_composite_zscore")
            else:
                self.logger.warning("缺少估值字段(pe_ttm/pb)，未生成 value_composite_zscore")
            
            # 配置一致性检查（避免“策略引用列不存在”导致名义适配但实际无效）
            required_cols = {
                'value_col': self.strategy.value_col,
                'quality_col': self.strategy.quality_col,
                'momentum_col': self.strategy.momentum_col,
                'size_col': self.strategy.size_col,
            }
            missing = [k for k, v in required_cols.items() if v not in self.factor_data.columns]
            if missing:
                self.logger.warning(f"⚠️ 策略配置的因子列在factor_data中缺失: {missing} -> {[(k, required_cols[k]) for k in missing]}")
            
            # 保存因子数据
            factor_path = DATA_PROCESSED_PATH / f"factors_{self.today.strftime('%Y%m%d')}.parquet"
            self.factor_data.to_parquet(factor_path)
            
            self.logger.info(f"因子计算完成: {len(self.factor_data)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"因子计算失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def is_rebalance_day(self) -> bool:
        """判断今天是否是调仓日"""
        frequency = self.strategy.rebalance_frequency
        
        if frequency == "weekly":
            return self.today.dayofweek == 4  # 周五
        else:  # monthly
            next_day = self.today + timedelta(days=1)
            return self.today.month != next_day.month
    
    def generate_target_positions(self) -> bool:
        """生成目标持仓"""
        self.logger.info("生成目标持仓...")
        
        try:
            if self.factor_data is None or self.factor_data.empty:
                self.logger.error("因子数据为空")
                return False
            
            # 过滤当日数据
            latest_date = pd.to_datetime(self.factor_data['trade_date']).max()
            day_data = self.factor_data[
                pd.to_datetime(self.factor_data['trade_date']) == latest_date
            ]
            
            if day_data.empty:
                self.logger.error("当日数据为空")
                return False

            # 标记是否为当前持仓（用于持股惯性加分/解释）
            if 'stock_code' in day_data.columns:
                holding_set = set(self.current_positions.keys())
                day_data = day_data.copy()
                day_data['is_holding'] = day_data['stock_code'].astype(str).isin(holding_set)

            # 计算 IC 监控（用于报告 + 因子在线自适应）
            ic_results = self._compute_ic_results()
            if hasattr(self.report_generator, "set_ic_results") and ic_results is not None and not ic_results.empty:
                self.report_generator.set_ic_results(ic_results)
            
            # 应用市场状态自适应权重（若启用）
            try:
                if hasattr(self.strategy, 'apply_adaptive_weights') and self.benchmark_data is not None:
                    self.strategy.apply_adaptive_weights(index_data=self.benchmark_data, date=latest_date)
            except Exception as e:
                self.logger.warning(f"自适应权重应用失败（忽略并降级）: {e}")

            # IC 熔断：弱预测力因子自动降权（避免“失效因子拖累”）
            try:
                ic_cfg = self.config.get("ic_monitor", {})
                if ic_cfg.get("enabled", False) and ic_cfg.get("circuit_breaker_enabled", False):
                    if hasattr(self.strategy, "apply_factor_circuit_breaker") and ic_results is not None and not ic_results.empty:
                        self.strategy.apply_factor_circuit_breaker(
                            ic_results=ic_results,
                            ic_threshold=float(ic_cfg.get("circuit_breaker_ic_threshold", 0.005)),
                            ir_threshold=float(ic_cfg.get("circuit_breaker_ir_threshold", 0.2))
                        )
            except Exception as e:
                self.logger.warning(f"因子熔断应用失败（忽略并降级）: {e}")

            # IC 方向校准：IC 为负时自动反向使用（把“反向预测力”转成 alpha）
            try:
                dir_cfg = self.config.get("ic_monitor", {}).get("directional_adjustment", {})
                if dir_cfg.get("enabled", True):
                    if hasattr(self.strategy, "apply_factor_direction_from_ic") and ic_results is not None and not ic_results.empty:
                        self.strategy.apply_factor_direction_from_ic(
                            ic_results=ic_results,
                            abs_ic_threshold=float(dir_cfg.get("abs_ic_threshold", 0.02)),
                            ir_threshold=float(dir_cfg.get("ir_threshold", 0.3)),
                            positive_ratio_threshold=float(dir_cfg.get("positive_ratio_threshold", 0.55))
                        )
            except Exception as e:
                self.logger.warning(f"因子方向校准失败（忽略并降级）: {e}")
            
            # 实盘可交易性过滤（涨跌停/一字板/流动性/ST等）
            try:
                if hasattr(self.strategy, 'filter_stocks'):
                    filtered_day_data = self.strategy.filter_stocks(day_data, date=latest_date)
                else:
                    filtered_day_data = day_data
            except Exception as e:
                self.logger.warning(f"过滤器执行失败（忽略并降级）: {e}")
                filtered_day_data = day_data

            # 行业映射（用于行业分散/黑名单；若失败自动降级）
            try:
                industry_cfg = self.config.get("strategy", {}).get("industry_constraints", {})
                if industry_cfg.get("enabled", False) and 'stock_code' in filtered_day_data.columns:
                    source = str(industry_cfg.get("source", "tushare_industry"))
                    if source.lower().startswith("sw"):
                        level = int(industry_cfg.get("sw_level", 1))
                        industry_map = self.tushare_loader.fetch_sw_industry_mapping(level=level)
                    else:
                        industry_map = self.tushare_loader.fetch_industry_mapping(use_cache=True)

                    if industry_map:
                        filtered_day_data = filtered_day_data.copy()
                        filtered_day_data['industry'] = filtered_day_data['stock_code'].astype(str).map(industry_map)
            except Exception as e:
                self.logger.warning(f"行业映射失败（忽略并降级）: {e}")
            
            # 选股
            selected_stocks = self.strategy.select_top_stocks(
                filtered_day_data,
                n=self.strategy.top_n,
                date=latest_date
            )
            
            if not selected_stocks:
                self.logger.warning("未选出任何股票")
                self.target_positions = {}
                return True

            # 选股解释数据（用于报告展示：分数分解、关键因子值）
            self._selection_details = self._build_selection_details(
                day_data=filtered_day_data,
                selected_stocks=selected_stocks
            )
            
            # 生成等权重持仓
            portfolio_config = self.config.get("portfolio", {})
            total_capital = portfolio_config.get("total_capital", 300000)

            # 动态仓位：弱市/高波动留现金（position_scale 由策略自适应模块提供）
            try:
                pos_scale = self.strategy.get_position_scale() if hasattr(self.strategy, "get_position_scale") else 1.0
            except Exception:
                pos_scale = 1.0

            invested_capital = float(total_capital) * float(np.clip(pos_scale, 0.0, 1.0))
            weight = invested_capital / len(selected_stocks)
            
            self.target_positions = {
                stock: weight
                for stock in selected_stocks
            }
            
            self.logger.info(
                f"目标持仓生成完成: {len(self.target_positions)} 只股票, "
                f"仓位系数 {pos_scale:.0%}, 每只约 ¥{weight:,.0f}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成目标持仓失败: {e}")
            return False

    def _build_selection_details(
        self,
        day_data: pd.DataFrame,
        selected_stocks: List[str]
    ) -> pd.DataFrame:
        """
        构建选股打分明细（用于报告解释）
        
        Parameters
        ----------
        day_data : pd.DataFrame
            当日数据（已过滤后的横截面）
        selected_stocks : List[str]
            入选股票列表
        
        Returns
        -------
        pd.DataFrame
            选股明细表
        """
        if day_data is None or day_data.empty:
            return pd.DataFrame()
        
        stock_col = 'stock_code' if 'stock_code' in day_data.columns else None
        if stock_col is None:
            return pd.DataFrame()
        
        # 先做日级去重，避免同一股票多行影响归一化
        full_df = day_data.copy()
        if 'trade_date' in full_df.columns:
            full_df = full_df.sort_values('trade_date', ascending=False)
        full_df = full_df.drop_duplicates(subset=[stock_col], keep='first')
        
        # ===== 从缓存读取情绪（不额外调用LLM）=====
        date_str = None
        if 'trade_date' in full_df.columns:
            try:
                date_str = pd.to_datetime(full_df['trade_date']).max().strftime('%Y-%m-%d')
            except Exception:
                date_str = None
        if date_str is None:
            date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        sentiment_df = self._load_sentiment_cache_for_date(date_str)
        sentiment_scores: Optional[pd.Series] = None
        if not sentiment_df.empty:
            sentiment_scores = pd.Series(
                sentiment_df['score'].values,
                index=sentiment_df['stock_code'].astype(str).values
            )
        
        # 分项贡献：用“同一批候选横截面”做归一化，避免只在入选集合上缩放
        try:
            score_ret = self.strategy.calculate_total_score(
                full_df,
                sentiment_scores=sentiment_scores,
                return_components=True
            )
            total_score, components = score_ret  # type: ignore[misc]
        except Exception as e:
            self.logger.warning(f"选股分数分解失败: {e}")
            total_score = pd.Series(0.0, index=full_df.index)
            components = {}
        
        full_df['base_score'] = total_score
        for name, series in components.items():
            full_df[f'contrib_{name}'] = series
        
        # 绑定缓存中的情绪分与置信度（原始值，便于解释）
        if not sentiment_df.empty:
            tmp = sentiment_df.copy()
            tmp['stock_code'] = tmp['stock_code'].astype(str)
            full_df[stock_col] = full_df[stock_col].astype(str)
            full_df = full_df.merge(
                tmp[['stock_code', 'score', 'confidence', 'category', 'summary']],
                left_on=stock_col,
                right_on='stock_code',
                how='left',
                suffixes=('', '_sent')
            )
            full_df = full_df.rename(columns={
                'score': 'sentiment_score',
                'confidence': 'sentiment_confidence',
                'category': 'sentiment_category',
                'summary': 'sentiment_summary',
            })
        
        sel_df = full_df[full_df[stock_col].astype(str).isin([str(s) for s in selected_stocks])].copy()
        if sel_df.empty:
            return pd.DataFrame()
        
        # 关键列（若存在则保留）
        keep_cols = [
            'trade_date', 'stock_code', 'name', 'close', 'amount', 'pct_change',
            'industry',
            self.strategy.value_col, self.strategy.quality_col, self.strategy.momentum_col, self.strategy.size_col,
            'quality_composite_zscore', 'momentum_composite_zscore', 'value_composite_zscore',
            'turnover_5d_zscore', 'volatility_20_zscore', 'efficiency_20_zscore',
            'sentiment_score', 'sentiment_confidence', 'sentiment_category',
            'is_holding', 'base_score',
            'contrib_value', 'contrib_quality', 'contrib_momentum', 'contrib_size',
            'contrib_sentiment',
        ]
        existing_cols = [c for c in keep_cols if c in sel_df.columns]
        sel_df = sel_df[existing_cols]
        
        # 统一为字符串代码并排序
        sel_df['stock_code'] = sel_df['stock_code'].astype(str)
        sel_df = sel_df.sort_values('base_score', ascending=False, kind='mergesort')
        sel_df = sel_df.drop_duplicates(subset=['stock_code'], keep='first')
        
        return sel_df.reset_index(drop=True)

    def _load_sentiment_cache_for_date(self, date_str: str) -> pd.DataFrame:
        """
        从 sentiment_cache.json 读取指定日期的情绪结果（不触发LLM请求）
        
        缓存键格式: "{stock_code}_{YYYY-MM-DD}"
        
        Parameters
        ----------
        date_str : str
            日期字符串 (YYYY-MM-DD)
        
        Returns
        -------
        pd.DataFrame
            列: stock_code, score, confidence, category, summary
        """
        llm_cfg = self.config.get("llm", {})
        cache_path = llm_cfg.get("cache_path", "data/processed/sentiment_cache.json")
        path = Path(cache_path)
        if not path.exists():
            return pd.DataFrame()
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            self.logger.warning(f"读取情绪缓存失败: {e}")
            return pd.DataFrame()
        
        records: List[Dict[str, Any]] = []
        suffix = f"_{date_str}"
        for k, v in raw.items():
            if not isinstance(k, str) or not k.endswith(suffix):
                continue
            stock_code = k[: -len(suffix)]
            if not stock_code:
                continue
            if not isinstance(v, dict):
                continue
            records.append({
                "stock_code": str(stock_code),
                "score": float(v.get("score", 0.0)),
                "confidence": float(v.get("confidence", 0.0)),
                "category": str(v.get("category", "")),
                "summary": str(v.get("summary", "")),
            })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["stock_code"], keep="last")
        return df
    
    def calculate_trade_orders(self) -> tuple:
        """
        计算交易订单（增强版）
        
        功能：
        1. 应用换仓缓冲带（减少不必要交易）
        2. 可执行性检查（停牌/涨跌停/流动性）
        3. 交易成本估算
        4. 执行优先级排序（先卖后买）
        
        Returns
        -------
        tuple
            (buy_orders, sell_orders, order_details)
        """
        buy_orders: Dict[str, float] = {}
        sell_orders: Dict[str, float] = {}
        
        # 获取配置参数
        strategy_config = self.config.get("strategy", {})
        trading_config = self.config.get("trading", {})
        portfolio_config = self.config.get("portfolio", {})
        
        rebalance_buffer = strategy_config.get("rebalance_buffer", 0.05)
        min_trade_amount = trading_config.get("min_trade_amount", 5000)  # 最小交易金额
        commission_rate = trading_config.get("commission_rate", 0.0003)
        stamp_duty = trading_config.get("stamp_duty", 0.001)
        min_commission = 5.0  # A股最低佣金
        total_capital = portfolio_config.get("total_capital", 300000)
        
        # 获取当日行情数据用于可执行性检查
        latest_data = self._get_latest_market_data()
        
        # 初始化订单详情
        self._order_details: Dict[str, Dict[str, Any]] = {}
        
        # ========================================
        # Step 1: 计算原始差额
        # ========================================
        raw_buy_orders: Dict[str, float] = {}
        raw_sell_orders: Dict[str, float] = {}
        
        # 卖出：当前持有但目标不持有或需要减仓的股票
        for stock, current_amount in self.current_positions.items():
            target_amount = self.target_positions.get(stock, 0)
            if target_amount < current_amount:
                raw_sell_orders[stock] = current_amount - target_amount
        
        # 买入：目标持有但当前不持有或需要加仓的股票
        for stock, target_amount in self.target_positions.items():
            current_amount = self.current_positions.get(stock, 0)
            if target_amount > current_amount:
                raw_buy_orders[stock] = target_amount - current_amount
        
        # ========================================
        # Step 2: 应用换仓缓冲带
        # ========================================
        for stock, amount in raw_sell_orders.items():
            current_amount = self.current_positions.get(stock, 0)
            if current_amount > 0:
                drift_ratio = amount / current_amount
                if drift_ratio > rebalance_buffer:
                    sell_orders[stock] = amount
                    self.logger.debug(f"卖出 {stock}: 偏移 {drift_ratio:.1%} > 缓冲 {rebalance_buffer:.1%}")
                else:
                    self.logger.debug(f"跳过卖出 {stock}: 偏移 {drift_ratio:.1%} <= 缓冲 {rebalance_buffer:.1%}")
            else:
                sell_orders[stock] = amount
        
        for stock, amount in raw_buy_orders.items():
            target_amount = self.target_positions.get(stock, 0)
            if target_amount > 0:
                drift_ratio = amount / target_amount
                # 买入使用更宽松的缓冲（新建仓位除外）
                current_amount = self.current_positions.get(stock, 0)
                if current_amount == 0 or drift_ratio > rebalance_buffer:
                    buy_orders[stock] = amount
                else:
                    self.logger.debug(f"跳过买入 {stock}: 偏移 {drift_ratio:.1%} <= 缓冲 {rebalance_buffer:.1%}")
            else:
                buy_orders[stock] = amount
        
        # ========================================
        # Step 3: 过滤最小交易金额
        # ========================================
        buy_orders = {k: v for k, v in buy_orders.items() if v >= min_trade_amount}
        sell_orders = {k: v for k, v in sell_orders.items() if v >= min_trade_amount}
        
        # ========================================
        # Step 4: 可执行性检查与交易成本估算
        # ========================================
        for stock, amount in {**sell_orders, **buy_orders}.items():
            is_buy = stock in buy_orders
            detail = self._check_executability(stock, amount, is_buy, latest_data)
            detail['amount'] = amount
            detail['side'] = 'BUY' if is_buy else 'SELL'
            
            # 估算交易成本
            cost = self._estimate_trade_cost(amount, is_buy, commission_rate, stamp_duty, min_commission)
            detail.update(cost)
            
            self._order_details[stock] = detail
        
        # ========================================
        # Step 5: 执行优先级排序（先卖后买，按流动性排序）
        # ========================================
        # 卖出按流动性从高到低（优先卖出流动性好的）
        sell_orders = dict(sorted(
            sell_orders.items(),
            key=lambda x: self._order_details.get(x[0], {}).get('daily_amount', 0),
            reverse=True
        ))
        
        # 买入按流动性从高到低
        buy_orders = dict(sorted(
            buy_orders.items(),
            key=lambda x: self._order_details.get(x[0], {}).get('daily_amount', 0),
            reverse=True
        ))
        
        # 统计
        executable_count = sum(1 for d in self._order_details.values() if d.get('is_executable', True))
        total_count = len(self._order_details)
        self.logger.info(
            f"交易订单计算完成: 买入 {len(buy_orders)} 只, 卖出 {len(sell_orders)} 只, "
            f"可执行 {executable_count}/{total_count}"
        )
        
        return buy_orders, sell_orders
    
    def _get_latest_market_data(self) -> Dict[str, Dict[str, Any]]:
        """获取最新市场数据用于可执行性检查"""
        result = {}
        
        if self.factor_data is None or self.factor_data.empty:
            return result
        
        latest_date = pd.to_datetime(self.factor_data['trade_date']).max()
        latest_df = self.factor_data[
            pd.to_datetime(self.factor_data['trade_date']) == latest_date
        ]
        
        for _, row in latest_df.iterrows():
            stock = str(row.get('stock_code', ''))
            if not stock:
                continue
            
            result[stock] = {
                'close': row.get('close', 0),
                'high': row.get('high', 0),
                'low': row.get('low', 0),
                'open': row.get('open', 0),
                'volume': row.get('volume', 0),
                'amount': row.get('amount', 0),
                'pct_change': row.get('pct_change', row.get('pctChg', 0)),
                'turnover_rate': row.get('turnover_rate', 0),
                'name': row.get('name', row.get('stock_name', '')),
                'is_suspended': row.get('is_suspended', False),
            }
        
        return result
    
    def _check_executability(
        self,
        stock: str,
        amount: float,
        is_buy: bool,
        market_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        检查订单可执行性
        
        Returns
        -------
        Dict[str, Any]
            包含可执行性状态和原因
        """
        result = {
            'is_executable': True,
            'executability_issues': [],
            'daily_amount': 0,
            'impact_ratio': 0,
        }
        
        data = market_data.get(stock, {})
        
        if not data:
            result['is_executable'] = True  # 无数据时默认可执行，人工确认
            result['executability_issues'].append('⚠️ 无最新行情数据')
            return result
        
        daily_amount = data.get('amount', 0)
        result['daily_amount'] = daily_amount
        
        # 1. 停牌检查
        if data.get('is_suspended', False) or data.get('volume', 0) == 0:
            result['is_executable'] = False
            result['executability_issues'].append('🚫 停牌')
        
        # 2. 涨跌停检查
        pct_change = data.get('pct_change', 0)
        high = data.get('high', 0)
        low = data.get('low', 0)
        close = data.get('close', 0)
        
        # 判断是否涨停（涨幅>=9.5% 且 最高=最低=收盘，或者涨幅>=9.5%且是买入）
        if pct_change >= 9.5:
            if high == low == close:
                result['is_executable'] = False
                result['executability_issues'].append('🔴 一字涨停(无法买入)')
            elif is_buy:
                result['executability_issues'].append('⚠️ 涨停(可能无法买入)')
        
        # 判断是否跌停
        if pct_change <= -9.5:
            if high == low == close:
                result['is_executable'] = False
                result['executability_issues'].append('🟢 一字跌停(无法卖出)')
            elif not is_buy:
                result['executability_issues'].append('⚠️ 跌停(可能无法卖出)')
        
        # 3. 流动性检查
        if daily_amount > 0:
            impact_ratio = amount / daily_amount
            result['impact_ratio'] = impact_ratio

            trading_cfg = self.config.get("trading", {})
            max_impact_ratio = float(trading_cfg.get("max_impact_ratio", 0.10))
            warn_impact_ratio = min(0.05, max_impact_ratio)

            if impact_ratio > warn_impact_ratio:
                result['executability_issues'].append(f'⚠️ 冲击成本高({impact_ratio:.1%})')
            if impact_ratio > max_impact_ratio:
                result['is_executable'] = False
                result['executability_issues'].append(f'🚫 流动性不足({impact_ratio:.1%})')
        else:
            result['executability_issues'].append('⚠️ 无成交额数据')
        
        # 4. ST/退市检查
        name = data.get('name', '')
        st_keywords = ('ST', '*ST', '退', 'S', 'PT')
        if any(kw in str(name) for kw in st_keywords):
            result['executability_issues'].append('⚠️ ST/退市风险')
        
        return result
    
    def _estimate_trade_cost(
        self,
        amount: float,
        is_buy: bool,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.001,
        min_commission: float = 5.0
    ) -> Dict[str, float]:
        """
        估算交易成本
        
        Returns
        -------
        Dict[str, float]
            交易成本明细
        """
        # 佣金（最低5元）
        commission = max(amount * commission_rate, min_commission)
        
        # 印花税（仅卖出）
        stamp = amount * stamp_duty if not is_buy else 0
        
        # 滑点（假设0.1%）
        slippage_rate = self.config.get("trading", {}).get("slippage", 0.001)
        slippage = amount * slippage_rate
        
        total_cost = commission + stamp + slippage
        
        return {
            'commission': commission,
            'stamp_duty': stamp,
            'slippage': slippage,
            'total_cost': total_cost,
            'cost_rate': total_cost / amount if amount > 0 else 0,
        }
    
    def generate_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        format: str = "markdown"
    ) -> str:
        """生成增强版报告（含交易成本、可执行性、风控信息）"""
        strategy_info = {
            'name': self.strategy.name,
            'value_weight': self.strategy.value_weight,
            'quality_weight': self.strategy.quality_weight,
            'momentum_weight': self.strategy.momentum_weight,
            'size_weight': getattr(self.strategy, 'size_weight', 0),
            'top_n': self.strategy.top_n,
        }
        
        # 获取订单详情
        order_details = getattr(self, '_order_details', {})
        selection_details = getattr(self, '_selection_details', pd.DataFrame())
        
        # 获取风控状态
        risk_status = self._get_risk_status()
        
        # 生成增强版报告
        return self._generate_enhanced_report(
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            target_positions=self.target_positions,
            strategy_info=strategy_info,
            order_details=order_details,
            selection_details=selection_details,
            risk_status=risk_status,
            report_date=self.today.strftime('%Y-%m-%d'),
            format=format
        )
    
    def _get_risk_status(self) -> Dict[str, Any]:
        """获取风控状态"""
        risk_status = {
            'market_risk_triggered': False,
            'market_risk_reason': '',
            'factor_breaker_triggered': [],
            'position_drift': 0.0,
        }
        
        # 检查大盘风控
        risk_config = self.config.get("risk", {}).get("market_risk", {})
        if risk_config.get("enabled", True) and self.benchmark_data is not None:
            try:
                benchmark_df = self.benchmark_data.copy()
                if not isinstance(benchmark_df.index, pd.DatetimeIndex):
                    if 'trade_date' in benchmark_df.columns:
                        benchmark_df['trade_date'] = pd.to_datetime(benchmark_df['trade_date'])
                        benchmark_df = benchmark_df.set_index('trade_date')
                
                benchmark_df = benchmark_df.sort_index()
                
                ma_period = risk_config.get("ma_period", 60)
                drop_threshold = risk_config.get("drop_threshold", 0.10)
                drop_lookback = risk_config.get("drop_lookback", 20)
                
                if len(benchmark_df) >= ma_period:
                    latest_close = benchmark_df['close'].iloc[-1]
                    ma_value = benchmark_df['close'].rolling(ma_period).mean().iloc[-1]
                    
                    # 检查是否跌破均线
                    below_ma = latest_close < ma_value
                    
                    # 检查回撤
                    if len(benchmark_df) >= drop_lookback:
                        lookback_high = benchmark_df['close'].iloc[-drop_lookback:].max()
                        drawdown = (latest_close - lookback_high) / lookback_high
                        
                        if below_ma and drawdown < -drop_threshold:
                            risk_status['market_risk_triggered'] = True
                            risk_status['market_risk_reason'] = (
                                f"指数跌破{ma_period}日均线 且 "
                                f"{drop_lookback}日回撤 {drawdown:.1%} < -{drop_threshold:.0%}"
                            )
                        else:
                            risk_status['market_risk_reason'] = (
                                f"指数{'低于' if below_ma else '高于'}{ma_period}日均线, "
                                f"回撤 {drawdown:.1%}"
                            )
            except Exception as e:
                self.logger.warning(f"风控状态检查失败: {e}")
        
        return risk_status
    
    def _generate_enhanced_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        target_positions: Dict[str, float],
        strategy_info: Dict[str, Any],
        order_details: Dict[str, Dict[str, Any]],
        selection_details: pd.DataFrame,
        risk_status: Dict[str, Any],
        report_date: str,
        format: str = "markdown"
    ) -> str:
        """生成增强版报告"""
        portfolio_config = self.config.get("portfolio", {})
        total_capital = portfolio_config.get("total_capital", 300000)
        
        # 计算汇总统计
        total_buy = sum(buy_orders.values())
        total_sell = sum(sell_orders.values())
        total_cost = sum(d.get('total_cost', 0) for d in order_details.values())
        executable_count = sum(1 for d in order_details.values() if d.get('is_executable', True))
        
        if format == "html":
            return self._generate_enhanced_html_report(
                buy_orders, sell_orders, target_positions, strategy_info,
                order_details, selection_details, risk_status, report_date, total_capital,
                total_buy, total_sell, total_cost, executable_count
            )
        else:
            return self._generate_enhanced_markdown_report(
                buy_orders, sell_orders, target_positions, strategy_info,
                order_details, selection_details, risk_status, report_date, total_capital,
                total_buy, total_sell, total_cost, executable_count
            )
    
    def _generate_enhanced_markdown_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        target_positions: Dict[str, float],
        strategy_info: Dict[str, Any],
        order_details: Dict[str, Dict[str, Any]],
        selection_details: pd.DataFrame,
        risk_status: Dict[str, Any],
        report_date: str,
        total_capital: float,
        total_buy: float,
        total_sell: float,
        total_cost: float,
        executable_count: int
    ) -> str:
        """生成增强版 Markdown 报告"""
        lines = [
            f"# 📊 每日调仓报告",
            f"",
            f"**报告日期**: {report_date}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
        ]
        
        # ========== 风控状态 ==========
        lines.extend([
            f"## 🛡️ 风控状态",
            f"",
        ])
        
        if risk_status.get('market_risk_triggered'):
            lines.extend([
                f"⚠️ **大盘风控触发**: {risk_status.get('market_risk_reason', '')}",
                f"",
                f"> 建议：降低仓位或暂停新开仓",
                f"",
            ])
        else:
            lines.extend([
                f"✅ 大盘风控未触发: {risk_status.get('market_risk_reason', '正常')}",
                f"",
            ])
        
        # ========== 策略概览 ==========
        lines.extend([
            f"## 📈 策略概览",
            f"",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 策略名称 | {strategy_info.get('name', 'N/A')} |",
            f"| 总资金 | ¥{total_capital:,.0f} |",
            f"| 目标持仓数 | {len(target_positions)} |",
            f"| 可执行订单 | {executable_count}/{len(order_details)} |",
            f"| 预估总成本 | ¥{total_cost:,.0f} ({total_cost/total_capital*100:.2f}%) |",
            f"",
        ])
        
        # ========== 卖出清单（先卖后买）==========
        lines.extend([
            f"## 📉 明日需卖出（按流动性排序，优先执行）",
            f"",
        ])
        
        if sell_orders:
            lines.extend([
                f"| 股票代码 | 卖出金额 | 佣金 | 印花税 | 总成本 | 可执行性 |",
                f"|----------|----------|------|--------|--------|----------|",
            ])
            for stock, amount in sell_orders.items():
                detail = order_details.get(stock, {})
                commission = detail.get('commission', 0)
                stamp = detail.get('stamp_duty', 0)
                cost = detail.get('total_cost', 0)
                issues = detail.get('executability_issues', [])
                exec_status = '✅' if detail.get('is_executable', True) else '🚫'
                if issues:
                    exec_status += ' ' + ' '.join(issues[:2])
                lines.append(f"| {stock} | ¥{amount:,.0f} | ¥{commission:.0f} | ¥{stamp:.0f} | ¥{cost:.0f} | {exec_status} |")
            lines.extend([
                f"",
                f"**卖出总金额**: ¥{total_sell:,.0f}",
                f"",
            ])
        else:
            lines.extend([f"*无需卖出*", f""])
        
        # ========== 买入清单 ==========
        lines.extend([
            f"## 📈 明日需买入（按流动性排序）",
            f"",
        ])
        
        if buy_orders:
            lines.extend([
                f"| 股票代码 | 买入金额 | 佣金 | 滑点 | 总成本 | 冲击比 | 可执行性 |",
                f"|----------|----------|------|------|--------|--------|----------|",
            ])
            for stock, amount in buy_orders.items():
                detail = order_details.get(stock, {})
                commission = detail.get('commission', 0)
                slippage = detail.get('slippage', 0)
                cost = detail.get('total_cost', 0)
                impact = detail.get('impact_ratio', 0)
                issues = detail.get('executability_issues', [])
                exec_status = '✅' if detail.get('is_executable', True) else '🚫'
                if issues:
                    exec_status += ' ' + ' '.join(issues[:2])
                lines.append(f"| {stock} | ¥{amount:,.0f} | ¥{commission:.0f} | ¥{slippage:.0f} | ¥{cost:.0f} | {impact:.1%} | {exec_status} |")
            lines.extend([
                f"",
                f"**买入总金额**: ¥{total_buy:,.0f}",
                f"",
            ])
        else:
            lines.extend([f"*无需买入*", f""])
        
        # ========== 目标持仓 ==========
        lines.extend([
            f"## 📋 目标持仓明细",
            f"",
            f"| 股票代码 | 目标金额 | 权重 |",
            f"|----------|----------|------|",
        ])
        
        total_target = sum(target_positions.values()) if target_positions else 1
        for stock, amount in sorted(target_positions.items(), key=lambda x: -x[1]):
            weight = amount / total_target
            lines.append(f"| {stock} | ¥{amount:,.0f} | {weight:.1%} |")

        # ========== 选股打分明细 ==========
        if selection_details is not None and not selection_details.empty:
            lines.extend([
                "",
                "## 🧮 选股打分明细（不含LLM情绪加分）",
                "",
                "| 股票代码 | 持仓? | base_score | 质量复合 | 动量复合 | 价值复合 | 情绪分 | 置信度 | 贡献:质量 | 贡献:动量 | 贡献:市值 | 贡献:情绪 |",
                "|----------|-------|-----------|----------|----------|----------|--------|--------|----------|----------|----------|----------|",
            ])
            for _, row in selection_details.iterrows():
                stock = str(row.get('stock_code', ''))
                is_hold = "✅" if bool(row.get('is_holding', False)) else ""
                base_score = float(row.get('base_score', 0.0)) if not isinstance(row.get('base_score', 0.0), pd.Series) else 0.0
                
                # 安全获取因子值（避免 Series 转 float 错误）
                def safe_float(val, default=0.0):
                    if val is None or (isinstance(val, pd.Series) and val.empty):
                        return default
                    if isinstance(val, pd.Series):
                        return float(val.iloc[0]) if len(val) > 0 else default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                q = safe_float(row.get('quality_composite_zscore', row.get(self.strategy.quality_col, 0.0)))
                m = safe_float(row.get('momentum_composite_zscore', row.get(self.strategy.momentum_col, 0.0)))
                v = safe_float(row.get('value_composite_zscore', row.get(self.strategy.value_col, 0.0)))
                s = safe_float(row.get('sentiment_score', 0.0))
                conf = safe_float(row.get('sentiment_confidence', 0.0))
                cq = safe_float(row.get('contrib_quality', 0.0))
                cm = safe_float(row.get('contrib_momentum', 0.0))
                cs = safe_float(row.get('contrib_size', 0.0))
                cse = safe_float(row.get('contrib_sentiment', 0.0))
                lines.append(
                    f"| {stock} | {is_hold} | {base_score:.3f} | {q:.2f} | {m:.2f} | {v:.2f} | "
                    f"{s:.2f} | {conf:.2f} | {cq:.3f} | {cm:.3f} | {cs:.3f} | {cse:.3f} |"
                )
        
        # ========== 执行SOP提醒 ==========
        lines.extend([
            f"",
            f"## 📝 执行SOP提醒",
            f"",
            f"1. **盘前确认**: 检查标的是否停牌/涨跌停/一字板",
            f"2. **执行顺序**: 先卖后买，优先处理流动性好的标的",
            f"3. **部分成交**: 如无法完全成交，记录实际成交金额",
            f"4. **收盘对账**: 从券商导出实际持仓，更新 `real_holdings.json`",
            f"",
            f"---",
            f"",
            f"*本报告由 A股量化交易系统 自动生成*",
        ])
        
        return "\n".join(lines)
    
    def _generate_enhanced_html_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        target_positions: Dict[str, float],
        strategy_info: Dict[str, Any],
        order_details: Dict[str, Dict[str, Any]],
        selection_details: pd.DataFrame,
        risk_status: Dict[str, Any],
        report_date: str,
        total_capital: float,
        total_buy: float,
        total_sell: float,
        total_cost: float,
        executable_count: int
    ) -> str:
        """生成增强版 HTML 报告"""
        # 生成卖出表格行
        sell_rows = ""
        for stock, amount in sell_orders.items():
            detail = order_details.get(stock, {})
            commission = detail.get('commission', 0)
            stamp = detail.get('stamp_duty', 0)
            cost = detail.get('total_cost', 0)
            is_exec = detail.get('is_executable', True)
            issues = ' '.join(detail.get('executability_issues', [])[:2])
            row_class = '' if is_exec else 'not-executable'
            sell_rows += f'''
            <tr class="{row_class}">
                <td>{stock}</td>
                <td>¥{amount:,.0f}</td>
                <td>¥{commission:.0f}</td>
                <td>¥{stamp:.0f}</td>
                <td>¥{cost:.0f}</td>
                <td>{'✅' if is_exec else '🚫'} {issues}</td>
            </tr>'''
        
        # 生成买入表格行
        buy_rows = ""
        for stock, amount in buy_orders.items():
            detail = order_details.get(stock, {})
            commission = detail.get('commission', 0)
            slippage = detail.get('slippage', 0)
            cost = detail.get('total_cost', 0)
            impact = detail.get('impact_ratio', 0)
            is_exec = detail.get('is_executable', True)
            issues = ' '.join(detail.get('executability_issues', [])[:2])
            row_class = '' if is_exec else 'not-executable'
            buy_rows += f'''
            <tr class="{row_class}">
                <td>{stock}</td>
                <td>¥{amount:,.0f}</td>
                <td>¥{commission:.0f}</td>
                <td>¥{slippage:.0f}</td>
                <td>¥{cost:.0f}</td>
                <td>{impact:.1%}</td>
                <td>{'✅' if is_exec else '🚫'} {issues}</td>
            </tr>'''
        
        # 生成持仓表格行
        position_rows = ""
        total_target = sum(target_positions.values()) if target_positions else 1
        for stock, amount in sorted(target_positions.items(), key=lambda x: -x[1]):
            weight = amount / total_target
            position_rows += f"<tr><td>{stock}</td><td>¥{amount:,.0f}</td><td>{weight:.1%}</td></tr>"

        # 选股打分明细表格
        selection_rows = ""
        if selection_details is not None and not selection_details.empty:
            for _, row in selection_details.iterrows():
                stock = str(row.get('stock_code', ''))
                is_hold = "✅" if bool(row.get('is_holding', False)) else ""
                
                # 安全获取因子值（避免 Series 转 float 错误）
                def safe_float(val, default=0.0):
                    if val is None or (isinstance(val, pd.Series) and val.empty):
                        return default
                    if isinstance(val, pd.Series):
                        return float(val.iloc[0]) if len(val) > 0 else default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                base_score = safe_float(row.get('base_score', 0.0))
                q = safe_float(row.get('quality_composite_zscore', row.get(self.strategy.quality_col, 0.0)))
                m = safe_float(row.get('momentum_composite_zscore', row.get(self.strategy.momentum_col, 0.0)))
                v = safe_float(row.get('value_composite_zscore', row.get(self.strategy.value_col, 0.0)))
                s = safe_float(row.get('sentiment_score', 0.0))
                conf = safe_float(row.get('sentiment_confidence', 0.0))
                cq = safe_float(row.get('contrib_quality', 0.0))
                cm = safe_float(row.get('contrib_momentum', 0.0))
                cs = safe_float(row.get('contrib_size', 0.0))
                cse = safe_float(row.get('contrib_sentiment', 0.0))
                selection_rows += (
                    f"<tr><td>{stock}</td><td>{is_hold}</td><td>{base_score:.3f}</td>"
                    f"<td>{q:.2f}</td><td>{m:.2f}</td><td>{v:.2f}</td>"
                    f"<td>{s:.2f}</td><td>{conf:.2f}</td>"
                    f"<td>{cq:.3f}</td><td>{cm:.3f}</td><td>{cs:.3f}</td><td>{cse:.3f}</td></tr>"
                )
        
        # 风控状态显示
        risk_class = "risk-alert" if risk_status.get('market_risk_triggered') else "risk-ok"
        risk_icon = "⚠️" if risk_status.get('market_risk_triggered') else "✅"
        risk_text = risk_status.get('market_risk_reason', '正常')
        
        selection_section = ""
        if selection_rows:
            selection_section = f"""
        <div class="card">
            <h2>🧮 选股打分明细（不含LLM情绪加分）</h2>
            <table>
                <thead>
                    <tr>
                        <th>股票代码</th><th>持仓?</th><th>base_score</th>
                        <th>质量复合</th><th>动量复合</th><th>价值复合</th>
                        <th>情绪分</th><th>置信度</th>
                        <th>贡献:质量</th><th>贡献:动量</th><th>贡献:市值</th>
                        <th>贡献:情绪</th>
                    </tr>
                </thead>
                <tbody>{selection_rows}</tbody>
            </table>
        </div>
            """
        
        html = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>每日调仓报告 - {report_date}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .meta {{ color: #888; margin-bottom: 2rem; }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .card h2 {{ font-size: 1.3rem; margin-bottom: 1rem; color: #00d9ff; }}
        .card.sell h2 {{ color: #ff6b6b; }}
        .card.buy h2 {{ color: #00ff88; }}
        .card.risk-alert {{ border-color: #ff6b6b; background: rgba(255, 107, 107, 0.1); }}
        .card.risk-ok {{ border-color: #00ff88; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }}
        .stat {{ text-align: center; padding: 1rem; background: rgba(0, 217, 255, 0.1); border-radius: 8px; }}
        .stat-value {{ font-size: 1.5rem; font-weight: bold; color: #00d9ff; }}
        .stat-label {{ font-size: 0.85rem; color: #888; margin-top: 0.25rem; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th, td {{ padding: 0.6rem; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }}
        th {{ color: #888; font-weight: 500; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.03); }}
        tr.not-executable {{ opacity: 0.6; background: rgba(255, 107, 107, 0.1); }}
        .total {{ margin-top: 1rem; padding-top: 1rem; border-top: 2px solid rgba(255, 255, 255, 0.1); font-weight: bold; }}
        .buy-total {{ color: #00ff88; }}
        .sell-total {{ color: #ff6b6b; }}
        .footer {{ text-align: center; color: #666; margin-top: 2rem; font-size: 0.85rem; }}
        .empty {{ text-align: center; color: #666; padding: 2rem; }}
        .sop {{ background: rgba(0, 217, 255, 0.05); padding: 1rem; border-radius: 8px; }}
        .sop li {{ margin: 0.5rem 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 每日调仓报告</h1>
        <p class="meta">报告日期: {report_date} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="card {risk_class}">
            <h2>🛡️ 风控状态</h2>
            <p>{risk_icon} {risk_text}</p>
        </div>
        
        <div class="card">
            <h2>📈 策略概览</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">¥{total_capital:,.0f}</div>
                    <div class="stat-label">总资金</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(target_positions)}</div>
                    <div class="stat-label">目标持仓数</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{executable_count}/{len(order_details)}</div>
                    <div class="stat-label">可执行订单</div>
                </div>
                <div class="stat">
                    <div class="stat-value">¥{total_cost:,.0f}</div>
                    <div class="stat-label">预估总成本</div>
                </div>
            </div>
        </div>
        
        <div class="card sell">
            <h2>📉 明日需卖出（按流动性排序，优先执行）</h2>
            {f"""
            <table>
                <thead><tr><th>股票代码</th><th>卖出金额</th><th>佣金</th><th>印花税</th><th>总成本</th><th>可执行性</th></tr></thead>
                <tbody>{sell_rows}</tbody>
            </table>
            <p class="total sell-total">卖出总金额: ¥{total_sell:,.0f}</p>
            """ if sell_orders else '<p class="empty">无需卖出</p>'}
        </div>
        
        <div class="card buy">
            <h2>📈 明日需买入（按流动性排序）</h2>
            {f"""
            <table>
                <thead><tr><th>股票代码</th><th>买入金额</th><th>佣金</th><th>滑点</th><th>总成本</th><th>冲击比</th><th>可执行性</th></tr></thead>
                <tbody>{buy_rows}</tbody>
            </table>
            <p class="total buy-total">买入总金额: ¥{total_buy:,.0f}</p>
            """ if buy_orders else '<p class="empty">无需买入</p>'}
        </div>
        
        <div class="card">
            <h2>📋 目标持仓明细</h2>
            <table>
                <thead><tr><th>股票代码</th><th>目标金额</th><th>权重</th></tr></thead>
                <tbody>{position_rows}</tbody>
            </table>
        </div>

        {selection_section}
        
        <div class="card">
            <h2>📝 执行SOP提醒</h2>
            <ul class="sop">
                <li><strong>盘前确认</strong>: 检查标的是否停牌/涨跌停/一字板</li>
                <li><strong>执行顺序</strong>: 先卖后买，优先处理流动性好的标的</li>
                <li><strong>部分成交</strong>: 如无法完全成交，记录实际成交金额</li>
                <li><strong>收盘对账</strong>: 从券商导出实际持仓，更新 real_holdings.json</li>
            </ul>
        </div>
        
        <p class="footer">本报告由 A股量化交易系统 自动生成</p>
    </div>
</body>
</html>
        '''
        return html
    
    def save_report(self, report_content: str, format: str = "markdown") -> Path:
        """保存报告"""
        return self.report_generator.save_report(
            report_content,
            self.today.strftime('%Y%m%d'),
            format
        )
    
    def run(self, force_rebalance: bool = False) -> bool:
        """
        执行完整的每日更新流程
        
        Parameters
        ----------
        force_rebalance : bool
            是否强制调仓
        
        Returns
        -------
        bool
            执行是否成功
        """
        self.logger.info("=" * 50)
        self.logger.info(f"开始每日更新任务: {self.today.strftime('%Y-%m-%d')}")
        self.logger.info("=" * 50)
        
        # Step 1: 更新市场数据
        self.logger.info("Step 1/6: 更新市场数据")
        if not self.update_market_data():
            self.logger.error("市场数据更新失败")
            return False
        
        # Step 2: 更新财务数据
        self.logger.info("Step 2/6: 更新财务数据")
        if not self.update_financial_data():
            self.logger.error("财务数据更新失败")
            return False
        
        # Step 3: 更新基准指数
        self.logger.info("Step 3/6: 更新基准指数")
        self.update_benchmark_data()
        
        # Step 4: 计算因子
        self.logger.info("Step 4/6: 计算因子数据")
        if not self.calculate_factors():
            self.logger.error("因子计算失败")
            return False
        
        # Step 5: 判断是否调仓日
        is_rebalance = force_rebalance or self.is_rebalance_day()
        
        if is_rebalance:
            self.logger.info("Step 5/6: 生成目标持仓（调仓日）")
            if not self.generate_target_positions():
                self.logger.error("目标持仓生成失败")
                return False
        else:
            self.logger.info("Step 5/6: 非调仓日，跳过持仓生成")
            self.target_positions = self.current_positions.copy()
        
        # Step 6: 生成报告
        self.logger.info("Step 6/6: 生成交易报告")
        buy_orders, sell_orders = self.calculate_trade_orders()
        
        report_paths = {}
        for fmt in ["markdown", "html"]:
            report_content = self.generate_report(buy_orders, sell_orders, format=fmt)
            report_paths[fmt] = self.save_report(report_content, format=fmt)
        
        # 更新持仓
        self.save_current_holdings(buy_orders, sell_orders)

        # PushPlus 推送（可选）
        try:
            notif_cfg = self.config.get("notification", {})
            if notif_cfg.get("enabled", False):
                token = str(notif_cfg.get("pushplus_token", "")).strip()
                if token:
                    buy_cnt = len(buy_orders)
                    sell_cnt = len(sell_orders)
                    selected = list(self.target_positions.keys())[:10]
                    title = f"Quant 日报 {self.today.strftime('%Y-%m-%d')}"
                    report_path = report_paths.get('html') or report_paths.get('markdown')
                    content_lines = [
                        "### 交易信号",
                        f"- 买入: {buy_cnt} 只",
                        f"- 卖出: {sell_cnt} 只",
                        f"- 目标持仓: {len(self.target_positions)} 只",
                        "",
                        "### 标的（前6只）",
                    ]
                    content_lines.extend([f"- {s}" for s in selected[:6]])
                    content_lines.extend(["", f"报告已生成：{report_path}"])
                    content = "\n".join(content_lines)

                    send_pushplus_msg(
                        token=token,
                        title=title,
                        content=content,
                        template="markdown",
                        topic=notif_cfg.get("topic"),
                        channel=notif_cfg.get("channel"),
                        timeout=float(notif_cfg.get("timeout", 30)),
                        max_retries=int(notif_cfg.get("max_retries", 3)),
                    )
                else:
                    self.logger.warning("PushPlus token 为空，已跳过推送")
        except Exception as e:
            self.logger.warning(f"PushPlus 推送失败（忽略并继续）: {e}")
        
        self.logger.info("=" * 50)
        self.logger.info("每日更新任务完成")
        self.logger.info("=" * 50)
        
        return True


def run_daily_update(
    force_rebalance: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    运行每日更新
    
    Parameters
    ----------
    force_rebalance : bool
        是否强制调仓
    config : Optional[Dict[str, Any]]
        配置参数
    
    Returns
    -------
    bool
        执行是否成功
    """
    runner = DailyUpdateRunner(config)
    return runner.run(force_rebalance)

