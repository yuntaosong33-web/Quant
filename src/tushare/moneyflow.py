"""
Tushare 资金流向与融资融券模块

该模块提供资金流向数据获取功能，作为 Mixin 类混入主类。

Features
--------
- 个股资金流向（大单/超大单）
- 北向资金持仓
- 融资融券数据
- Smart Money 因子
- 杠杆过热因子
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import time

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TushareMoneyflowMixin:
    """
    Tushare 资金流向 Mixin
    
    提供资金流向和融资融券数据获取功能，需要与 TushareDataLoaderBase 组合使用。
    """
    
    # ==================== 资金流向 ====================
    
    def fetch_moneyflow(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        获取个股资金流向数据（大单/超大单）
        
        使用 Tushare Pro moneyflow 接口获取个股资金流向明细数据，
        包含大单、超大单、中单、小单的买入卖出金额。
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位），如 "000001"
        start_date : str
            开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date : str
            结束日期，格式 YYYYMMDD 或 YYYY-MM-DD
        
        Returns
        -------
        Optional[pd.DataFrame]
            资金流向数据
        """
        ts_code = self._to_ts_code(stock_code)
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        logger.debug(f"获取资金流向: {stock_code}, {start_date} ~ {end_date}")
        
        cache_file = self.cache_dir / f"moneyflow_{stock_code}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载资金流向: {stock_code}")
                    return df
            except Exception:
                pass
        
        df = self._fetch_with_retry(
            self.pro.moneyflow,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取资金流向失败: {stock_code}")
            return None
        
        df["stock_code"] = df["ts_code"].str[:6]
        
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")
        
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取资金流向成功: {stock_code}, {len(df)} 条")
        return df
    
    def fetch_moneyflow_batch(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        show_progress: bool = True,
        batch_size: int = 150,
        batch_sleep: float = 8.0
    ) -> pd.DataFrame:
        """
        批量获取资金流向数据
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        show_progress : bool
            是否显示进度条
        batch_size : int
            每批次处理的股票数量
        batch_sleep : float
            每批次之间的休息时间（秒）
        
        Returns
        -------
        pd.DataFrame
            合并后的资金流向数据
        """
        all_data = []
        total = len(stock_list)
        success_count = 0
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(stock_list),
                    total=total,
                    desc="💰 获取资金流向",
                    unit="只",
                    ncols=80
                )
            except ImportError:
                iterator = enumerate(stock_list)
                logger.info(f"开始获取资金流向: {total} 只股票...")
        else:
            iterator = enumerate(stock_list)
        
        for i, stock in iterator:
            df = self.fetch_moneyflow(stock, start_date, end_date)
            if df is not None and not df.empty:
                all_data.append(df)
                success_count += 1
            
            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({"成功": success_count, "当前": stock})
            
            if (i + 1) % batch_size == 0 and (i + 1) < total:
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description(f"💰 休息{batch_sleep}s")
                time.sleep(batch_sleep)
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description("💰 获取资金流向")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"批量获取资金流向完成: {success_count}/{total} 只, {len(result)} 条记录")
        return result
    
    # ==================== 北向资金 ====================
    
    def fetch_hk_hold(
        self,
        trade_date: str,
        stock_code: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取沪深港通持股数据（北向资金持仓）
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD 或 YYYY-MM-DD
        stock_code : Optional[str]
            股票代码（6位），如果提供则只返回该股票的数据
        
        Returns
        -------
        Optional[pd.DataFrame]
            北向持仓数据
        """
        trade_date = trade_date.replace("-", "")
        
        logger.debug(f"获取北向持仓: {trade_date}")
        
        cache_file = self.cache_dir / f"hk_hold_{trade_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载北向持仓: {trade_date}, {len(df)} 条")
                    if stock_code:
                        df = df[df["stock_code"] == stock_code[:6]]
                    return df
            except Exception:
                pass
        
        df = self._fetch_with_retry(
            self.pro.hk_hold,
            trade_date=trade_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取北向持仓失败: {trade_date}")
            return None
        
        df["stock_code"] = df["ts_code"].str[:6]
        
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取北向持仓成功: {trade_date}, {len(df)} 条")
        
        if stock_code:
            df = df[df["stock_code"] == stock_code[:6]]
        
        return df
    
    def fetch_hk_hold_change(
        self,
        stock_list: List[str],
        current_date: str,
        days_back: int = 5
    ) -> pd.DataFrame:
        """
        计算北向资金持仓占比变化
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表（6位代码）
        current_date : str
            当前日期
        days_back : int
            回溯天数，默认 5 天
        
        Returns
        -------
        pd.DataFrame
            北向持仓变化数据
        """
        current_date = current_date.replace("-", "")
        
        start_date = (
            datetime.strptime(current_date, "%Y%m%d") - timedelta(days=days_back + 10)
        ).strftime("%Y%m%d")
        
        calendar = self.fetch_trade_calendar(start_date, current_date)
        
        if len(calendar) < 2:
            logger.warning("交易日不足，无法计算北向持仓变化")
            return pd.DataFrame()
        
        current_trade_date = calendar[-1].strftime("%Y%m%d")
        prev_trade_date = calendar[max(0, len(calendar) - days_back - 1)].strftime("%Y%m%d")
        
        logger.info(f"计算北向持仓变化: {prev_trade_date} -> {current_trade_date}")
        
        current_hold = self.fetch_hk_hold(current_trade_date)
        prev_hold = self.fetch_hk_hold(prev_trade_date)
        
        result = pd.DataFrame({"stock_code": stock_list})
        result["hk_ratio"] = np.nan
        result["hk_ratio_prev"] = np.nan
        result["hk_ratio_change"] = 0.0
        
        if current_hold is not None and not current_hold.empty:
            current_hold_subset = current_hold[["stock_code", "ratio"]].rename(
                columns={"ratio": "hk_ratio"}
            )
            result = result.merge(
                current_hold_subset, on="stock_code", how="left", suffixes=("_drop", "")
            )
            if "hk_ratio_drop" in result.columns:
                result["hk_ratio"] = result["hk_ratio"].fillna(result["hk_ratio_drop"])
                result = result.drop(columns=["hk_ratio_drop"])
        
        if prev_hold is not None and not prev_hold.empty:
            prev_hold_subset = prev_hold[["stock_code", "ratio"]].rename(
                columns={"ratio": "hk_ratio_prev"}
            )
            result = result.merge(
                prev_hold_subset, on="stock_code", how="left", suffixes=("_drop", "")
            )
            if "hk_ratio_prev_drop" in result.columns:
                result["hk_ratio_prev"] = result["hk_ratio_prev"].fillna(
                    result["hk_ratio_prev_drop"]
                )
                result = result.drop(columns=["hk_ratio_prev_drop"])
        
        result["hk_ratio_change"] = (
            result["hk_ratio"].fillna(0) - result["hk_ratio_prev"].fillna(0)
        )
        
        valid_mask = result["hk_ratio_change"].notna()
        if valid_mask.sum() > 0:
            result.loc[valid_mask, "hk_hold_score"] = (
                result.loc[valid_mask, "hk_ratio_change"].rank(pct=True)
            )
        else:
            result["hk_hold_score"] = 0.5
        
        result["hk_hold_score"] = result["hk_hold_score"].fillna(0.5)
        
        logger.info(f"北向持仓变化计算完成: {len(result)} 只股票")
        return result
    
    def calculate_smart_money_score(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        north_weight: float = 0.6,
        large_order_weight: float = 0.4
    ) -> pd.DataFrame:
        """
        计算全息主力资金因子 (Holographic Smart Money Score)
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        north_weight : float
            北向资金因子权重，默认 0.6
        large_order_weight : float
            大单流向因子权重，默认 0.4
        
        Returns
        -------
        pd.DataFrame
            主力资金得分数据
        """
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        logger.info(f"📊 计算全息主力资金因子: {len(stock_list)} 只股票")
        
        hk_change = self.fetch_hk_hold_change(
            stock_list=stock_list,
            current_date=end_date,
            days_back=5
        )
        
        moneyflow_df = self.fetch_moneyflow_batch(
            stock_list=stock_list,
            start_date=start_date,
            end_date=end_date,
            show_progress=True
        )
        
        if not moneyflow_df.empty:
            flow_cols = ["buy_elg_amount", "sell_elg_amount", "buy_lg_amount", "sell_lg_amount"]
            if all(col in moneyflow_df.columns for col in flow_cols):
                moneyflow_df["main_net_inflow"] = (
                    moneyflow_df["buy_elg_amount"] - moneyflow_df["sell_elg_amount"] +
                    moneyflow_df["buy_lg_amount"] - moneyflow_df["sell_lg_amount"]
                )
            elif "net_mf_amount" in moneyflow_df.columns:
                moneyflow_df["main_net_inflow"] = moneyflow_df["net_mf_amount"]
            else:
                moneyflow_df["main_net_inflow"] = 0
            
            flow_summary = moneyflow_df.groupby("stock_code").agg({
                "main_net_inflow": "sum"
            }).reset_index()
            
            if len(flow_summary) > 0:
                flow_summary["large_order_score"] = flow_summary["main_net_inflow"].rank(pct=True)
            else:
                flow_summary["large_order_score"] = 0.5
        else:
            flow_summary = pd.DataFrame({
                "stock_code": stock_list,
                "main_net_inflow": 0,
                "large_order_score": 0.5
            })
        
        result = pd.DataFrame({"stock_code": stock_list})
        
        if not hk_change.empty:
            result = result.merge(
                hk_change[["stock_code", "hk_hold_score", "hk_ratio_change"]],
                on="stock_code",
                how="left"
            )
            result["north_score"] = result["hk_hold_score"].fillna(0.5)
        else:
            result["north_score"] = 0.5
            result["hk_ratio_change"] = 0.0
        
        result = result.merge(
            flow_summary[["stock_code", "large_order_score", "main_net_inflow"]],
            on="stock_code",
            how="left"
        )
        result["large_order_score"] = result["large_order_score"].fillna(0.5)
        result["main_net_inflow"] = result["main_net_inflow"].fillna(0)
        
        result["smart_money_score"] = (
            north_weight * result["north_score"] +
            large_order_weight * result["large_order_score"]
        )
        
        if "hk_hold_score" in result.columns:
            result = result.drop(columns=["hk_hold_score"])
        
        logger.info(
            f"✅ 主力资金因子计算完成: "
            f"均值={result['smart_money_score'].mean():.3f}, "
            f"top10均值={result.nlargest(10, 'smart_money_score')['smart_money_score'].mean():.3f}"
        )
        
        return result
    
    # ==================== 融资融券 ====================
    
    def fetch_margin_detail(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        获取个股融资融券明细数据
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位）
        start_date : str
            开始日期
        end_date : str
            结束日期
        
        Returns
        -------
        Optional[pd.DataFrame]
            融资融券明细数据
        """
        ts_code = self._to_ts_code(stock_code)
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        logger.debug(f"获取融资融券数据: {stock_code}, {start_date} ~ {end_date}")
        
        cache_file = self.cache_dir / f"margin_{stock_code}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载融资融券数据: {stock_code}")
                    return df
            except Exception:
                pass
        
        df = self._fetch_with_retry(
            self.pro.margin_detail,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取融资融券数据失败: {stock_code}")
            return None
        
        df["stock_code"] = df["ts_code"].str[:6]
        
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")
        
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取融资融券数据成功: {stock_code}, {len(df)} 条")
        return df
    
    def fetch_margin(
        self,
        trade_date: str
    ) -> Optional[pd.DataFrame]:
        """
        获取全市场融资融券汇总数据
        
        Parameters
        ----------
        trade_date : str
            交易日期
        
        Returns
        -------
        Optional[pd.DataFrame]
            全市场融资融券汇总数据
        """
        trade_date = trade_date.replace("-", "")
        
        logger.debug(f"获取全市场融资融券数据: {trade_date}")
        
        cache_file = self.cache_dir / f"margin_{trade_date}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载融资融券数据: {trade_date}, {len(df)} 条")
                    return df
            except Exception:
                pass
        
        df = self._fetch_with_retry(
            self.pro.margin,
            trade_date=trade_date
        )
        
        if df is None or df.empty:
            logger.debug(f"获取融资融券数据失败: {trade_date}")
            return None
        
        df["stock_code"] = df["ts_code"].str[:6]
        
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取融资融券数据成功: {trade_date}, {len(df)} 条")
        return df
    
    def fetch_margin_batch(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        show_progress: bool = True,
        batch_size: int = 150,
        batch_sleep: float = 8.0
    ) -> pd.DataFrame:
        """
        批量获取融资融券数据
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        show_progress : bool
            是否显示进度条
        batch_size : int
            每批次处理的股票数量
        batch_sleep : float
            每批次之间的休息时间
        
        Returns
        -------
        pd.DataFrame
            合并后的融资融券数据
        """
        all_data = []
        total = len(stock_list)
        success_count = 0
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(stock_list),
                    total=total,
                    desc="📊 获取融资融券",
                    unit="只",
                    ncols=80
                )
            except ImportError:
                iterator = enumerate(stock_list)
                logger.info(f"开始获取融资融券数据: {total} 只股票...")
        else:
            iterator = enumerate(stock_list)
        
        for i, stock in iterator:
            df = self.fetch_margin_detail(stock, start_date, end_date)
            if df is not None and not df.empty:
                all_data.append(df)
                success_count += 1
            
            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({"成功": success_count, "当前": stock})
            
            if (i + 1) % batch_size == 0 and (i + 1) < total:
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description(f"📊 休息{batch_sleep}s")
                time.sleep(batch_sleep)
                if show_progress and hasattr(iterator, 'set_description'):
                    iterator.set_description("📊 获取融资融券")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"批量获取融资融券完成: {success_count}/{total} 只, {len(result)} 条记录")
        return result
    
    def calculate_leverage_risk(
        self,
        stock_list: List[str],
        trade_date: str,
        lookback_days: int = 20
    ) -> pd.DataFrame:
        """
        计算杠杆过热因子 (Leverage Overheat Factor)
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        trade_date : str
            交易日期
        lookback_days : int
            回溯天数，默认 20 天
        
        Returns
        -------
        pd.DataFrame
            杠杆过热因子数据
        """
        trade_date = trade_date.replace("-", "")
        start_date = (
            datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days + 10)
        ).strftime("%Y%m%d")
        
        logger.info(f"🔥 计算杠杆过热因子: {len(stock_list)} 只股票, {trade_date}")
        
        margin_df = self.fetch_margin_batch(
            stock_list=stock_list,
            start_date=start_date,
            end_date=trade_date,
            show_progress=True
        )
        
        daily_basic = self.fetch_daily_basic(trade_date, stock_list)
        
        daily_data = self.fetch_daily_data_batch(
            stock_list=stock_list,
            start_date=start_date,
            end_date=trade_date,
            show_progress=False
        )
        
        if margin_df.empty:
            logger.warning("无融资融券数据，返回空结果")
            return pd.DataFrame({"stock_code": stock_list, "leverage_heat": 0, "leverage_risk_score": 0.5})
        
        if not daily_data.empty and "amount" in daily_data.columns:
            if "trade_date" in margin_df.columns:
                margin_df["trade_date"] = pd.to_datetime(margin_df["trade_date"])
            if "date" in daily_data.columns:
                daily_data = daily_data.rename(columns={"date": "trade_date"})
            
            margin_df = margin_df.merge(
                daily_data[["stock_code", "trade_date", "amount"]],
                on=["stock_code", "trade_date"],
                how="left"
            )
        
        if "rzmre" in margin_df.columns and "amount" in margin_df.columns:
            margin_df["margin_buy_ratio"] = (
                margin_df["rzmre"] / margin_df["amount"].replace(0, np.nan)
            )
        else:
            margin_df["margin_buy_ratio"] = np.nan
        
        result_list = []
        for stock in stock_list:
            stock_margin = margin_df[margin_df["stock_code"] == stock].copy()
            
            if stock_margin.empty:
                result_list.append({
                    "stock_code": stock,
                    "rzye": np.nan,
                    "rzmre": np.nan,
                    "margin_buy_ratio": np.nan,
                    "margin_balance_ratio": np.nan,
                    "leverage_heat": 0,
                    "leverage_risk_score": 0.5
                })
                continue
            
            stock_margin = stock_margin.sort_values("trade_date")
            
            stock_margin["ratio_mean"] = stock_margin["margin_buy_ratio"].rolling(
                window=lookback_days, min_periods=5
            ).mean()
            stock_margin["ratio_std"] = stock_margin["margin_buy_ratio"].rolling(
                window=lookback_days, min_periods=5
            ).std()
            
            latest = stock_margin.iloc[-1]
            
            if pd.notna(latest.get("ratio_std")) and latest["ratio_std"] > 0:
                leverage_heat = (
                    (latest["margin_buy_ratio"] - latest["ratio_mean"]) / latest["ratio_std"]
                )
            else:
                leverage_heat = 0
            
            margin_balance_ratio = np.nan
            if daily_basic is not None and not daily_basic.empty:
                stock_basic = daily_basic[daily_basic["stock_code"] == stock]
                if not stock_basic.empty and "rzye" in latest:
                    total_mv = stock_basic["total_mv"].iloc[0]
                    if pd.notna(total_mv) and total_mv > 0:
                        margin_balance_ratio = latest.get("rzye", 0) / total_mv
            
            result_list.append({
                "stock_code": stock,
                "rzye": latest.get("rzye", np.nan),
                "rzmre": latest.get("rzmre", np.nan),
                "margin_buy_ratio": latest.get("margin_buy_ratio", np.nan),
                "margin_balance_ratio": margin_balance_ratio,
                "leverage_heat": leverage_heat,
            })
        
        result = pd.DataFrame(result_list)
        
        valid_mask = result["leverage_heat"].notna() & (result["leverage_heat"] != 0)
        if valid_mask.sum() > 0:
            result.loc[valid_mask, "leverage_risk_score"] = (
                result.loc[valid_mask, "leverage_heat"].rank(pct=True)
            )
        else:
            result["leverage_risk_score"] = 0.5
        
        result["leverage_risk_score"] = result["leverage_risk_score"].fillna(0.5)
        
        overheated_count = (result["leverage_heat"] > 2).sum()
        cold_count = (result["leverage_heat"] < -1).sum()
        
        logger.info(
            f"✅ 杠杆过热因子计算完成: "
            f"{len(result)} 只, 过热={overheated_count}, 偏冷={cold_count}"
        )
        
        return result
    
    def calculate_market_leverage_sentiment(
        self,
        trade_date: str,
        index_code: str = "hs300"
    ) -> Dict[str, Any]:
        """
        计算市场整体杠杆情绪
        
        Parameters
        ----------
        trade_date : str
            交易日期
        index_code : str
            指数代码
        
        Returns
        -------
        Dict[str, Any]
            市场杠杆情绪指标
        """
        trade_date = trade_date.replace("-", "")
        
        logger.info(f"📈 计算市场整体杠杆情绪: {index_code}, {trade_date}")
        
        stock_list = self.fetch_index_constituents(index_code)
        
        if not stock_list:
            return {
                "avg_leverage_heat": 0,
                "overheated_ratio": 0,
                "cold_ratio": 0,
                "market_risk_level": "unknown",
                "signal": "hold"
            }
        
        sample_size = min(100, len(stock_list))
        sampled_stocks = stock_list[:sample_size]
        
        leverage_df = self.calculate_leverage_risk(
            stock_list=sampled_stocks,
            trade_date=trade_date
        )
        
        if leverage_df.empty:
            return {
                "avg_leverage_heat": 0,
                "overheated_ratio": 0,
                "cold_ratio": 0,
                "market_risk_level": "unknown",
                "signal": "hold"
            }
        
        valid_heat = leverage_df["leverage_heat"].dropna()
        avg_heat = valid_heat.mean() if len(valid_heat) > 0 else 0
        overheated_ratio = (valid_heat > 2).sum() / len(valid_heat) if len(valid_heat) > 0 else 0
        cold_ratio = (valid_heat < -1).sum() / len(valid_heat) if len(valid_heat) > 0 else 0
        
        if avg_heat > 2 or overheated_ratio > 0.3:
            risk_level = "extreme"
            signal = "sell"
        elif avg_heat > 1 or overheated_ratio > 0.2:
            risk_level = "high"
            signal = "reduce"
        elif avg_heat > 0.5:
            risk_level = "medium"
            signal = "hold"
        elif avg_heat < -1 and cold_ratio > 0.3:
            risk_level = "low"
            signal = "buy"
        else:
            risk_level = "normal"
            signal = "hold"
        
        result = {
            "trade_date": trade_date,
            "index_code": index_code,
            "sample_size": len(sampled_stocks),
            "avg_leverage_heat": round(avg_heat, 3),
            "overheated_ratio": round(overheated_ratio, 3),
            "cold_ratio": round(cold_ratio, 3),
            "market_risk_level": risk_level,
            "signal": signal
        }
        
        logger.info(
            f"✅ 市场杠杆情绪: 风险={risk_level}, "
            f"过热={avg_heat:.2f}, 过热比={overheated_ratio:.1%}"
        )
        
        return result
    
    def get_leverage_warning_stocks(
        self,
        stock_list: List[str],
        trade_date: str,
        heat_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        获取杠杆过热预警股票
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        trade_date : str
            交易日期
        heat_threshold : float
            过热阈值，默认 2.0
        
        Returns
        -------
        pd.DataFrame
            过热预警股票列表
        """
        leverage_df = self.calculate_leverage_risk(stock_list, trade_date)
        
        if leverage_df.empty:
            return pd.DataFrame()
        
        warnings = leverage_df[leverage_df["leverage_heat"] >= heat_threshold].copy()
        warnings = warnings.sort_values("leverage_heat", ascending=False)
        
        if len(warnings) > 0:
            logger.warning(
                f"⚠️ 发现 {len(warnings)} 只杠杆过热股票 "
                f"(heat >= {heat_threshold})"
            )
        
        return warnings

