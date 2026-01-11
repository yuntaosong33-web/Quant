"""
Tushare 涨跌停与龙头因子模块

该模块提供涨跌停分析和龙头因子计算功能，作为 Mixin 类混入主类。

Features
--------
- 涨跌停列表获取
- 可交易性检查
- 连板天数计算
- 龙头信仰因子
- 退市股票信息
"""

from typing import Optional, List
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TushareLimitFactorMixin:
    """
    Tushare 涨跌停/龙头因子 Mixin
    
    提供涨跌停分析和龙头因子计算功能，需要与 TushareDataLoaderBase 组合使用。
    """
    
    def fetch_limit_list(
        self,
        trade_date: str,
        limit_type: str = "U"
    ) -> Optional[pd.DataFrame]:
        """
        获取每日涨跌停股票列表
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD 或 YYYY-MM-DD
        limit_type : str
            涨跌停类型："U"(涨停) 或 "D"(跌停)
        
        Returns
        -------
        Optional[pd.DataFrame]
            涨跌停明细数据
        """
        trade_date = trade_date.replace("-", "")
        
        logger.debug(f"获取涨跌停列表: {trade_date}, 类型={limit_type}")
        
        cache_file = self.cache_dir / f"limit_list_{trade_date}_{limit_type}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.debug(f"从缓存加载涨跌停列表: {trade_date}, {len(df)} 条")
                    return df
            except Exception:
                pass
        
        df = self._fetch_with_retry(
            self.pro.limit_list,
            trade_date=trade_date,
            limit_type=limit_type
        )
        
        if df is None or df.empty:
            logger.debug(f"获取涨跌停列表失败: {trade_date}")
            return None
        
        df["stock_code"] = df["ts_code"].str[:6]
        
        rename_map = {"open_times": "open_num"}
        df = df.rename(columns=rename_map)
        
        if "open_num" not in df.columns:
            df["open_num"] = 0
        if "fc_ratio" not in df.columns and "fd_amount" in df.columns and "amount" in df.columns:
            df["fc_ratio"] = df["fd_amount"] / df["amount"].replace(0, np.nan) * 100
        
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass
        
        logger.debug(f"获取涨跌停列表成功: {trade_date}, {len(df)} 条")
        return df
    
    def fetch_limit_list_batch(
        self,
        start_date: str,
        end_date: str,
        limit_type: str = "U",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        批量获取多日涨跌停列表
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        limit_type : str
            涨跌停类型
        show_progress : bool
            是否显示进度条
        
        Returns
        -------
        pd.DataFrame
            合并后的涨跌停数据
        """
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        
        calendar = self.fetch_trade_calendar(start_date, end_date)
        
        all_data = []
        total = len(calendar)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    calendar,
                    desc="🔥 获取涨停数据",
                    unit="天",
                    ncols=80
                )
            except ImportError:
                iterator = calendar
                logger.info(f"开始获取涨停数据: {total} 个交易日...")
        else:
            iterator = calendar
        
        for date in iterator:
            date_str = date.strftime("%Y%m%d")
            df = self.fetch_limit_list(date_str, limit_type)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"批量获取涨停数据完成: {len(calendar)} 天, {len(result)} 条记录")
        return result
    
    def check_tradability(
        self,
        stock_list: List[str],
        trade_date: str,
        check_limit_up: bool = True,
        check_suspend: bool = True
    ) -> pd.DataFrame:
        """
        检查股票的可交易性
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        trade_date : str
            交易日期
        check_limit_up : bool
            是否检查涨停
        check_suspend : bool
            是否检查停牌
        
        Returns
        -------
        pd.DataFrame
            可交易性结果
        """
        trade_date = trade_date.replace("-", "")
        
        result = pd.DataFrame({
            'stock_code': stock_list,
            'is_tradable': True,
            'is_limit_up': False,
            'is_one_word_limit': False,
            'is_limit_down': False,
            'is_suspended': False,
            'limit_strength': 0.0,
            'reason': ''
        })
        
        if not stock_list:
            return result
        
        if check_limit_up:
            limit_up_df = self.fetch_limit_list(trade_date, "U")
            if limit_up_df is not None and not limit_up_df.empty:
                limit_up_codes = set(limit_up_df['stock_code'].tolist())
                
                for idx, row in result.iterrows():
                    code = row['stock_code']
                    if code in limit_up_codes:
                        result.at[idx, 'is_limit_up'] = True
                        
                        stock_limit = limit_up_df[limit_up_df['stock_code'] == code].iloc[0]
                        
                        open_times = stock_limit.get('open_times', 0) or 0
                        fc_ratio = stock_limit.get('fc_ratio', 0) or 0
                        strength = stock_limit.get('strth', 0) or 0
                        
                        result.at[idx, 'limit_strength'] = strength
                        
                        if open_times == 0 or fc_ratio > 50:
                            result.at[idx, 'is_one_word_limit'] = True
                            result.at[idx, 'is_tradable'] = False
                            result.at[idx, 'reason'] = f'一字涨停(开板{open_times}次,封比{fc_ratio:.0f}%)'
                        elif fc_ratio > 30:
                            result.at[idx, 'reason'] = f'涨停(封比{fc_ratio:.0f}%,可能难买)'
        
        limit_down_df = self.fetch_limit_list(trade_date, "D")
        if limit_down_df is not None and not limit_down_df.empty:
            limit_down_codes = set(limit_down_df['stock_code'].tolist())
            
            for idx, row in result.iterrows():
                if row['stock_code'] in limit_down_codes:
                    result.at[idx, 'is_limit_down'] = True
                    if not result.at[idx, 'reason']:
                        result.at[idx, 'reason'] = '跌停(卖出可能受限)'
        
        if check_suspend:
            try:
                suspend_df = self._fetch_with_retry(
                    self.pro.suspend_d,
                    trade_date=trade_date,
                    suspend_type='S'
                )
                
                if suspend_df is not None and not suspend_df.empty:
                    suspend_codes = set(suspend_df['ts_code'].str[:6].tolist())
                    
                    for idx, row in result.iterrows():
                        if row['stock_code'] in suspend_codes:
                            result.at[idx, 'is_suspended'] = True
                            result.at[idx, 'is_tradable'] = False
                            result.at[idx, 'reason'] = '停牌'
            except Exception as e:
                logger.debug(f"获取停牌信息失败: {e}")
        
        tradable_count = result['is_tradable'].sum()
        logger.info(
            f"可交易性检查 {trade_date}: "
            f"总计 {len(stock_list)} 只, 可交易 {tradable_count} 只, "
            f"涨停 {result['is_limit_up'].sum()} 只, "
            f"一字板 {result['is_one_word_limit'].sum()} 只"
        )
        
        return result
    
    def calculate_consecutive_limits(
        self,
        stock_code: str,
        end_date: str,
        days_back: int = 30
    ) -> int:
        """
        计算股票的连板天数
        
        Parameters
        ----------
        stock_code : str
            股票代码
        end_date : str
            截止日期
        days_back : int
            回溯天数
        
        Returns
        -------
        int
            连续涨停天数
        """
        end_date = end_date.replace("-", "")
        start_date = (
            datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days_back)
        ).strftime("%Y%m%d")
        
        calendar = self.fetch_trade_calendar(start_date, end_date)
        if len(calendar) == 0:
            return 0
        
        consecutive_count = 0
        
        for date in reversed(calendar):
            date_str = date.strftime("%Y%m%d")
            limit_df = self.fetch_limit_list(date_str, limit_type="U")
            
            if limit_df is None or limit_df.empty:
                if consecutive_count == 0:
                    continue
                else:
                    break
            
            if stock_code in limit_df["stock_code"].values:
                consecutive_count += 1
            else:
                if consecutive_count > 0:
                    break
                if consecutive_count == 0:
                    date_diff = (datetime.strptime(end_date, "%Y%m%d") - date).days
                    if date_diff > 5:
                        return 0
        
        return consecutive_count
    
    def fetch_delisted_stocks(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取退市股票信息
        
        Parameters
        ----------
        start_date : Optional[str]
            退市日期起始
        end_date : Optional[str]
            退市日期结束
        
        Returns
        -------
        pd.DataFrame
            退市股票信息
        """
        logger.info("获取退市股票信息...")
        
        cache_file = self.cache_dir / "delisted_stocks.parquet"
        
        if cache_file.exists():
            try:
                cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.now() - cache_mtime).days < 7:
                    df = pd.read_parquet(cache_file)
                    logger.debug(f"从缓存加载退市股票: {len(df)} 只")
                    return df
            except Exception:
                pass
        
        try:
            df = self._fetch_with_retry(
                self.pro.stock_basic,
                exchange='',
                list_status='D',
                fields='ts_code,name,list_date,delist_date'
            )
            
            if df is None or df.empty:
                logger.warning("未获取到退市股票信息")
                return pd.DataFrame()
            
            df['stock_code'] = df['ts_code'].str[:6]
            df['is_delisted'] = True
            
            if start_date or end_date:
                if 'delist_date' in df.columns:
                    df['delist_date'] = pd.to_datetime(df['delist_date'])
                    if start_date:
                        start = pd.to_datetime(start_date)
                        df = df[df['delist_date'] >= start]
                    if end_date:
                        end = pd.to_datetime(end_date)
                        df = df[df['delist_date'] <= end]
            
            try:
                df.to_parquet(cache_file)
            except Exception as e:
                logger.debug(f"缓存退市股票信息失败: {e}")
            
            logger.info(f"获取退市股票完成: {len(df)} 只")
            return df
            
        except Exception as e:
            logger.warning(f"获取退市股票失败: {e}")
            return pd.DataFrame()
    
    def fetch_name_change_history(
        self,
        stock_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取股票更名历史
        
        Parameters
        ----------
        stock_code : Optional[str]
            股票代码
        start_date : Optional[str]
            更名日期起始
        end_date : Optional[str]
            更名日期结束
        
        Returns
        -------
        pd.DataFrame
            更名历史
        """
        logger.debug(f"获取股票更名历史: {stock_code or '全部'}")
        
        try:
            kwargs = {}
            if stock_code:
                if not ('.' in stock_code):
                    suffix = '.SH' if stock_code.startswith(('6', '9')) else '.SZ'
                    kwargs['ts_code'] = stock_code + suffix
                else:
                    kwargs['ts_code'] = stock_code
            
            if start_date:
                kwargs['start_date'] = start_date.replace('-', '')
            if end_date:
                kwargs['end_date'] = end_date.replace('-', '')
            
            df = self._fetch_with_retry(
                self.pro.namechange,
                **kwargs
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            df['stock_code'] = df['ts_code'].str[:6]
            
            logger.debug(f"获取更名历史完成: {len(df)} 条")
            return df
            
        except Exception as e:
            logger.warning(f"获取更名历史失败: {e}")
            return pd.DataFrame()
    
    def calculate_limit_strength(
        self,
        trade_date: str,
        min_fl_ratio: float = 1.0
    ) -> pd.DataFrame:
        """
        计算涨停封板强度因子（龙头信仰因子）
        
        Parameters
        ----------
        trade_date : str
            交易日期
        min_fl_ratio : float
            最低封流比阈值（%）
        
        Returns
        -------
        pd.DataFrame
            涨停强度因子数据
        """
        trade_date = trade_date.replace("-", "")
        
        logger.info(f"🐉 计算龙头信仰因子: {trade_date}")
        
        limit_df = self.fetch_limit_list(trade_date, limit_type="U")
        
        if limit_df is None or limit_df.empty:
            logger.warning(f"无涨停数据: {trade_date}")
            return pd.DataFrame()
        
        result = limit_df.copy()
        
        if "fd_amount" in result.columns and "amount" in result.columns:
            result["bid_strength"] = (
                result["fd_amount"] / result["amount"].replace(0, np.nan)
            )
        elif "fc_ratio" in result.columns:
            result["bid_strength"] = result["fc_ratio"] / 100
        else:
            result["bid_strength"] = 0.5
        
        if "fl_ratio" not in result.columns:
            result["fl_ratio"] = 0
        if "open_num" not in result.columns:
            result["open_num"] = 0
        
        result["is_strong_limit"] = (
            (result["fl_ratio"] >= min_fl_ratio) & 
            (result["open_num"] == 0)
        )
        
        result["dragon_score"] = result["bid_strength"].rank(pct=True)
        result.loc[~result["is_strong_limit"], "dragon_score"] *= 0.5
        result["dragon_score"] = result["dragon_score"].clip(0, 1)
        result["dragon_score"] = result["dragon_score"].fillna(0)
        
        output_cols = [
            "stock_code", "ts_code", "name", "close", "pct_chg",
            "fd_amount", "amount", "bid_strength", "fl_ratio", 
            "open_num", "is_strong_limit", "dragon_score"
        ]
        output_cols = [c for c in output_cols if c in result.columns]
        
        result = result[output_cols].copy()
        result = result.sort_values("dragon_score", ascending=False)
        
        strong_count = result["is_strong_limit"].sum()
        logger.info(
            f"✅ 龙头因子计算完成: {len(result)} 只涨停, "
            f"{strong_count} 只强势板, "
            f"top5得分={result['dragon_score'].head(5).mean():.3f}"
        )
        
        return result
    
    def calculate_dragon_head_factor(
        self,
        trade_date: str,
        days_back: int = 5,
        consecutive_weight: float = 0.3,
        strength_weight: float = 0.7
    ) -> pd.DataFrame:
        """
        计算完整龙头信仰因子（含连板溢价）
        
        Parameters
        ----------
        trade_date : str
            交易日期
        days_back : int
            回溯天数
        consecutive_weight : float
            连板溢价权重
        strength_weight : float
            封板强度权重
        
        Returns
        -------
        pd.DataFrame
            龙头信仰因子数据
        """
        trade_date = trade_date.replace("-", "")
        
        logger.info(f"🐲 计算完整龙头信仰因子: {trade_date}")
        
        strength_df = self.calculate_limit_strength(trade_date)
        
        if strength_df.empty:
            return pd.DataFrame()
        
        result = strength_df.copy()
        
        logger.info(f"计算连板天数: {len(result)} 只股票...")
        
        consecutive_days_list = []
        for stock in result["stock_code"]:
            cons_days = self.calculate_consecutive_limits(
                stock, trade_date, days_back=days_back
            )
            consecutive_days_list.append(cons_days)
        
        result["consecutive_days"] = consecutive_days_list
        
        def calc_consecutive_score(days: int) -> float:
            if days <= 0:
                return 0.0
            elif days == 1:
                return 1.0
            elif days == 2:
                return 1.5
            elif days == 3:
                return 2.0
            else:
                return 2.5 + 0.2 * (days - 4)
        
        result["consecutive_premium"] = result["consecutive_days"].apply(calc_consecutive_score)
        
        max_premium = result["consecutive_premium"].max()
        if max_premium > 0:
            result["consecutive_score"] = result["consecutive_premium"] / max_premium
        else:
            result["consecutive_score"] = 0
        
        result["dragon_head_factor"] = (
            strength_weight * result["dragon_score"] +
            consecutive_weight * result["consecutive_score"]
        )
        
        result["dragon_head_factor"] = result["dragon_head_factor"].clip(0, 1)
        result = result.sort_values("dragon_head_factor", ascending=False)
        
        multi_limit = (result["consecutive_days"] >= 2).sum()
        logger.info(
            f"✅ 龙头因子计算完成: "
            f"{len(result)} 只涨停, {multi_limit} 只连板, "
            f"最高连板={result['consecutive_days'].max()}天"
        )
        
        return result
    
    def get_dragon_candidates(
        self,
        trade_date: str,
        min_consecutive: int = 1,
        min_factor: float = 0.6,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        获取龙头候选股
        
        Parameters
        ----------
        trade_date : str
            交易日期
        min_consecutive : int
            最低连板天数
        min_factor : float
            最低龙头因子得分
        top_n : int
            返回股票数量上限
        
        Returns
        -------
        pd.DataFrame
            筛选后的龙头候选股列表
        """
        dragon_df = self.calculate_dragon_head_factor(trade_date)
        
        if dragon_df.empty:
            return pd.DataFrame()
        
        mask = (
            (dragon_df["consecutive_days"] >= min_consecutive) &
            (dragon_df["dragon_head_factor"] >= min_factor) &
            (dragon_df["is_strong_limit"] == True)
        )
        
        candidates = dragon_df[mask].head(top_n).copy()
        
        logger.info(
            f"🎯 龙头候选股筛选完成: "
            f"{len(candidates)} 只 (条件: 连板>={min_consecutive}, 因子>={min_factor})"
        )
        
        return candidates

