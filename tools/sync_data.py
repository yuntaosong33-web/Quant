#!/usr/bin/env python3
"""
数据同步脚本

盘后运行的独立脚本，用于从 Tushare Pro 下载数据并存储到本地数据湖。
支持增量更新和并发下载。

Usage
-----
    # 同步沪深300成分股
    python tools/sync_data.py

    # 同步指定股票
    python tools/sync_data.py --symbols 000001 000002 600519

    # 全量重建（忽略本地数据）
    python tools/sync_data.py --full-rebuild

Examples
--------
    >>> # 作为模块导入
    >>> from tools.sync_data import DataSyncer
    >>> syncer = DataSyncer()
    >>> syncer.sync_all()
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import argparse
import logging
import sys
import time

import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, DataStandardizer

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """
    同步结果数据类
    
    Attributes
    ----------
    symbol : str
        股票代码
    success : bool
        是否成功
    source : Optional[str]
        数据来源（tushare）
    rows_added : int
        新增行数
    error : Optional[str]
        错误信息
    """
    symbol: str
    success: bool
    source: Optional[str] = None
    rows_added: int = 0
    error: Optional[str] = None


class DataSyncer:
    """
    数据同步器
    
    从 Tushare Pro 下载 A 股日线数据，
    支持增量更新并存储为 Parquet 格式。
    
    Attributes
    ----------
    config : Dict[str, Any]
        配置字典
    storage_path : Path
        数据湖存储路径
    tushare_token : str
        Tushare Pro token
    standardizer : DataStandardizer
        数据标准化器
    
    Examples
    --------
    >>> syncer = DataSyncer()
    >>> 
    >>> # 同步单只股票
    >>> result = syncer.sync_symbol("000001")
    >>> print(f"同步结果: {result.success}, 新增 {result.rows_added} 行")
    >>> 
    >>> # 批量同步
    >>> results = syncer.sync_all()
    >>> success_count = sum(1 for r in results if r.success)
    >>> print(f"成功: {success_count}/{len(results)}")
    """
    
    def __init__(
        self,
        config_path: str = "config/data_config.yaml"
    ) -> None:
        """
        初始化数据同步器
        
        Parameters
        ----------
        config_path : str, optional
            配置文件路径，默认 'config/data_config.yaml'
        """
        # 加载配置
        self.config = load_config(config_path)
        
        # 数据湖路径
        lake_config = self.config.get("lake", {})
        self.storage_path = Path(lake_config.get("storage_path", "data/lake/daily"))
        self.compression = lake_config.get("compression", "snappy")
        
        # 确保目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Tushare 配置
        tushare_config = self.config.get("tushare", {})
        self.tushare_token = tushare_config.get("token", "")
        self._ts_api = None
        
        # 同步配置
        sync_config = self.config.get("sync", {})
        self.start_date = sync_config.get("start_date", "2020-01-01")
        self.max_workers = sync_config.get("max_workers", 3)
        self.batch_size = sync_config.get("batch_size", 50)
        
        # 数据源配置
        data_source_config = self.config.get("data_source", {})
        self.retry_times = data_source_config.get("retry_times", 3)
        self.retry_delay = data_source_config.get("retry_delay", 3)
        self.request_delay = data_source_config.get("request_delay", 0.5)
        
        # 标准化器
        self.standardizer = DataStandardizer()
        
        logger.info(
            f"DataSyncer 初始化完成: "
            f"storage_path={self.storage_path}, "
            f"max_workers={self.max_workers}"
        )
    
    def _get_tushare_api(self) -> Optional[Any]:
        """
        获取 Tushare Pro API 实例
        
        Returns
        -------
        Optional[Any]
            Tushare Pro API 实例，如果 token 未配置则返回 None
        """
        if self._ts_api is not None:
            return self._ts_api
        
        if not self.tushare_token:
            logger.error("Tushare token 未配置")
            return None
        
        try:
            import tushare as ts
            self._ts_api = ts.pro_api(self.tushare_token)
            logger.info("Tushare Pro API 初始化成功")
            return self._ts_api
        except ImportError:
            logger.error("tushare 库未安装，请运行: pip install tushare")
            return None
        except Exception as e:
            logger.error(f"Tushare API 初始化失败: {e}")
            return None
    
    def get_stock_pool(self) -> List[str]:
        """
        获取股票池
        
        从 Tushare 读取指数成分股，作为同步目标。
        
        Returns
        -------
        List[str]
            股票代码列表
        """
        from src.tushare_loader import TushareDataLoader
        
        universe_config = self.config.get("universe", {})
        index_codes = universe_config.get("index_codes", ["hs300"])
        
        # 初始化 Tushare 加载器
        loader = TushareDataLoader(api_token=self.tushare_token)
        
        all_symbols = set()
        
        for index_code in index_codes:
            logger.info(f"获取指数 {index_code} 成分股...")
            try:
                symbols = loader.fetch_index_constituents(index_code=index_code)
                
                if symbols:
                    all_symbols.update(symbols)
                    logger.info(f"指数 {index_code}: {len(symbols)} 只股票")
            except Exception as e:
                logger.error(f"获取指数 {index_code} 成分股失败: {e}")
        
        result = sorted(list(all_symbols))
        logger.info(f"股票池共 {len(result)} 只股票")
        return result
    
    def get_local_max_date(self, symbol: str) -> Optional[datetime]:
        """
        获取本地数据的最大日期
        
        Parameters
        ----------
        symbol : str
            股票代码
        
        Returns
        -------
        Optional[datetime]
            本地数据的最大日期，如果文件不存在返回 None
        """
        filepath = self.storage_path / f"{symbol}.parquet"
        
        if not filepath.exists():
            return None
        
        try:
            df = pd.read_parquet(filepath)
            if df.empty:
                return None
            
            # 获取最大日期
            if isinstance(df.index, pd.DatetimeIndex):
                max_date = df.index.max()
            elif 'date' in df.columns:
                max_date = pd.to_datetime(df['date']).max()
            else:
                return None
            
            return max_date.to_pydatetime()
        except Exception as e:
            logger.warning(f"读取本地文件 {filepath} 失败: {e}")
            return None
    
    def _fetch_from_tushare(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        从 Tushare Pro 获取数据
        
        Parameters
        ----------
        symbol : str
            股票代码（纯数字，如 '000001'）
        start_date : str
            开始日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        end_date : str
            结束日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        
        Returns
        -------
        Optional[pd.DataFrame]
            日线数据，失败返回 None
        """
        api = self._get_tushare_api()
        if api is None:
            return None
        
        # 转换股票代码格式（Tushare 需要 000001.SZ 格式）
        if symbol.startswith(('6', '5')):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        # 格式化日期（Tushare 需要 YYYYMMDD 格式）
        start_date_fmt = start_date.replace("-", "")
        end_date_fmt = end_date.replace("-", "")
        
        for attempt in range(self.retry_times):
            try:
                # 获取日线行情
                df = api.daily(
                    ts_code=ts_code,
                    start_date=start_date_fmt,
                    end_date=end_date_fmt
                )
                
                if df is None or df.empty:
                    logger.debug(f"Tushare {symbol} 无数据")
                    return None
                
                # 获取复权因子
                try:
                    adj_df = api.adj_factor(
                        ts_code=ts_code,
                        start_date=start_date_fmt,
                        end_date=end_date_fmt
                    )
                    if adj_df is not None and not adj_df.empty:
                        df = df.merge(
                            adj_df[['trade_date', 'adj_factor']],
                            on='trade_date',
                            how='left'
                        )
                except Exception as e:
                    logger.debug(f"获取复权因子失败: {e}")
                
                # 标准化列名
                df = self.standardizer.standardize(df, source='tushare')
                
                logger.debug(f"Tushare {symbol} 获取成功: {len(df)} 行")
                return df
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # 检查是否为限流错误
                if 'limit' in error_msg or '每分钟' in error_msg:
                    wait_time = 60  # 限流等待1分钟
                    logger.warning(f"Tushare 限流，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                else:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(
                        f"Tushare {symbol} 请求失败 "
                        f"(尝试 {attempt + 1}/{self.retry_times}): {e}"
                    )
                    if attempt < self.retry_times - 1:
                        time.sleep(wait_time)
        
        return None
    
    
    def sync_symbol(
        self,
        symbol: str,
        full_rebuild: bool = False
    ) -> SyncResult:
        """
        同步单只股票数据
        
        Parameters
        ----------
        symbol : str
            股票代码
        full_rebuild : bool, optional
            是否全量重建（忽略本地数据），默认 False
        
        Returns
        -------
        SyncResult
            同步结果
        """
        try:
            # 确定日期范围
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            if full_rebuild:
                start_date = self.start_date
                logger.debug(f"{symbol}: 全量重建模式，起始日期 {start_date}")
            else:
                # 增量更新：检查本地数据
                local_max_date = self.get_local_max_date(symbol)
                
                if local_max_date is not None:
                    # 从本地最大日期的下一天开始
                    start_date = (local_max_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    
                    # 检查是否需要更新
                    if start_date > end_date:
                        logger.debug(f"{symbol}: 数据已是最新")
                        return SyncResult(
                            symbol=symbol,
                            success=True,
                            rows_added=0
                        )
                    
                    logger.debug(f"{symbol}: 增量更新，起始日期 {start_date}")
                else:
                    start_date = self.start_date
                    logger.debug(f"{symbol}: 本地无数据，起始日期 {start_date}")
            
            # 从 Tushare 获取数据
            df = None
            source = None
            
            df = self._fetch_from_tushare(symbol, start_date, end_date)
            if df is not None and not df.empty:
                source = "tushare"
            
            if df is None or df.empty:
                return SyncResult(
                    symbol=symbol,
                    success=False,
                    error="无数据"
                )
            
            # 保存/追加数据
            rows_added = self._save_data(symbol, df, full_rebuild)
            
            # 请求间隔
            time.sleep(self.request_delay)
            
            return SyncResult(
                symbol=symbol,
                success=True,
                source=source,
                rows_added=rows_added
            )
            
        except Exception as e:
            logger.error(f"{symbol} 同步异常: {e}")
            return SyncResult(
                symbol=symbol,
                success=False,
                error=str(e)
            )
    
    def _save_data(
        self,
        symbol: str,
        new_df: pd.DataFrame,
        full_rebuild: bool
    ) -> int:
        """
        保存/追加数据到 Parquet 文件
        
        Parameters
        ----------
        symbol : str
            股票代码
        new_df : pd.DataFrame
            新数据
        full_rebuild : bool
            是否全量重建
        
        Returns
        -------
        int
            新增行数
        """
        filepath = self.storage_path / f"{symbol}.parquet"
        
        # 确保索引格式正确
        if not isinstance(new_df.index, pd.DatetimeIndex):
            if 'date' in new_df.columns:
                new_df = new_df.set_index('date')
                new_df.index = pd.to_datetime(new_df.index)
        
        new_df = new_df.sort_index()
        rows_added = len(new_df)
        
        if full_rebuild or not filepath.exists():
            # 全量写入
            new_df.to_parquet(filepath, compression=self.compression)
        else:
            # 增量追加：读取现有数据并合并
            try:
                existing_df = pd.read_parquet(filepath)
                
                # 确保索引格式一致
                if not isinstance(existing_df.index, pd.DatetimeIndex):
                    if 'date' in existing_df.columns:
                        existing_df = existing_df.set_index('date')
                    existing_df.index = pd.to_datetime(existing_df.index)
                
                # 合并数据（去重）
                merged_df = pd.concat([existing_df, new_df])
                merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                merged_df = merged_df.sort_index()
                
                rows_added = len(merged_df) - len(existing_df)
                
                # 保存
                merged_df.to_parquet(filepath, compression=self.compression)
                
            except Exception as e:
                logger.warning(f"合并数据失败，执行全量写入: {e}")
                new_df.to_parquet(filepath, compression=self.compression)
        
        logger.debug(f"{symbol}: 保存 {rows_added} 行到 {filepath}")
        return rows_added
    
    def sync_all(
        self,
        symbols: Optional[List[str]] = None,
        full_rebuild: bool = False
    ) -> List[SyncResult]:
        """
        批量同步所有股票
        
        使用线程池并发下载，包含进度条显示。
        
        Parameters
        ----------
        symbols : Optional[List[str]]
            股票代码列表，如果为 None 则使用股票池
        full_rebuild : bool, optional
            是否全量重建，默认 False
        
        Returns
        -------
        List[SyncResult]
            同步结果列表
        """
        from tqdm import tqdm
        
        if symbols is None:
            symbols = self.get_stock_pool()
        
        total = len(symbols)
        logger.info(
            f"开始同步 {total} 只股票, "
            f"max_workers={self.max_workers}, "
            f"full_rebuild={full_rebuild}"
        )
        
        results: List[SyncResult] = []
        success_count = 0
        failed_count = 0
        
        # 使用线程池并发同步
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(self.sync_symbol, symbol, full_rebuild): symbol
                for symbol in symbols
            }
            
            # 进度条
            with tqdm(total=total, desc="同步进度", unit="只") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        results.append(result)
                        
                        if result.success:
                            success_count += 1
                            if result.rows_added > 0:
                                pbar.set_postfix({
                                    "当前": symbol,
                                    "来源": result.source or "缓存",
                                    "新增": result.rows_added
                                })
                        else:
                            failed_count += 1
                            logger.warning(f"{symbol} 同步失败: {result.error}")
                            
                    except TimeoutError:
                        failed_count += 1
                        results.append(SyncResult(
                            symbol=symbol,
                            success=False,
                            error="超时"
                        ))
                        logger.warning(f"{symbol} 同步超时")
                        
                    except Exception as e:
                        failed_count += 1
                        results.append(SyncResult(
                            symbol=symbol,
                            success=False,
                            error=str(e)
                        ))
                        logger.error(f"{symbol} 同步异常: {e}")
                    
                    pbar.update(1)
        
        # 统计
        total_rows = sum(r.rows_added for r in results if r.success)
        tushare_count = sum(1 for r in results if r.source == "tushare")
        
        logger.info(
            f"同步完成: 成功 {success_count}/{total}, "
            f"失败 {failed_count}, "
            f"新增 {total_rows} 行"
        )
        
        return results
    
    def get_sync_status(self) -> pd.DataFrame:
        """
        获取同步状态报告
        
        Returns
        -------
        pd.DataFrame
            同步状态报告
        """
        files = list(self.storage_path.glob("*.parquet"))
        
        records = []
        for filepath in files:
            symbol = filepath.stem
            try:
                df = pd.read_parquet(filepath)
                
                if isinstance(df.index, pd.DatetimeIndex):
                    min_date = df.index.min()
                    max_date = df.index.max()
                elif 'date' in df.columns:
                    min_date = pd.to_datetime(df['date']).min()
                    max_date = pd.to_datetime(df['date']).max()
                else:
                    min_date = max_date = None
                
                records.append({
                    'symbol': symbol,
                    'rows': len(df),
                    'min_date': min_date,
                    'max_date': max_date,
                    'file_size_mb': filepath.stat().st_size / 1024 / 1024
                })
            except Exception as e:
                records.append({
                    'symbol': symbol,
                    'rows': 0,
                    'min_date': None,
                    'max_date': None,
                    'file_size_mb': 0,
                    'error': str(e)
                })
        
        return pd.DataFrame(records)


def main() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="A股日线数据同步脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 同步股票池中的所有股票
    python tools/sync_data.py
    
    # 同步指定股票
    python tools/sync_data.py --symbols 000001 000002 600519
    
    # 全量重建（忽略本地数据）
    python tools/sync_data.py --full-rebuild
    
    # 查看同步状态
    python tools/sync_data.py --status
        """
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="指定要同步的股票代码列表"
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="全量重建（忽略本地已有数据）"
    )
    parser.add_argument(
        "--config",
        default="config/data_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="显示同步状态报告"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试日志"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level, log_file="logs/sync_data.log")
    
    # 创建同步器
    syncer = DataSyncer(config_path=args.config)
    
    if args.status:
        # 显示状态报告
        status_df = syncer.get_sync_status()
        print("\n=== 数据同步状态 ===")
        print(f"共 {len(status_df)} 只股票")
        print(f"总行数: {status_df['rows'].sum():,}")
        print(f"总大小: {status_df['file_size_mb'].sum():.2f} MB")
        print("\n最近更新的股票:")
        print(status_df.nlargest(10, 'max_date')[
            ['symbol', 'rows', 'min_date', 'max_date', 'file_size_mb']
        ].to_string(index=False))
    else:
        # 执行同步
        results = syncer.sync_all(
            symbols=args.symbols,
            full_rebuild=args.full_rebuild
        )
        
        # 输出失败列表
        failed = [r for r in results if not r.success]
        if failed:
            print(f"\n失败的股票 ({len(failed)}):")
            for r in failed[:20]:  # 最多显示20个
                print(f"  {r.symbol}: {r.error}")


if __name__ == "__main__":
    main()

