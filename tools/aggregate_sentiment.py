#!/usr/bin/env python3
"""
每日情感分数聚合脚本

该脚本按股票代码聚合每日情感分数，并计算 3 日移动平均。
支持异步新闻获取和 GPU 加速推理。

Usage
-----
>>> python tools/aggregate_sentiment.py --stocks 000001,000002 --date 2024-01-15
>>> python tools/aggregate_sentiment.py --index HS300 --start 2024-01-01 --end 2024-01-15

Examples
--------
1. 分析指定股票的单日情感：
   python tools/aggregate_sentiment.py --stocks 000001,600519 --date 2024-01-15

2. 分析沪深300成分股的多日情感：
   python tools/aggregate_sentiment.py --index HS300 --start 2024-01-10 --end 2024-01-15

3. 导出结果到 CSV：
   python tools/aggregate_sentiment.py --stocks 000001 --date 2024-01-15 --output sentiment.csv
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import asyncio
import logging
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# 延迟导入，避免在模块加载时就需要所有依赖
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_stock_list(
    stocks: Optional[str] = None,
    index: Optional[str] = None
) -> List[str]:
    """
    获取股票列表
    
    Parameters
    ----------
    stocks : Optional[str]
        逗号分隔的股票代码
    index : Optional[str]
        指数代码 (HS300, ZZ500, ZZ1000)
    
    Returns
    -------
    List[str]
        股票代码列表
    """
    if stocks:
        return [s.strip() for s in stocks.split(",")]
    
    if index:
        try:
            from src.tushare_loader import TushareDataLoader
            
            index_map = {
                "HS300": "hs300",
                "ZZ500": "zz500",
                "ZZ1000": "zz1000"
            }
            
            index_code = index_map.get(index.upper(), index.lower())
            loader = TushareDataLoader()
            stock_list = loader.fetch_index_constituents(index_code=index_code)
            
            if stock_list:
                return stock_list
            
            logger.warning(f"无法从指数 {index} 获取成分股")
            return []
            
        except Exception as e:
            logger.error(f"获取指数成分股失败: {e}")
            return []
    
    return []


def aggregate_daily_sentiment(
    stock_list: List[str],
    date: str,
    device: str = "auto"
) -> pd.DataFrame:
    """
    聚合单日股票情感分数
    
    Parameters
    ----------
    stock_list : List[str]
        股票代码列表
    date : str
        日期 (YYYY-MM-DD)
    device : str
        计算设备
    
    Returns
    -------
    pd.DataFrame
        情感分析结果
    """
    from src.sentiment_analyzer import SentimentPipeline
    
    pipeline = SentimentPipeline(device=device)
    result = pipeline.analyze_daily(stock_list, date)
    
    return result


def aggregate_multi_day_sentiment(
    stock_list: List[str],
    start_date: str,
    end_date: str,
    ma_window: int = 3,
    device: str = "auto"
) -> pd.DataFrame:
    """
    聚合多日股票情感分数并计算移动平均
    
    Parameters
    ----------
    stock_list : List[str]
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    ma_window : int
        移动平均窗口
    device : str
        计算设备
    
    Returns
    -------
    pd.DataFrame
        包含移动平均的情感分析结果
    """
    from src.sentiment_analyzer import SentimentPipeline
    
    pipeline = SentimentPipeline(device=device)
    result = pipeline.analyze_with_moving_average(
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date,
        ma_window=ma_window
    )
    
    return result


def calculate_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算情感分数摘要统计
    
    Parameters
    ----------
    df : pd.DataFrame
        情感分析结果
    
    Returns
    -------
    pd.DataFrame
        摘要统计
    """
    if df.empty:
        return pd.DataFrame()
    
    summary = df.groupby("stock_code").agg({
        "sentiment_score": ["mean", "std", "min", "max"],
        "confidence": "mean",
        "news_count": "sum"
    }).round(4)
    
    # 展平列名
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in summary.columns
    ]
    summary = summary.reset_index()
    
    return summary


def export_results(
    df: pd.DataFrame,
    output_path: str,
    format: str = "csv"
) -> None:
    """
    导出结果
    
    Parameters
    ----------
    df : pd.DataFrame
        结果数据
    output_path : str
        输出路径
    format : str
        输出格式 (csv, parquet, json)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    logger.info(f"结果已导出: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="按股票代码聚合每日情感分数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 股票选择
    parser.add_argument(
        "--stocks",
        type=str,
        help="逗号分隔的股票代码，如 '000001,000002,600519'"
    )
    parser.add_argument(
        "--index",
        type=str,
        choices=["HS300", "ZZ500", "ZZ1000"],
        help="指数代码，获取成分股"
    )
    
    # 日期范围
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="单日分析日期 (YYYY-MM-DD)，默认今天"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="开始日期 (YYYY-MM-DD)，用于多日分析"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="结束日期 (YYYY-MM-DD)，用于多日分析"
    )
    
    # 移动平均
    parser.add_argument(
        "--ma-window",
        type=int,
        default=3,
        help="移动平均窗口大小，默认 3"
    )
    
    # 设备和性能
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="计算设备，默认 auto"
    )
    
    # 输出
    parser.add_argument(
        "--output",
        type=str,
        help="输出文件路径"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "parquet", "json"],
        help="输出格式，默认 csv"
    )
    
    # 日志
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细日志输出"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.verbose)
    
    # 获取股票列表
    stock_list = get_stock_list(args.stocks, args.index)
    
    if not stock_list:
        logger.error("未指定股票列表。请使用 --stocks 或 --index 参数。")
        sys.exit(1)
    
    logger.info(f"股票列表: {len(stock_list)} 只")
    
    # 执行分析
    if args.start and args.end:
        # 多日分析
        logger.info(f"多日分析: {args.start} ~ {args.end}")
        result = aggregate_multi_day_sentiment(
            stock_list=stock_list,
            start_date=args.start,
            end_date=args.end,
            ma_window=args.ma_window,
            device=args.device
        )
    else:
        # 单日分析
        logger.info(f"单日分析: {args.date}")
        result = aggregate_daily_sentiment(
            stock_list=stock_list,
            date=args.date,
            device=args.device
        )
    
    if result.empty:
        logger.warning("分析结果为空")
        sys.exit(0)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("情感分析结果摘要")
    print("=" * 60)
    
    summary = calculate_sentiment_summary(result)
    print(summary.to_string(index=False))
    
    # 打印详细结果（如果数量不多）
    if len(result) <= 20:
        print("\n" + "-" * 60)
        print("详细结果")
        print("-" * 60)
        cols_to_show = ["stock_code", "date", "sentiment_score", 
                        "confidence", "news_count", "label"]
        cols_to_show = [c for c in cols_to_show if c in result.columns]
        
        # 添加移动平均列
        ma_col = f"sentiment_ma{args.ma_window}"
        if ma_col in result.columns:
            cols_to_show.append(ma_col)
        
        print(result[cols_to_show].to_string(index=False))
    
    # 导出结果
    if args.output:
        export_results(result, args.output, args.format)
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

