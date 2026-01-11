"""
命令行接口入口

本模块提供量化交易系统的 CLI 接口。
原始的 main.py 保留完整功能，本文件提供基于新模块结构的精简接口。
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 确保目录存在
LOGS_PATH = Path("logs")
LOGS_PATH.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO, log_file: str = None):
    """配置日志"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="A股量化交易系统 (重构版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m src.cli --daily-update              # 运行每日更新
    python -m src.cli --daily-update --force      # 强制调仓
    python -m src.cli --backtest                  # 运行回测
    python -m src.cli --backtest --start 2022-01-01 --end 2023-12-31
        """
    )
    
    parser.add_argument(
        "--daily-update", "-d",
        action="store_true",
        help="运行每日更新流程"
    )
    
    parser.add_argument(
        "--force-rebalance", "-f",
        action="store_true",
        help="强制调仓（忽略日期检查）"
    )
    
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="运行回测"
    )
    
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="multi_factor",
        choices=["multi_factor", "ma_cross"],
        help="回测策略类型"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="回测开始日期 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2024-01-01",
        help="回测结束日期 (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="禁用 LLM 风控功能"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/strategy_config.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    log_file = LOGS_PATH / f"quant_{datetime.now().strftime('%Y%m%d')}.log"
    setup_logging(level=log_level, log_file=str(log_file))
    
    logger = logging.getLogger(__name__)
    logger.info("A股量化交易系统启动 (重构版)")
    
    # 加载配置
    config = None
    config_path = Path(args.config)
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置已加载: {config_path}")
        except Exception as e:
            logger.warning(f"加载配置失败: {e}")
    
    if config is None:
        config = {}
    
    # 处理 LLM 禁用
    if args.no_llm:
        config["llm"] = {}
    
    if args.daily_update:
        from .daily_runner import run_daily_update
        
        success = run_daily_update(
            force_rebalance=args.force_rebalance,
            config=config
        )
        exit(0 if success else 1)
    
    elif args.backtest:
        from .backtest_runner import run_backtest
        
        logger.info(f"回测模式: {args.start} ~ {args.end}, 策略: {args.strategy}")
        success = run_backtest(
            start_date=args.start,
            end_date=args.end,
            config=config,
            strategy_type=args.strategy,
            no_llm=args.no_llm
        )
        exit(0 if success else 1)
    
    else:
        parser.print_help()
        exit(0)


if __name__ == "__main__":
    main()

