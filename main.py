#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股量化交易系统 - 主入口

该模块作为系统的主入口点，提供每日更新和回测功能。
所有核心逻辑已模块化到 src/ 目录下。

Usage
-----
    # 运行每日更新
    python main.py --daily-update
    
    # 强制调仓（忽略日期检查）
    python main.py --daily-update --force-rebalance
    
    # 生成回测报告
    python main.py --backtest --start 2023-01-01 --end 2024-01-01
    
    # 使用模块化 CLI (推荐)
    python -m src.cli --daily-update
    python -m src.cli --backtest
    
Notes
-----
遗留代码已移至 main_legacy.py，新开发请使用 src/ 模块。
"""
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """主函数 - 委托给 src.cli 模块"""
    try:
        from src.cli import main as cli_main
        cli_main()
    except ImportError as e:
        print(f"错误: 无法导入 CLI 模块: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n用户中断，退出...")
        sys.exit(0)
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

