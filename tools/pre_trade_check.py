#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易前检查工具

在每天运行 main.py 之前执行此脚本，确保：
1. 数据源（Tushare Pro）可正常访问
2. 系统持仓记录与券商APP实际持仓一致
3. 今天是交易日

Usage
-----
    python tools/pre_trade_check.py

Examples
--------
    # 每日开盘前运行
    python tools/pre_trade_check.py
    
    # 如果检查通过，再运行主程序
    python main.py --daily-update
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
HOLDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "real_holdings.json"


def print_header(title: str) -> None:
    """
    打印分隔标题
    
    Parameters
    ----------
    title : str
        标题文字
    """
    width = 60
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width)


def print_result(success: bool, message: str) -> None:
    """
    打印检查结果
    
    Parameters
    ----------
    success : bool
        检查是否通过
    message : str
        结果消息
    """
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")


def check_data_source() -> Tuple[bool, str]:
    """
    检查数据源是否可访问
    
    尝试从 Tushare Pro 获取沪深300指数数据，验证网络和接口可用性。
    
    Returns
    -------
    Tuple[bool, str]
        (是否成功, 描述信息)
    """
    print_header("数据源检查")
    
    try:
        # 添加项目根目录到路径
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.tushare_loader import TushareDataLoader
        
        print("正在连接 Tushare Pro 获取沪深300指数数据...")
        
        loader = TushareDataLoader()
        
        # 获取近5天的指数数据
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - pd.Timedelta(days=10)).strftime("%Y%m%d")
        
        df = loader.pro.index_daily(
            ts_code="000300.SH",
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            return False, "Tushare 返回空数据"
        
        # 获取最新一条数据
        df = df.sort_values('trade_date', ascending=True)
        latest = df.tail(1).iloc[0]
        latest_date = latest['trade_date']
        latest_close = latest['close']
        
        msg = f"Tushare Pro 连接正常 | 沪深300最新: {latest_date} 收盘 {latest_close:.2f}"
        print_result(True, msg)
        return True, msg
        
    except ImportError as e:
        msg = f"未安装 tushare 或导入失败: {e}"
        print_result(False, msg)
        return False, msg
        
    except Exception as e:
        msg = f"Tushare Pro 连接失败: {e}"
        print_result(False, msg)
        return False, msg


def check_trading_day() -> Tuple[bool, str]:
    """
    检查今天是否是交易日
    
    Returns
    -------
    Tuple[bool, str]
        (是否是交易日, 描述信息)
    """
    print_header("交易日历检查")
    
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    weekday = today.weekday()
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    print(f"当前日期: {today_str} ({weekday_names[weekday]})")
    
    # 先检查是否周末
    if weekday >= 5:
        msg = f"今天是 {weekday_names[weekday]}，非交易日"
        print_result(False, msg)
        return False, msg
    
    # 尝试从 Tushare 获取交易日历
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.tushare_loader import TushareDataLoader
        
        print("正在获取交易日历...")
        
        loader = TushareDataLoader()
        is_trade_day = loader.is_trade_day(today_str.replace("-", ""))
        
        if is_trade_day:
            msg = f"{today_str} 是交易日"
            print_result(True, msg)
            return True, msg
        else:
            msg = f"{today_str} 是节假日，非交易日"
            print_result(False, msg)
            return False, msg
            
    except Exception as e:
        # 降级：仅根据周末判断
        print(f"⚠️  无法获取交易日历: {e}")
        msg = f"{today_str} 是工作日（未验证是否节假日）"
        print_result(True, msg)
        print("   ⚠️  请自行确认今天不是节假日")
        return True, msg


def load_holdings() -> Optional[Dict[str, Any]]:
    """
    加载持仓数据
    
    Returns
    -------
    Optional[Dict[str, Any]]
        持仓数据，如果文件不存在返回 None
    """
    if not HOLDINGS_PATH.exists():
        return None
    
    try:
        with open(HOLDINGS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  读取持仓文件失败: {e}")
        return None


def check_holdings() -> Tuple[bool, str]:
    """
    检查持仓记录并要求用户确认
    
    Returns
    -------
    Tuple[bool, str]
        (用户是否确认一致, 描述信息)
    """
    print_header("持仓记录检查")
    
    holdings = load_holdings()
    
    if holdings is None:
        print_result(False, f"持仓文件不存在: {HOLDINGS_PATH}")
        print("\n💡 建议操作:")
        print(f"   1. 创建持仓文件: python tools/update_holdings.py --show")
        print(f"   2. 或手动创建 {HOLDINGS_PATH}")
        return False, "持仓文件不存在"
    
    # 打印持仓信息
    update_time = holdings.get('update_time', '未知')
    positions = holdings.get('positions', {})
    cash = holdings.get('cash', 0.0)
    total_value = holdings.get('total_value', 0.0)
    
    print(f"\n📁 持仓文件: {HOLDINGS_PATH}")
    print(f"📅 最后更新: {update_time}")
    print("-" * 50)
    
    if not positions:
        print("   (无股票持仓)")
    else:
        print(f"{'股票代码':<12} {'持仓市值':>15} {'占比':>10}")
        print("-" * 50)
        
        # 计算总市值（不含现金）
        stock_total = sum(positions.values())
        
        for stock_code, amount in sorted(positions.items(), key=lambda x: -x[1]):
            pct = amount / stock_total * 100 if stock_total > 0 else 0
            print(f"{stock_code:<12} ¥{amount:>14,.0f} {pct:>9.1f}%")
    
    print("-" * 50)
    print(f"{'股票市值合计':<12} ¥{sum(positions.values()):>14,.0f}")
    print(f"{'可用现金':<12} ¥{cash:>14,.0f}")
    print(f"{'账户总资产':<12} ¥{total_value:>14,.0f}")
    print("=" * 50)
    
    # 要求用户确认
    print("\n⚠️  请核对以上持仓与您券商APP中的【实际持仓】是否一致")
    print("   （包括股票代码、持仓市值、可用现金）")
    
    while True:
        try:
            user_input = input("\n是否一致？[y/n/q]: ").strip().lower()
            
            if user_input == 'y':
                msg = "用户确认持仓一致"
                print_result(True, msg)
                return True, msg
                
            elif user_input == 'n':
                print_result(False, "用户确认持仓不一致")
                print("\n💡 建议操作:")
                print("   1. 使用工具更新持仓:")
                print("      python tools/update_holdings.py --stock <代码> --amount <金额>")
                print("      python tools/update_holdings.py --cash <现金>")
                print(f"   2. 或直接编辑文件: {HOLDINGS_PATH}")
                print("   3. 修改后重新运行此检查脚本")
                return False, "持仓不一致"
                
            elif user_input == 'q':
                print("用户取消检查")
                return False, "用户取消"
                
            else:
                print("   请输入 y (一致) / n (不一致) / q (退出)")
                
        except KeyboardInterrupt:
            print("\n\n用户中断")
            return False, "用户中断"


def run_all_checks() -> bool:
    """
    运行所有检查
    
    Returns
    -------
    bool
        所有检查是否通过
    """
    print("\n" + "🔍 " + "=" * 56)
    print("          交易前检查  Pre-Trade Checklist")
    print("🔍 " + "=" * 56)
    print(f"   检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # 1. 检查数据源
    success, msg = check_data_source()
    results.append(("数据源", success))
    
    # 2. 检查交易日
    success, msg = check_trading_day()
    results.append(("交易日历", success))
    
    # 3. 检查持仓（如果不是交易日，可跳过）
    is_trading_day = results[-1][1]
    if is_trading_day:
        success, msg = check_holdings()
        results.append(("持仓确认", success))
    else:
        print_header("持仓记录检查")
        print("ℹ️  非交易日，跳过持仓确认")
        results.append(("持仓确认", True))  # 非交易日默认通过
    
    # 汇总结果
    print_header("检查结果汇总")
    
    all_passed = True
    for name, passed in results:
        icon = "✅" if passed else "❌"
        status = "通过" if passed else "未通过"
        print(f"  {icon} {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有检查通过！可以运行 main.py")
        print("   python main.py --daily-update")
    else:
        print("⚠️  部分检查未通过，请先解决问题后再运行 main.py")
    
    print()
    return all_passed


def main() -> None:
    """
    主函数
    """
    try:
        all_passed = run_all_checks()
        sys.exit(0 if all_passed else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断检查")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 检查过程发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

