#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
持仓校准工具

用于手动校准系统的持仓记录 data/processed/real_holdings.json。

Usage
-----
    # 设置某只股票的持仓市值
    python tools/update_holdings.py --stock 600519 --amount 50000
    
    # 设置多只股票
    python tools/update_holdings.py --stock 600519 --amount 50000 --stock 000001 --amount 30000
    
    # 设置可用现金
    python tools/update_holdings.py --cash 100000
    
    # 删除某只股票的持仓（设置金额为0）
    python tools/update_holdings.py --stock 600519 --amount 0
    
    # 清空所有持仓
    python tools/update_holdings.py --clear
    
    # 查看当前持仓
    python tools/update_holdings.py --show

Examples
--------
    # 持有5万元茅台 + 3万元平安银行 + 10万现金
    python tools/update_holdings.py --stock 600519 --amount 50000 --stock 000001 --amount 30000 --cash 100000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


# 持仓文件路径
HOLDINGS_PATH = Path(__file__).parent.parent / "data" / "processed" / "real_holdings.json"


def load_holdings() -> Dict[str, Any]:
    """
    加载当前持仓数据
    
    Returns
    -------
    Dict[str, Any]
        持仓数据字典
    """
    if HOLDINGS_PATH.exists():
        try:
            with open(HOLDINGS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  持仓文件格式错误，将创建新文件")
            return create_empty_holdings()
    else:
        print(f"📝 持仓文件不存在，将创建新文件")
        return create_empty_holdings()


def create_empty_holdings() -> Dict[str, Any]:
    """
    创建空的持仓数据结构
    
    Returns
    -------
    Dict[str, Any]
        空的持仓数据字典
    """
    return {
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "update_date": datetime.now().strftime("%Y-%m-%d"),
        "positions": {},
        "cash": 0.0,
        "total_value": 0.0,
        "num_stocks": 0,
        "note": "此文件由 update_holdings.py 工具手动更新"
    }


def save_holdings(data: Dict[str, Any]) -> None:
    """
    保存持仓数据
    
    Parameters
    ----------
    data : Dict[str, Any]
        持仓数据字典
    """
    # 确保目录存在
    HOLDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 更新时间戳
    data["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["update_date"] = datetime.now().strftime("%Y-%m-%d")
    
    # 计算汇总信息
    positions = data.get("positions", {})
    cash = data.get("cash", 0.0)
    
    data["num_stocks"] = len(positions)
    data["total_value"] = sum(positions.values()) + cash
    
    with open(HOLDINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 持仓已保存至: {HOLDINGS_PATH}")


def print_holdings(data: Dict[str, Any]) -> None:
    """
    打印持仓明细
    
    Parameters
    ----------
    data : Dict[str, Any]
        持仓数据字典
    """
    print("\n" + "=" * 50)
    print("📊 当前持仓明细")
    print("=" * 50)
    
    positions = data.get("positions", {})
    cash = data.get("cash", 0.0)
    total_value = data.get("total_value", 0.0)
    
    if not positions and cash <= 0:
        print("\n  (空仓)")
    else:
        # 打印股票持仓
        if positions:
            print("\n🏦 股票持仓:")
            print("-" * 40)
            print(f"{'股票代码':<12} {'市值':>15} {'占比':>10}")
            print("-" * 40)
            
            for stock, amount in sorted(positions.items(), key=lambda x: -x[1]):
                pct = amount / total_value * 100 if total_value > 0 else 0
                print(f"{stock:<12} ¥{amount:>13,.0f} {pct:>9.1f}%")
            
            print("-" * 40)
            print(f"{'股票小计':<12} ¥{sum(positions.values()):>13,.0f}")
        
        # 打印现金
        if cash > 0:
            print(f"\n💵 可用现金: ¥{cash:,.0f}")
    
    # 打印汇总
    print("\n" + "-" * 40)
    print(f"📈 持仓股票数: {len(positions)} 只")
    print(f"💰 总市值:     ¥{total_value:,.0f}")
    print(f"🕐 更新时间:   {data.get('update_time', 'N/A')}")
    print("=" * 50 + "\n")


def update_stock(
    data: Dict[str, Any],
    stock_code: str,
    amount: float
) -> None:
    """
    更新股票持仓
    
    Parameters
    ----------
    data : Dict[str, Any]
        持仓数据字典
    stock_code : str
        股票代码
    amount : float
        持仓市值（0表示清仓）
    """
    if "positions" not in data:
        data["positions"] = {}
    
    # 标准化股票代码（移除前缀后缀，保留6位数字）
    stock_code = stock_code.strip()
    if len(stock_code) > 6:
        # 处理如 600519.SH 或 SH600519 格式
        import re
        match = re.search(r'\d{6}', stock_code)
        if match:
            stock_code = match.group()
    
    if amount <= 0:
        # 清仓该股票
        if stock_code in data["positions"]:
            del data["positions"][stock_code]
            print(f"🗑️  已清仓: {stock_code}")
        else:
            print(f"⚠️  股票 {stock_code} 不在持仓中")
    else:
        old_amount = data["positions"].get(stock_code, 0)
        data["positions"][stock_code] = amount
        
        if old_amount > 0:
            diff = amount - old_amount
            sign = "+" if diff >= 0 else ""
            print(f"📝 更新持仓: {stock_code} ¥{old_amount:,.0f} → ¥{amount:,.0f} ({sign}{diff:,.0f})")
        else:
            print(f"➕ 新增持仓: {stock_code} ¥{amount:,.0f}")


def update_cash(data: Dict[str, Any], cash: float) -> None:
    """
    更新可用现金
    
    Parameters
    ----------
    data : Dict[str, Any]
        持仓数据字典
    cash : float
        可用现金金额
    """
    old_cash = data.get("cash", 0.0)
    data["cash"] = max(0, cash)  # 不允许负数
    
    if old_cash > 0:
        diff = cash - old_cash
        sign = "+" if diff >= 0 else ""
        print(f"💵 更新现金: ¥{old_cash:,.0f} → ¥{cash:,.0f} ({sign}{diff:,.0f})")
    else:
        print(f"💵 设置现金: ¥{cash:,.0f}")


def clear_holdings(data: Dict[str, Any]) -> None:
    """
    清空所有持仓
    
    Parameters
    ----------
    data : Dict[str, Any]
        持仓数据字典
    """
    data["positions"] = {}
    data["cash"] = 0.0
    print("🗑️  已清空所有持仓和现金")


def parse_stock_amount_pairs(args: argparse.Namespace) -> List[tuple]:
    """
    解析股票和金额配对
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    
    Returns
    -------
    List[tuple]
        (股票代码, 金额) 配对列表
    """
    stocks = args.stock or []
    amounts = args.amount or []
    
    if len(stocks) != len(amounts):
        print(f"❌ 错误: --stock 和 --amount 参数数量不匹配")
        print(f"   股票数量: {len(stocks)}, 金额数量: {len(amounts)}")
        sys.exit(1)
    
    return list(zip(stocks, amounts))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="持仓校准工具 - 手动更新 real_holdings.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 查看当前持仓
    python tools/update_holdings.py --show
    
    # 设置单只股票持仓
    python tools/update_holdings.py --stock 600519 --amount 50000
    
    # 设置多只股票
    python tools/update_holdings.py --stock 600519 --amount 50000 --stock 000001 --amount 30000
    
    # 设置可用现金
    python tools/update_holdings.py --cash 100000
    
    # 清仓某只股票
    python tools/update_holdings.py --stock 600519 --amount 0
    
    # 清空所有持仓
    python tools/update_holdings.py --clear
        """
    )
    
    parser.add_argument(
        "--stock", "-s",
        action="append",
        metavar="CODE",
        help="股票代码（可多次使用）"
    )
    
    parser.add_argument(
        "--amount", "-a",
        action="append",
        type=float,
        metavar="VALUE",
        help="持仓市值（与 --stock 配对使用，0 表示清仓）"
    )
    
    parser.add_argument(
        "--cash", "-c",
        type=float,
        metavar="VALUE",
        help="设置可用现金"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="清空所有持仓和现金"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="仅显示当前持仓（不做修改）"
    )
    
    args = parser.parse_args()
    
    # 如果没有任何参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # 加载现有持仓
    data = load_holdings()
    
    # 仅显示模式
    if args.show:
        print_holdings(data)
        return
    
    # 清空持仓
    if args.clear:
        confirm = input("⚠️  确认清空所有持仓？[y/N]: ").strip().lower()
        if confirm == 'y':
            clear_holdings(data)
            save_holdings(data)
            print_holdings(data)
        else:
            print("❌ 已取消")
        return
    
    # 标记是否有更新
    has_update = False
    
    # 更新股票持仓
    if args.stock:
        pairs = parse_stock_amount_pairs(args)
        for stock, amount in pairs:
            update_stock(data, stock, amount)
            has_update = True
    
    # 更新现金
    if args.cash is not None:
        update_cash(data, args.cash)
        has_update = True
    
    # 保存并显示
    if has_update:
        save_holdings(data)
        print_holdings(data)
    else:
        print("⚠️  没有指定任何更新操作，使用 --help 查看帮助")


if __name__ == "__main__":
    main()

