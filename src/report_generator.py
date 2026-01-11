"""
报告生成器模块

本模块提供交易报告的生成功能，支持 Markdown 和 HTML 两种格式。
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    交易报告生成器
    
    生成每日调仓报告、历史业绩报告等。
    
    Parameters
    ----------
    config : Dict[str, Any]
        配置参数
    reports_path : Path
        报告输出目录
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        reports_path: Optional[Path] = None
    ):
        self.config = config or {}
        self.reports_path = reports_path or Path("reports")
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # 历史业绩记录路径
        history_config = self.config.get("performance_history", {})
        self.history_path = Path(history_config.get(
            "file_path",
            "data/processed/performance_history.json"
        ))
        
        # 缓存
        self._ic_results: Optional[pd.DataFrame] = None
    
    def set_ic_results(self, ic_results: pd.DataFrame) -> None:
        """设置因子 IC 结果（用于报告）"""
        self._ic_results = ic_results
    
    def generate_markdown_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        target_positions: Dict[str, float],
        strategy_info: Dict[str, Any],
        report_date: str
    ) -> str:
        """
        生成 Markdown 格式报告
        
        Parameters
        ----------
        buy_orders : Dict[str, float]
            买入订单 {股票代码: 金额}
        sell_orders : Dict[str, float]
            卖出订单 {股票代码: 金额}
        target_positions : Dict[str, float]
            目标持仓 {股票代码: 金额}
        strategy_info : Dict[str, Any]
            策略信息
        report_date : str
            报告日期
        
        Returns
        -------
        str
            Markdown 格式报告内容
        """
        lines = [
            f"# 每日调仓报告",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"**报告日期**: {report_date}",
            f"",
            f"---",
            f"",
        ]
        
        # 策略信息
        lines.extend([
            f"## 策略信息",
            f"",
            f"| 参数 | 值 |",
            f"|------|-----|",
            f"| 策略名称 | {strategy_info.get('name', 'N/A')} |",
            f"| 价值因子权重 | {strategy_info.get('value_weight', 0):.0%} |",
            f"| 质量因子权重 | {strategy_info.get('quality_weight', 0):.0%} |",
            f"| 动量因子权重 | {strategy_info.get('momentum_weight', 0):.0%} |",
            f"| 选股数量 | {strategy_info.get('top_n', 5)} |",
            f"",
        ])
        
        # 持仓汇总
        portfolio_config = self.config.get("portfolio", {})
        total_capital = portfolio_config.get("total_capital", 1000000)
        
        lines.extend([
            f"## 持仓汇总",
            f"",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 总资金 | ¥{total_capital:,.0f} |",
            f"| 目标持仓数 | {len(target_positions)} |",
            f"| 买入股票数 | {len(buy_orders)} |",
            f"| 卖出股票数 | {len(sell_orders)} |",
            f"",
        ])
        
        # 买入清单
        lines.extend([
            f"## 📈 明日需买入",
            f"",
        ])
        
        if buy_orders:
            lines.extend([
                f"| 股票代码 | 买入金额 |",
                f"|----------|----------|",
            ])
            for stock, amount in sorted(buy_orders.items(), key=lambda x: -x[1]):
                lines.append(f"| {stock} | ¥{amount:,.0f} |")
            lines.append(f"")
            lines.append(f"**买入总金额**: ¥{sum(buy_orders.values()):,.0f}")
        else:
            lines.append(f"*无需买入*")
        
        lines.append(f"")
        
        # 卖出清单
        lines.extend([
            f"## 📉 明日需卖出",
            f"",
        ])
        
        if sell_orders:
            lines.extend([
                f"| 股票代码 | 卖出金额 |",
                f"|----------|----------|",
            ])
            for stock, amount in sorted(sell_orders.items(), key=lambda x: -x[1]):
                lines.append(f"| {stock} | ¥{amount:,.0f} |")
            lines.append(f"")
            lines.append(f"**卖出总金额**: ¥{sum(sell_orders.values()):,.0f}")
        else:
            lines.append(f"*无需卖出*")
        
        lines.append(f"")
        
        # 目标持仓明细
        lines.extend([
            f"## 目标持仓明细",
            f"",
            f"| 股票代码 | 目标金额 | 权重 |",
            f"|----------|----------|------|",
        ])
        
        total_target = sum(target_positions.values()) if target_positions else 1
        for stock, amount in sorted(target_positions.items(), key=lambda x: -x[1]):
            weight = amount / total_target
            lines.append(f"| {stock} | ¥{amount:,.0f} | {weight:.2%} |")
        
        # 添加因子 IC 监控部分
        ic_section = self._generate_ic_report_section(format="markdown")
        if ic_section:
            lines.append(ic_section)
        
        # 添加历史业绩统计部分
        performance_section = self._generate_performance_report_section(format="markdown")
        if performance_section:
            lines.append(performance_section)
        
        lines.extend([
            f"",
            f"---",
            f"",
            f"*本报告由 A股量化交易系统 自动生成*",
        ])
        
        return "\n".join(lines)
    
    def generate_html_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        target_positions: Dict[str, float],
        strategy_info: Dict[str, Any],
        report_date: str
    ) -> str:
        """生成 HTML 格式报告"""
        portfolio_config = self.config.get("portfolio", {})
        total_capital = portfolio_config.get("total_capital", 1000000)
        
        # 买入表格行
        buy_rows = ""
        for stock, amount in sorted(buy_orders.items(), key=lambda x: -x[1]):
            buy_rows += f"<tr><td>{stock}</td><td>¥{amount:,.0f}</td></tr>"
        
        # 卖出表格行
        sell_rows = ""
        for stock, amount in sorted(sell_orders.items(), key=lambda x: -x[1]):
            sell_rows += f"<tr><td>{stock}</td><td>¥{amount:,.0f}</td></tr>"
        
        # 持仓表格行
        position_rows = ""
        total_target = sum(target_positions.values()) if target_positions else 1
        for stock, amount in sorted(target_positions.items(), key=lambda x: -x[1]):
            weight = amount / total_target
            position_rows += f"<tr><td>{stock}</td><td>¥{amount:,.0f}</td><td>{weight:.2%}</td></tr>"
        
        html = f"""
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
        .container {{ max-width: 1000px; margin: 0 auto; }}
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
        .card.buy h2 {{ color: #00ff88; }}
        .card.sell h2 {{ color: #ff6b6b; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }}
        .stat {{ text-align: center; padding: 1rem; background: rgba(0, 217, 255, 0.1); border-radius: 8px; }}
        .stat-value {{ font-size: 1.5rem; font-weight: bold; color: #00d9ff; }}
        .stat-label {{ font-size: 0.85rem; color: #888; margin-top: 0.25rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }}
        th {{ color: #888; font-weight: 500; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.03); }}
        .total {{ margin-top: 1rem; padding-top: 1rem; border-top: 2px solid rgba(255, 255, 255, 0.1); font-weight: bold; }}
        .buy-total {{ color: #00ff88; }}
        .sell-total {{ color: #ff6b6b; }}
        .footer {{ text-align: center; color: #666; margin-top: 2rem; font-size: 0.85rem; }}
        .empty {{ text-align: center; color: #666; padding: 2rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 每日调仓报告</h1>
        <p class="meta">报告日期: {report_date} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="card">
            <h2>策略概览</h2>
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
                    <div class="stat-value">{len(buy_orders)}</div>
                    <div class="stat-label">买入股票数</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(sell_orders)}</div>
                    <div class="stat-label">卖出股票数</div>
                </div>
            </div>
        </div>
        
        <div class="card buy">
            <h2>📈 明日需买入</h2>
            {f'''
            <table>
                <thead><tr><th>股票代码</th><th>买入金额</th></tr></thead>
                <tbody>{buy_rows}</tbody>
            </table>
            <p class="total buy-total">买入总金额: ¥{sum(buy_orders.values()):,.0f}</p>
            ''' if buy_orders else '<p class="empty">无需买入</p>'}
        </div>
        
        <div class="card sell">
            <h2>📉 明日需卖出</h2>
            {f'''
            <table>
                <thead><tr><th>股票代码</th><th>卖出金额</th></tr></thead>
                <tbody>{sell_rows}</tbody>
            </table>
            <p class="total sell-total">卖出总金额: ¥{sum(sell_orders.values()):,.0f}</p>
            ''' if sell_orders else '<p class="empty">无需卖出</p>'}
        </div>
        
        <div class="card">
            <h2>📋 目标持仓明细</h2>
            <table>
                <thead><tr><th>股票代码</th><th>目标金额</th><th>权重</th></tr></thead>
                <tbody>{position_rows}</tbody>
            </table>
        </div>
        
        {self._generate_ic_report_section(format="html")}
        {self._generate_performance_report_section(format="html")}
        
        <p class="footer">本报告由 A股量化交易系统 自动生成</p>
    </div>
</body>
</html>
        """
        return html
    
    def generate_report(
        self,
        buy_orders: Dict[str, float],
        sell_orders: Dict[str, float],
        target_positions: Dict[str, float],
        strategy_info: Dict[str, Any],
        report_date: str,
        format: str = "markdown"
    ) -> str:
        """
        生成报告
        
        Parameters
        ----------
        format : str
            报告格式，'markdown' 或 'html'
        
        Returns
        -------
        str
            报告内容
        """
        if format == "html":
            return self.generate_html_report(
                buy_orders, sell_orders, target_positions, strategy_info, report_date
            )
        else:
            return self.generate_markdown_report(
                buy_orders, sell_orders, target_positions, strategy_info, report_date
            )
    
    def save_report(self, report_content: str, report_date: str, format: str = "markdown") -> Path:
        """保存报告"""
        extension = "md" if format == "markdown" else "html"
        report_path = self.reports_path / f"daily_report_{report_date}.{extension}"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"报告已保存至 {report_path}")
        return report_path
    
    def _generate_ic_report_section(self, format: str = "markdown") -> str:
        """生成因子 IC 监控报告片段"""
        if self._ic_results is None or self._ic_results.empty:
            return ""
        
        if format == "markdown":
            lines = [
                "",
                "## 📈 因子 IC 监控",
                "",
                "| 因子 | IC均值 | IC标准差 | IC_IR | 正IC占比 | 状态 |",
                "|------|--------|----------|-------|----------|------|",
            ]
            
            for _, row in self._ic_results.iterrows():
                ic_mean = row.get('ic_mean', 0)
                ic_std = row.get('ic_std', 0)
                ic_ir = row.get('ic_ir', 0)
                positive_ratio = row.get('ic_positive_ratio', row.get('positive_ratio', 0))
                
                if abs(ic_mean) >= 0.03:
                    status = "✅ 有效"
                elif abs(ic_mean) >= 0.01:
                    status = "⚠️ 边际"
                else:
                    status = "❌ 失效"
                
                lines.append(
                    f"| {row['factor']} | {ic_mean:.4f} | {ic_std:.4f} | "
                    f"{ic_ir:.2f} | {positive_ratio:.1%} | {status} |"
                )
            
            return "\n".join(lines)
        
        else:  # HTML
            rows = ""
            for _, row in self._ic_results.iterrows():
                ic_mean = row.get('ic_mean', 0)
                ic_std = row.get('ic_std', 0)
                ic_ir = row.get('ic_ir', 0)
                positive_ratio = row.get('ic_positive_ratio', row.get('positive_ratio', 0))
                
                if abs(ic_mean) >= 0.03:
                    status_class = "ic-valid"
                    status = "✅ 有效"
                elif abs(ic_mean) >= 0.01:
                    status_class = "ic-marginal"
                    status = "⚠️ 边际"
                else:
                    status_class = "ic-invalid"
                    status = "❌ 失效"
                
                rows += f"""
                <tr class="{status_class}">
                    <td>{row['factor']}</td>
                    <td>{ic_mean:.4f}</td>
                    <td>{ic_std:.4f}</td>
                    <td>{ic_ir:.2f}</td>
                    <td>{positive_ratio:.1%}</td>
                    <td>{status}</td>
                </tr>
                """
            
            return f"""
            <div class="card">
                <h2>📈 因子 IC 监控</h2>
                <table>
                    <thead>
                        <tr>
                            <th>因子</th>
                            <th>IC均值</th>
                            <th>IC标准差</th>
                            <th>IC_IR</th>
                            <th>正IC占比</th>
                            <th>状态</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """
    
    def _generate_performance_report_section(self, format: str = "markdown") -> str:
        """生成历史业绩报告片段"""
        stats = self.get_performance_stats(30)
        if not stats:
            return ""
        
        if format == "markdown":
            return f"""

## 📊 历史业绩统计（近30个交易日）

| 指标 | 数值 |
|------|------|
| 累计收益 | {stats.get('total_return', 0):.2%} |
| 最大回撤 | {stats.get('max_drawdown', 0):.2%} |
| 夏普比率 | {stats.get('sharpe_ratio', 0):.2f} |
| 日胜率 | {stats.get('win_rate', 0):.1%} |
| 平均日收益 | {stats.get('avg_daily_return', 0):.3%} |
| 日波动率 | {stats.get('volatility', 0):.3%} |
| 交易天数 | {stats.get('trading_days', 0)} |
"""
        
        else:  # HTML
            return f"""
            <div class="card">
                <h2>📊 历史业绩统计（近30个交易日）</h2>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{stats.get('total_return', 0):.2%}</div>
                        <div class="stat-label">累计收益</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{stats.get('max_drawdown', 0):.2%}</div>
                        <div class="stat-label">最大回撤</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{stats.get('sharpe_ratio', 0):.2f}</div>
                        <div class="stat-label">夏普比率</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{stats.get('win_rate', 0):.1%}</div>
                        <div class="stat-label">日胜率</div>
                    </div>
                </div>
            </div>
            """
    
    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """获取历史业绩统计"""
        if not self.history_path.exists():
            return {}
        
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            return {}
        
        if len(history) < 2:
            return {}
        
        sorted_dates = sorted(history.keys())[-days:]
        
        navs = [history[d].get('nav', 1.0) for d in sorted_dates]
        returns = [history[d].get('daily_return', 0.0) for d in sorted_dates]
        
        returns_array = np.array(returns)
        navs_array = np.array(navs)
        
        total_return = navs_array[-1] / navs_array[0] - 1 if navs_array[0] > 0 else 0
        
        peak = np.maximum.accumulate(navs_array)
        drawdown = (navs_array - peak) / peak
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        avg_daily_return = returns_array.mean() if len(returns_array) > 0 else 0
        volatility = returns_array.std() if len(returns_array) > 1 else 0
        
        risk_free = self.config.get("portfolio", {}).get("risk_free_rate", 0.02)
        daily_rf = risk_free / 252
        sharpe_ratio = (avg_daily_return - daily_rf) / volatility * np.sqrt(252) if volatility > 0 else 0
        
        win_rate = (returns_array > 0).mean() if len(returns_array) > 0 else 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'trading_days': len(sorted_dates),
            'nav_series': {d: history[d].get('nav', 1.0) for d in sorted_dates}
        }
    
    def update_performance_history(
        self,
        target_positions: Dict[str, float],
        today: pd.Timestamp
    ) -> None:
        """更新历史业绩记录"""
        history_config = self.config.get("performance_history", {})
        if not history_config.get("enabled", True):
            return
        
        history = {}
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception as e:
                logger.warning(f"加载历史业绩失败: {e}")
        
        today_str = today.strftime('%Y-%m-%d')
        total_value = sum(target_positions.values()) if target_positions else 0
        
        from datetime import timedelta
        yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        yesterday_value = history.get(yesterday, {}).get('total_value', total_value)
        daily_return = (total_value / yesterday_value - 1) if yesterday_value > 0 else 0
        
        initial_capital = self.config.get("portfolio", {}).get("total_capital", 300000)
        nav = total_value / initial_capital if initial_capital > 0 else 1.0
        
        history[today_str] = {
            'nav': nav,
            'total_value': total_value,
            'positions': len(target_positions),
            'daily_return': daily_return
        }
        
        max_days = history_config.get("max_days", 365)
        if len(history) > max_days:
            sorted_dates = sorted(history.keys(), reverse=True)[:max_days]
            history = {k: history[k] for k in sorted_dates}
        
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            logger.info(f"历史业绩已更新: NAV={nav:.4f}, 日收益={daily_return:.2%}")
        except Exception as e:
            logger.warning(f"保存历史业绩失败: {e}")

