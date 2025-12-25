# A股量化交易系统 (A-Share Quant System)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于 Python 的 A 股量化交易框架，提供数据获取、因子计算、策略回测和绩效分析等功能。

## ✨ 特性

- 📊 **数据获取**: 基于 AkShare 的 A 股数据获取，支持日线、基本面等多种数据
- 🔢 **因子计算**: 丰富的技术指标库，支持自定义因子扩展
- 📈 **策略框架**: 灵活的策略抽象接口，支持均线、RSI、组合策略等
- ⚡ **高性能回测**: 基于 VectorBT 的向量化回测引擎
- 🛠️ **可扩展性**: 使用抽象基类设计，便于扩展新策略和数据源

## 📁 项目结构

```
ashare_quant_system/
├── config/                    # 配置文件
│   ├── strategy_config.yaml   # 策略配置
│   └── data_config.yaml       # 数据配置
├── data/                      # 数据存储
│   ├── raw/                   # 原始数据 (Parquet格式)
│   └── processed/             # 清洗后的特征数据
├── src/                       # 源代码
│   ├── __init__.py           
│   ├── data_loader.py         # 数据获取与ETL类
│   ├── features.py            # 因子计算引擎
│   ├── strategy.py            # 策略逻辑实现
│   ├── backtest.py            # VectorBT回测流程
│   └── utils.py               # 通用工具函数
├── notebooks/                 # Jupyter Notebooks
│   └── exploratory_analysis.ipynb
├── tests/                     # 单元测试
├── pyproject.toml             # 依赖管理
└── README.md
```

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/ashare-quant-system.git
cd ashare-quant-system

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .

# 安装开发依赖（可选）
pip install -e ".[dev,notebook]"
```

### 基础使用

```python
from src import (
    AkshareDataLoader,
    TechnicalFeatures,
    MACrossStrategy,
    BacktestEngine,
    load_config
)

# 加载配置
data_config = load_config("config/data_config.yaml")
strategy_config = load_config("config/strategy_config.yaml")

# 获取数据
loader = AkshareDataLoader(data_config)
data = loader.fetch_daily_data("000001", "2023-01-01", "2024-12-31")

# 计算技术指标
features = TechnicalFeatures()
data_with_features = features.calculate(data)

# 初始化策略
strategy = MACrossStrategy(
    name="双均线策略",
    config={"short_window": 5, "long_window": 20}
)

# 回测
backtest_config = strategy_config.get("backtest", {})
engine = BacktestEngine(backtest_config)
result = engine.run(strategy, data_with_features)

# 查看结果
print(f"总收益率: {result.total_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
```

## 📖 核心模块

### DataHandler (数据处理)

抽象基类定义数据获取接口，`AkshareDataLoader` 提供具体实现：

```python
from src import AkshareDataLoader

loader = AkshareDataLoader(config)

# 获取日线数据
daily_data = loader.fetch_daily_data("000001", "2023-01-01", "2024-12-31")

# 获取股票列表
hs300_stocks = loader.get_stock_list(index_code="000300")

# 获取基本面数据
fundamental = loader.fetch_fundamental_data("000001")
```

### FeatureEngine (因子计算)

支持技术指标和 Alpha 因子计算：

```python
from src import TechnicalFeatures

engine = TechnicalFeatures()

# 计算所有默认因子
data_with_features = engine.calculate(ohlcv_data)

# 添加自定义因子
engine.add_feature("my_factor", lambda df: df["close"] / df["open"])
```

**内置技术指标**:
- 均线: SMA, EMA
- 动量: RSI, MACD, KDJ, ROC
- 波动: ATR, 布林带, 波动率
- 其他: Williams %R, 动量

### Strategy (策略)

抽象策略接口，支持多种策略实现：

```python
from src import MACrossStrategy, RSIStrategy, CompositeStrategy

# 均线交叉策略
ma_strategy = MACrossStrategy(config={"short_window": 5, "long_window": 20})

# RSI策略
rsi_strategy = RSIStrategy(config={"oversold": 30, "overbought": 70})

# 组合策略
composite = CompositeStrategy()
composite.add_strategy(ma_strategy, weight=0.6)
composite.add_strategy(rsi_strategy, weight=0.4)
```

### BacktestEngine (回测)

高性能向量化回测引擎：

```python
from src import BacktestEngine

engine = BacktestEngine({
    "initial_capital": 1000000,
    "commission": 0.0003,
    "slippage": 0.001
})

# 执行回测
result = engine.run(strategy, price_data)

# 参数优化
optimization_result = engine.run_optimization(
    MACrossStrategy,
    price_data,
    param_grid={
        "short_window": [3, 5, 10],
        "long_window": [15, 20, 30]
    }
)

# 策略对比
comparison = engine.compare_strategies([ma_strategy, rsi_strategy], price_data)
```

## ⚙️ 配置说明

### strategy_config.yaml

```yaml
strategy:
  name: "双均线策略"
  
parameters:
  short_window: 5
  long_window: 20
  stop_loss: 0.08
  take_profit: 0.20

backtest:
  initial_capital: 1000000
  commission: 0.0003
```

### data_config.yaml

```yaml
data_source:
  provider: "akshare"
  retry_times: 3

universe:
  index_codes:
    - "000300"

storage:
  file_format: "parquet"
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行带覆盖率报告
pytest --cov=src --cov-report=html

# 运行特定测试
pytest tests/test_strategy.py -v
```

## 📝 开发规范

- **类型提示**: 所有函数必须包含类型注解
- **向量化**: 优先使用 Pandas/NumPy 向量化操作
- **文档**: 使用 NumPy 风格 docstring
- **代码风格**: 遵循 Black 和 isort 格式化规范

```bash
# 代码格式化
black src tests
isort src tests

# 类型检查
mypy src
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。

