"""
LLM 客户端模块 (优化版)

该模块提供与 OpenAI 兼容 API（如 Qwen/通义千问、DeepSeek、GPT-4o）的交互功能，
主要用于金融新闻情绪分析。

Performance Optimizations
-------------------------
- 异步并发请求：使用 asyncio + httpx 实现高效批量处理
- 智能缓存：支持 TTL 过期、LRU 淘汰、批量持久化
- Token 优化：使用 tiktoken 精确计算和截断
- 熔断器：连续失败超过阈值时抛出异常停止交易
- 模型降级：主模型失败时自动切换到备用模型
- 性能监控：API 延迟、Token 使用量、成功率统计

Notes
-----
- 使用 Pydantic 进行响应验证
- 支持从环境变量或配置文件读取 API 密钥
- 中文金融新闻专业化提示词工程
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# 可选依赖导入
try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

class SentimentCategory(str, Enum):
    """情绪类别枚举"""
    VERY_NEGATIVE = "very_negative"  # 极度负面：破产、立案调查、退市风险
    NEGATIVE = "negative"            # 负面：业绩下滑、处罚、诉讼
    NEUTRAL = "neutral"              # 中性：无重大影响
    POSITIVE = "positive"            # 正面：业绩增长、获奖、合作
    VERY_POSITIVE = "very_positive"  # 极度正面：重大突破、巨额利润


if PYDANTIC_AVAILABLE:
    class SentimentResponse(BaseModel):
        """
        LLM 情绪分析响应模型 (Pydantic 验证)
        
        Attributes
        ----------
        score : float
            情绪分数，范围 [-1.0, 1.0]
        confidence : float
            置信度，范围 [0.0, 1.0]
        category : str
            情绪类别
        summary : str
            分析摘要（中文）
        key_factors : list[str]
            关键影响因素
        """
        score: float = Field(ge=-1.0, le=1.0, description="情绪分数")
        confidence: float = Field(ge=0.0, le=1.0, default=0.5, description="置信度")
        category: str = Field(default="neutral", description="情绪类别")
        summary: str = Field(default="", description="分析摘要")
        key_factors: List[str] = Field(default_factory=list, description="关键因素")
        
        @field_validator('score', mode='before')
        @classmethod
        def clamp_score(cls, v: Any) -> float:
            """确保分数在有效范围内"""
            try:
                value = float(v)
                return max(-1.0, min(1.0, value))
            except (TypeError, ValueError):
                return 0.0
        
        @field_validator('confidence', mode='before')
        @classmethod
        def clamp_confidence(cls, v: Any) -> float:
            """确保置信度在有效范围内"""
            try:
                value = float(v)
                return max(0.0, min(1.0, value))
            except (TypeError, ValueError):
                return 0.5
else:
    # Pydantic 不可用时的回退类
    class SentimentResponse:  # type: ignore
        """情绪分析响应（无 Pydantic 版本）"""
        def __init__(
            self,
            score: float = 0.0,
            confidence: float = 0.5,
            category: str = "neutral",
            summary: str = "",
            key_factors: Optional[List[str]] = None
        ):
            self.score = max(-1.0, min(1.0, float(score)))
            self.confidence = max(0.0, min(1.0, float(confidence)))
            self.category = category
            self.summary = summary
            self.key_factors = key_factors or []


@dataclass
class SentimentResult:
    """
    情绪分析结果数据类
    
    Attributes
    ----------
    score : float
        情绪分数，范围 [-1.0, 1.0]
    confidence : float
        置信度，范围 [0.0, 1.0]
    summary : str
        分析摘要
    category : str
        情绪类别
    key_factors : list[str]
        关键影响因素
    latency_ms : float
        API 调用延迟（毫秒）
    tokens_used : int
        使用的 Token 数量
    model : str
        使用的模型名称
    cached : bool
        是否来自缓存
    """
    score: float
    confidence: float
    summary: str = ""
    category: str = "neutral"
    key_factors: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: int = 0
    model: str = ""
    cached: bool = False


@dataclass
class LLMMetrics:
    """
    LLM 性能指标
    
    Attributes
    ----------
    total_calls : int
        总调用次数
    success_count : int
        成功次数
    failure_count : int
        失败次数
    cache_hits : int
        缓存命中次数
    total_tokens : int
        总 Token 使用量
    total_latency_ms : float
        总延迟（毫秒）
    model_calls : dict
        各模型调用统计
    """
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    cache_hits: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    model_calls: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls
    
    @property
    def avg_latency_ms(self) -> float:
        """平均延迟（毫秒）"""
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count
    
    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_calls == 0:
            return 0.0
        return self.cache_hits / self.total_calls
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_calls": self.total_calls,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": f"{self.success_rate:.2%}",
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{self.cache_hit_rate:.2%}",
            "total_tokens": self.total_tokens,
            "avg_latency_ms": f"{self.avg_latency_ms:.1f}",
            "model_calls": self.model_calls,
        }


# ==================== 异常定义 ====================

class LLMCircuitBreakerError(RuntimeError):
    """
    LLM 熔断器触发异常
    
    当 LLM API 连续失败次数超过阈值时抛出此异常，
    用于停止交易以避免在 API 不可用时继续操作。
    """
    pass


class LLMRateLimitError(RuntimeError):
    """LLM API 速率限制异常"""
    pass


class LLMTokenLimitError(RuntimeError):
    """LLM Token 限制超出异常"""
    pass


# ==================== 缓存管理 ====================

class LRUCache:
    """
    LRU 缓存，支持 TTL 过期
    
    Parameters
    ----------
    max_size : int
        最大缓存条目数
    ttl_seconds : int
        缓存过期时间（秒）
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 86400 * 7) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        # 延迟初始化异步锁，避免在导入时调用 get_event_loop() 导致的问题
        self._lock: Optional[asyncio.Lock] = None
    
    def _get_lock(self) -> Optional[asyncio.Lock]:
        """延迟获取异步锁（避免在非异步上下文中初始化）"""
        if self._lock is None:
            try:
                # 仅在事件循环运行时创建锁
                loop = asyncio.get_running_loop()
                self._lock = asyncio.Lock()
            except RuntimeError:
                # 没有运行中的事件循环，返回 None
                pass
        return self._lock
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        
        # 检查过期
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None
        
        # 移到最后（最近使用）
        self._cache.move_to_end(key)
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        if key in self._cache:
            self._cache.move_to_end(key)
        
        self._cache[key] = (value, time.time())
        
        # LRU 淘汰
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在（考虑过期）"""
        return self.get(key) is not None
    
    def __len__(self) -> int:
        """返回缓存大小"""
        return len(self._cache)
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（不包含过期条目）"""
        current_time = time.time()
        return {
            key: value
            for key, (value, timestamp) in self._cache.items()
            if current_time - timestamp <= self.ttl_seconds
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载"""
        current_time = time.time()
        for key, value in data.items():
            if isinstance(value, dict) and "_timestamp" in value:
                # 新格式：包含时间戳
                self._cache[key] = (value["data"], value["_timestamp"])
            else:
                # 旧格式：使用当前时间
                self._cache[key] = (value, current_time)


class SentimentCache:
    """
    情绪分数持久化缓存
    
    支持内存 LRU 缓存 + 文件持久化 + TTL 过期
    
    Parameters
    ----------
    cache_path : str
        缓存文件路径
    max_memory_size : int
        内存缓存最大条目数
    ttl_days : int
        缓存过期天数
    auto_save_interval : int
        自动保存间隔（条目数）
    """
    
    def __init__(
        self,
        cache_path: str = "data/processed/sentiment_cache.json",
        max_memory_size: int = 10000,
        ttl_days: int = 7,
        auto_save_interval: int = 10
    ) -> None:
        self.cache_path = Path(cache_path)
        self.auto_save_interval = auto_save_interval
        self._memory_cache = LRUCache(max_memory_size, ttl_days * 86400)
        self._pending_writes = 0
        self._load_cache()
    
    def _load_cache(self) -> None:
        """从文件加载缓存"""
        if not self.cache_path.exists():
            return
        
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            # 兼容旧格式
            converted = {}
            for key, value in raw_data.items():
                if isinstance(value, (int, float)):
                    # 旧版格式：纯分数
                    converted[key] = {"score": float(value), "confidence": 0.5}
                elif isinstance(value, dict):
                    converted[key] = value
            
            self._memory_cache.from_dict(converted)
            logger.info(f"加载情绪缓存: {len(self._memory_cache)} 条记录")
            
        except Exception as e:
            logger.warning(f"加载情绪缓存失败: {e}")
    
    def save(self, force: bool = False) -> None:
        """保存缓存到文件"""
        if not force and self._pending_writes < self.auto_save_interval:
            return
        
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 导出时添加时间戳
            export_data = {}
            current_time = time.time()
            for key, (value, timestamp) in self._memory_cache._cache.items():
                export_data[key] = {
                    **value,
                    "_timestamp": timestamp
                }
            
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self._pending_writes = 0
            logger.debug(f"保存情绪缓存: {len(export_data)} 条记录")
            
        except Exception as e:
            logger.warning(f"保存情绪缓存失败: {e}")
    
    def get(self, stock_code: str, date: str) -> Optional[Dict[str, float]]:
        """获取缓存的情绪分数"""
        key = self._make_key(stock_code, date)
        return self._memory_cache.get(key)
    
    def set(self, stock_code: str, date: str, result: SentimentResult) -> None:
        """设置情绪分数缓存"""
        key = self._make_key(stock_code, date)
        self._memory_cache.set(key, {
            "score": result.score,
            "confidence": result.confidence,
            "summary": result.summary,
            "category": result.category,
        })
        self._pending_writes += 1
        self.save()  # 自动保存（如果达到阈值）
    
    @staticmethod
    def _make_key(stock_code: str, date: str) -> str:
        """生成缓存键"""
        date_str = str(date)[:10]
        return f"{stock_code}_{date_str}"
    
    def clear(self) -> None:
        """清空缓存"""
        self._memory_cache.clear()
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("情绪缓存已清空")
    
    def __len__(self) -> int:
        return len(self._memory_cache)


# ==================== Token 工具 ====================

class TokenCounter:
    """
    Token 计数器
    
    使用 tiktoken 精确计算文本的 token 数量
    """
    
    # 默认编码器（GPT-3.5/4 系列）
    DEFAULT_ENCODING = "cl100k_base"
    
    # 模型到编码的映射
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "deepseek-chat": "cl100k_base",
        "qwen": "cl100k_base",
    }
    
    def __init__(self, model: str = "gpt-4") -> None:
        self.model = model
        self._encoder = None
        
        if TIKTOKEN_AVAILABLE:
            encoding_name = self.MODEL_ENCODINGS.get(model, self.DEFAULT_ENCODING)
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                self._encoder = tiktoken.get_encoding(self.DEFAULT_ENCODING)
    
    def count(self, text: str) -> int:
        """计算文本的 token 数量"""
        if self._encoder is None:
            # 回退：粗略估算（中文约 0.5 token/字，英文约 0.25 token/词）
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            return int(chinese_chars * 0.7 + len(text.split()) * 0.25)
        
        return len(self._encoder.encode(text))
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数量"""
        if self._encoder is None:
            # 回退：粗略截断
            return text[:int(max_tokens * 1.5)]
        
        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        return self._encoder.decode(tokens[:max_tokens])


# ==================== 提示词工程 ====================

class PromptTemplate:
    """
    金融情绪分析提示词模板
    
    支持中文金融新闻专业化分析，使用思维链 (CoT) 引导
    """
    
    # 系统提示词：金融风控专家角色
    SYSTEM_PROMPT_CN = """你是一位资深的A股量化投资分析师和金融风控专家。

你的专业领域：
1. 深度理解A股市场规则、监管政策和市场惯例
2. 准确识别财报造假、内幕交易、市场操纵等风险信号
3. 评估公司基本面变化对股价的潜在影响
4. 区分短期噪音和实质性利好/利空

分析原则：
- 立案调查、财务造假、退市警示 = 极度负面（-1.0）
- 业绩大幅下滑、重大诉讼、监管处罚 = 负面（-0.5 ~ -0.8）
- 日常经营新闻、人事变动 = 中性（-0.2 ~ 0.2）
- 业绩超预期、重大合同、战略合作 = 正面（0.3 ~ 0.7）
- 技术突破、行业龙头地位确立、巨额利润 = 极度正面（0.8 ~ 1.0）

输出要求：严格返回 JSON 格式，不要添加任何额外解释。"""

    # 用户提示词模板（CoT 引导）
    USER_PROMPT_TEMPLATE_CN = """请分析以下股票 {symbol} 的相关新闻，评估其对股价的潜在影响。

【新闻内容】
{news_content}

【分析步骤】
1. 首先识别新闻的核心事件类型（财报、监管、合作、诉讼等）
2. 评估事件的严重程度和持续性
3. 考虑对公司基本面的实质影响
4. 给出量化的情绪分数

请返回以下 JSON 格式：
{{
    "score": <float, -1.0 到 1.0, 越负面分数越低>,
    "confidence": <float, 0.0 到 1.0, 分析结果的确定程度>,
    "category": <string, 分类: very_negative/negative/neutral/positive/very_positive>,
    "summary": <string, 一句话总结分析结论（中文）>,
    "key_factors": [<string, 影响判断的关键因素列表>]
}}"""

    # 简化版提示词（用于省 Token）
    USER_PROMPT_TEMPLATE_SIMPLE = """分析股票 {symbol} 新闻的情绪倾向：

{news_content}

返回 JSON: {{"score": <-1到1>, "confidence": <0到1>, "summary": "<一句话分析>"}}"""

    @classmethod
    def get_system_prompt(cls, language: str = "cn") -> str:
        """获取系统提示词"""
        return cls.SYSTEM_PROMPT_CN
    
    @classmethod
    def get_user_prompt(
        cls,
        news_content: str,
        symbol: str,
        detailed: bool = True
    ) -> str:
        """
        生成用户提示词
        
        Parameters
        ----------
        news_content : str
            新闻内容
        symbol : str
            股票代码
        detailed : bool
            是否使用详细模板
        
        Returns
        -------
        str
            格式化的提示词
        """
        template = (
            cls.USER_PROMPT_TEMPLATE_CN if detailed 
            else cls.USER_PROMPT_TEMPLATE_SIMPLE
        )
        return template.format(symbol=symbol, news_content=news_content)


# ==================== LLM 客户端 ====================

class LLMClient:
    """
    LLM 客户端类（优化版）
    
    封装 OpenAI 兼容 API 的调用，用于金融新闻情绪分析。
    支持 Qwen/通义千问、DeepSeek、OpenAI 等兼容接口。
    
    核心特性：
    - 异步并发请求支持
    - 智能缓存（LRU + TTL + 批量持久化）
    - 熔断器机制
    - 模型降级（fallback）
    - 性能监控
    - Token 优化
    
    Parameters
    ----------
    api_key : Optional[str]
        API 密钥
    base_url : Optional[str]
        API Base URL
    model : str
        主模型名称
    fallback_model : Optional[str]
        备用模型名称
    timeout : int
        请求超时时间（秒）
    max_retries : int
        最大重试次数
    max_consecutive_failures : int
        熔断器阈值
    max_concurrent : int
        最大并发请求数
    cache_path : str
        缓存文件路径
    cache_ttl_days : int
        缓存过期天数
    detailed_prompts : bool
        是否使用详细提示词（更多 Token 但更准确）
    
    Examples
    --------
    >>> client = LLMClient()
    >>> result = client.get_sentiment_score("公司业绩大幅增长", "000001")
    >>> print(result.score, result.confidence)
    
    >>> # 异步批量分析
    >>> import asyncio
    >>> results = asyncio.run(client.analyze_batch_async({
    ...     "000001": "业绩增长",
    ...     "000002": "被立案调查"
    ... }))
    """
    
    # 默认 API 端点
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        fallback_model: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        max_consecutive_failures: int = 5,
        max_concurrent: int = 5,
        cache_path: str = "data/processed/sentiment_cache.json",
        cache_ttl_days: int = 7,
        detailed_prompts: bool = True
    ) -> None:
        """初始化 LLM 客户端"""
        self.model = model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.detailed_prompts = detailed_prompts
        
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None
        
        # 熔断器状态
        self._consecutive_failures: int = 0
        self._max_consecutive_failures: int = max_consecutive_failures
        
        # 性能指标
        self._metrics = LLMMetrics()
        
        # 缓存
        self._cache = SentimentCache(
            cache_path=cache_path,
            ttl_days=cache_ttl_days
        )
        
        # Token 计数器
        self._token_counter = TokenCounter(model)
        
        # 并发信号量
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        # 初始化客户端
        self._init_clients(api_key, base_url)
    
    def _init_clients(
        self,
        api_key: Optional[str],
        base_url: Optional[str]
    ) -> None:
        """初始化同步和异步客户端"""
        if not OPENAI_AVAILABLE:
            logger.warning(
                "openai 库未安装，LLM 功能不可用。"
                "安装命令: pip install openai"
            )
            return
        
        resolved_api_key = api_key or self._get_api_key_from_env()
        if not resolved_api_key:
            logger.warning(
                "未配置 API Key，LLM 功能不可用。"
                "请设置环境变量 QWEN_API_KEY、DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
            )
            return
        
        resolved_base_url = base_url or self._get_base_url_from_env()
        
        try:
            self._client = OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                timeout=self.timeout
            )
            
            self._async_client = AsyncOpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                timeout=self.timeout
            )
            
            logger.info(
                f"LLM 客户端初始化成功: model={self.model}, "
                f"fallback={self.fallback_model}, "
                f"base_url={resolved_base_url}"
            )
        except Exception as e:
            logger.error(f"LLM 客户端初始化失败: {e}")
    
    @staticmethod
    def _get_api_key_from_env() -> Optional[str]:
        """从环境变量获取 API Key"""
        return (
            os.environ.get("QWEN_API_KEY") or
            os.environ.get("DEEPSEEK_API_KEY") or
            os.environ.get("OPENAI_API_KEY")
        )
    
    def _get_base_url_from_env(self) -> str:
        """从环境变量获取 Base URL"""
        return os.environ.get("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)
    
    @property
    def is_available(self) -> bool:
        """检查 LLM 客户端是否可用"""
        return self._client is not None
    
    @property
    def consecutive_failures(self) -> int:
        """获取当前连续失败次数"""
        return self._consecutive_failures
    
    @property
    def is_circuit_breaker_open(self) -> bool:
        """检查熔断器是否触发"""
        return self._consecutive_failures >= self._max_consecutive_failures
    
    @property
    def metrics(self) -> LLMMetrics:
        """获取性能指标"""
        return self._metrics
    
    def reset_circuit_breaker(self) -> None:
        """手动重置熔断器状态"""
        self._consecutive_failures = 0
        logger.info("LLM 熔断器已手动重置")
    
    def reset_metrics(self) -> None:
        """重置性能指标"""
        self._metrics = LLMMetrics()
        logger.info("LLM 性能指标已重置")
    
    def _check_circuit_breaker(self) -> None:
        """检查熔断器状态"""
        if self._consecutive_failures >= self._max_consecutive_failures:
            error_msg = (
                f"LLM Circuit Breaker Triggered: {self._consecutive_failures} "
                f"consecutive failures (threshold: {self._max_consecutive_failures}). "
                f"Trading halted. Manual intervention required."
            )
            logger.critical(error_msg)
            raise LLMCircuitBreakerError(error_msg)
    
    def _prepare_news_content(self, news_content: str, max_tokens: int = 1500) -> str:
        """
        预处理新闻内容
        
        使用 Token 计数器精确截断，而非简单字符截断
        """
        if not news_content or not news_content.strip():
            return ""
        
        current_tokens = self._token_counter.count(news_content)
        
        if current_tokens <= max_tokens:
            return news_content.strip()
        
        # 智能截断
        truncated = self._token_counter.truncate(news_content, max_tokens)
        return truncated.strip() + "..."
    
    def get_sentiment_score(
        self,
        news_content: str,
        symbol: str = "UNKNOWN",
        use_cache: bool = True,
        date: Optional[str] = None
    ) -> SentimentResult:
        """
        获取新闻情绪分数（同步版）
        
        Parameters
        ----------
        news_content : str
            新闻内容文本
        symbol : str
            股票代码
        use_cache : bool
            是否使用缓存
        date : Optional[str]
            日期（用于缓存键）
        
        Returns
        -------
        SentimentResult
            情绪分析结果
        
        Raises
        ------
        LLMCircuitBreakerError
            当熔断器触发时抛出
        """
        self._check_circuit_breaker()
        self._metrics.total_calls += 1
        
        # 日期默认使用今天
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        
        # 检查缓存
        if use_cache:
            cached = self._cache.get(symbol, date_str)
            if cached is not None:
                self._metrics.cache_hits += 1
                return SentimentResult(
                    score=cached.get("score", 0.0),
                    confidence=cached.get("confidence", 0.5),
                    summary=cached.get("summary", ""),
                    category=cached.get("category", "neutral"),
                    cached=True
                )
        
        # 检查客户端可用性
        if not self.is_available:
            logger.debug(f"LLM 客户端不可用，返回中性分数: symbol={symbol}")
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                summary="LLM client unavailable"
            )
        
        # 检查输入
        processed_content = self._prepare_news_content(news_content)
        if not processed_content:
            logger.debug(f"新闻内容为空，返回中性分数: symbol={symbol}")
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                summary="No news content"
            )
        
        # 调用 API
        start_time = time.time()
        try:
            result = self._call_api_with_retry(processed_content, symbol)
            
            # 成功：更新指标
            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms
            self._consecutive_failures = 0
            self._metrics.success_count += 1
            self._metrics.total_latency_ms += latency_ms
            self._metrics.total_tokens += result.tokens_used
            
            # 更新模型调用统计
            model_key = result.model or self.model
            self._metrics.model_calls[model_key] = (
                self._metrics.model_calls.get(model_key, 0) + 1
            )
            
            # 更新缓存
            if use_cache:
                self._cache.set(symbol, date_str, result)
            
            return result
            
        except LLMCircuitBreakerError:
            raise
        except Exception as e:
            self._consecutive_failures += 1
            self._metrics.failure_count += 1
            
            logger.warning(
                f"获取情绪分数失败 (symbol={symbol}): {e}. "
                f"连续失败次数: {self._consecutive_failures}/{self._max_consecutive_failures}"
            )
            
            self._check_circuit_breaker()
            
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                summary=f"API call failed: {str(e)[:100]}"
            )
    
    def _call_api_with_retry(
        self,
        news_content: str,
        symbol: str,
        use_fallback: bool = True
    ) -> SentimentResult:
        """带重试的 API 调用"""
        models_to_try = [self.model]
        if use_fallback and self.fallback_model:
            models_to_try.append(self.fallback_model)
        
        last_exception = None
        
        for model in models_to_try:
            try:
                if TENACITY_AVAILABLE:
                    return self._call_api_with_tenacity(news_content, symbol, model)
                else:
                    return self._call_api_simple_retry(news_content, symbol, model)
            except Exception as e:
                last_exception = e
                if model != models_to_try[-1]:
                    logger.warning(
                        f"模型 {model} 调用失败，尝试降级到 {models_to_try[-1]}: {e}"
                    )
        
        raise last_exception if last_exception else RuntimeError("API 调用失败")
    
    def _call_api_with_tenacity(
        self,
        news_content: str,
        symbol: str,
        model: str
    ) -> SentimentResult:
        """使用 tenacity 的重试逻辑"""
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((Exception,)),
            reraise=True
        )
        def _do_call() -> SentimentResult:
            return self._single_api_call(news_content, symbol, model)
        
        return _do_call()
    
    def _call_api_simple_retry(
        self,
        news_content: str,
        symbol: str,
        model: str
    ) -> SentimentResult:
        """简单重试逻辑（无 tenacity 时使用）"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return self._single_api_call(news_content, symbol, model)
            except Exception as e:
                last_exception = e
                wait_time = min(2 ** attempt, 10)
                logger.debug(
                    f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}, "
                    f"等待 {wait_time} 秒后重试"
                )
                time.sleep(wait_time)
        
        raise last_exception if last_exception else RuntimeError("API 调用失败")
    
    def _single_api_call(
        self,
        news_content: str,
        symbol: str,
        model: str
    ) -> SentimentResult:
        """单次 API 调用"""
        logger.debug(f"调用 LLM API: symbol={symbol}, model={model}")
        
        system_prompt = PromptTemplate.get_system_prompt()
        user_prompt = PromptTemplate.get_user_prompt(
            news_content=news_content,
            symbol=symbol,
            detailed=self.detailed_prompts
        )
        
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        tokens_used = getattr(response.usage, 'total_tokens', 0) if response.usage else 0
        
        result = self._parse_response(content)
        result.model = model
        result.tokens_used = tokens_used
        
        logger.debug(
            f"LLM 返回: symbol={symbol}, score={result.score:.2f}, "
            f"confidence={result.confidence:.2f}, tokens={tokens_used}"
        )
        
        return result
    
    def _parse_response(self, content: str) -> SentimentResult:
        """解析 LLM 响应"""
        if not content:
            raise ValueError("LLM 返回空响应")
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 JSON 响应: {content}") from e
        
        if "score" not in data:
            raise ValueError(f"响应中缺少 'score' 字段: {content}")
        
        # 使用 Pydantic 验证或手动解析
        if PYDANTIC_AVAILABLE:
            response = SentimentResponse(**data)
            return SentimentResult(
                score=response.score,
                confidence=response.confidence,
                summary=response.summary,
                category=response.category,
                key_factors=response.key_factors
            )
        else:
            score = max(-1.0, min(1.0, float(data["score"])))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            
            return SentimentResult(
                score=score,
                confidence=confidence,
                summary=str(data.get("summary", "")),
                category=str(data.get("category", "neutral")),
                key_factors=data.get("key_factors", [])
            )
    
    # ==================== 异步方法 ====================
    
    async def get_sentiment_score_async(
        self,
        news_content: str,
        symbol: str = "UNKNOWN",
        use_cache: bool = True,
        date: Optional[str] = None
    ) -> SentimentResult:
        """
        获取新闻情绪分数（异步版）
        
        Parameters
        ----------
        news_content : str
            新闻内容
        symbol : str
            股票代码
        use_cache : bool
            是否使用缓存
        date : Optional[str]
            日期
        
        Returns
        -------
        SentimentResult
            情绪分析结果
        """
        self._check_circuit_breaker()
        self._metrics.total_calls += 1
        
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        
        # 检查缓存
        if use_cache:
            cached = self._cache.get(symbol, date_str)
            if cached is not None:
                self._metrics.cache_hits += 1
                return SentimentResult(
                    score=cached.get("score", 0.0),
                    confidence=cached.get("confidence", 0.5),
                    summary=cached.get("summary", ""),
                    category=cached.get("category", "neutral"),
                    cached=True
                )
        
        if self._async_client is None:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                summary="Async client unavailable"
            )
        
        processed_content = self._prepare_news_content(news_content)
        if not processed_content:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                summary="No news content"
            )
        
        # 使用信号量限制并发
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with self._semaphore:
            start_time = time.time()
            try:
                result = await self._single_api_call_async(
                    processed_content, symbol, self.model
                )
                
                latency_ms = (time.time() - start_time) * 1000
                result.latency_ms = latency_ms
                self._consecutive_failures = 0
                self._metrics.success_count += 1
                self._metrics.total_latency_ms += latency_ms
                self._metrics.total_tokens += result.tokens_used
                
                if use_cache:
                    self._cache.set(symbol, date_str, result)
                
                return result
                
            except Exception as e:
                self._consecutive_failures += 1
                self._metrics.failure_count += 1
                
                logger.warning(f"异步 API 调用失败 (symbol={symbol}): {e}")
                
                return SentimentResult(
                    score=0.0,
                    confidence=0.0,
                    summary=f"Async API failed: {str(e)[:100]}"
                )
    
    async def _single_api_call_async(
        self,
        news_content: str,
        symbol: str,
        model: str
    ) -> SentimentResult:
        """单次异步 API 调用"""
        system_prompt = PromptTemplate.get_system_prompt()
        user_prompt = PromptTemplate.get_user_prompt(
            news_content=news_content,
            symbol=symbol,
            detailed=self.detailed_prompts
        )
        
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        tokens_used = getattr(response.usage, 'total_tokens', 0) if response.usage else 0
        
        result = self._parse_response(content)
        result.model = model
        result.tokens_used = tokens_used
        
        return result
    
    async def analyze_batch_async(
        self,
        news_dict: Dict[str, str],
        date: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, SentimentResult]:
        """
        异步批量分析多条新闻
        
        Parameters
        ----------
        news_dict : Dict[str, str]
            股票代码到新闻内容的映射
        date : Optional[str]
            日期
        use_cache : bool
            是否使用缓存
        
        Returns
        -------
        Dict[str, SentimentResult]
            股票代码到情绪分析结果的映射
        """
        if not news_dict:
            return {}
        
        tasks = []
        symbols = []
        
        for symbol, news_content in news_dict.items():
            task = self.get_sentiment_score_async(
                news_content=news_content,
                symbol=symbol,
                use_cache=use_cache,
                date=date
            )
            tasks.append(task)
            symbols.append(symbol)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"批量分析异常 (symbol={symbol}): {result}")
                output[symbol] = SentimentResult(
                    score=0.0,
                    confidence=0.0,
                    summary=f"Error: {str(result)[:50]}"
                )
            else:
                output[symbol] = result
        
        # 批量保存缓存
        self._cache.save(force=True)
        
        return output
    
    def analyze_batch(
        self,
        news_dict: Dict[str, str],
        date: Optional[str] = None,
        use_cache: bool = True,
        parallel: bool = True
    ) -> Dict[str, SentimentResult]:
        """
        批量分析多条新闻（同步包装）
        
        Parameters
        ----------
        news_dict : Dict[str, str]
            股票代码到新闻内容的映射
        date : Optional[str]
            日期
        use_cache : bool
            是否使用缓存
        parallel : bool
            是否并行处理
        
        Returns
        -------
        Dict[str, SentimentResult]
            股票代码到情绪分析结果的映射
        """
        if not parallel:
            # 串行处理
            results = {}
            for symbol, news_content in news_dict.items():
                results[symbol] = self.get_sentiment_score(
                    news_content=news_content,
                    symbol=symbol,
                    use_cache=use_cache,
                    date=date
                )
            return results
        
        # 并行处理
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已运行的事件循环中
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.analyze_batch_async(news_dict, date, use_cache)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.analyze_batch_async(news_dict, date, use_cache)
                )
        except RuntimeError:
            # 没有事件循环
            return asyncio.run(
                self.analyze_batch_async(news_dict, date, use_cache)
            )
    
    def save_cache(self) -> None:
        """强制保存缓存"""
        self._cache.save(force=True)
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "size": len(self._cache),
            "path": str(self._cache.cache_path),
        }


def create_llm_client_from_config(config: Dict[str, Any]) -> LLMClient:
    """
    从配置字典创建 LLM 客户端
    
    Parameters
    ----------
    config : Dict[str, Any]
        配置字典，应包含 'llm' 部分
    
    Returns
    -------
    LLMClient
        配置好的 LLM 客户端实例
    """
    llm_config = config.get("llm", {})
    
    return LLMClient(
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        model=llm_config.get("model", "deepseek-chat"),
        fallback_model=llm_config.get("fallback_model"),
        timeout=llm_config.get("request_timeout", 30),
        max_retries=llm_config.get("max_retries", 3),
        max_consecutive_failures=llm_config.get("max_consecutive_failures", 5),
        max_concurrent=llm_config.get("max_concurrent", 5),
        cache_path=llm_config.get("cache_path", "data/processed/sentiment_cache.json"),
        cache_ttl_days=llm_config.get("cache_ttl_days", 7),
        detailed_prompts=llm_config.get("detailed_prompts", True)
    )
