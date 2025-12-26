"""
LLM 客户端模块

该模块提供与 OpenAI 兼容 API（如 DeepSeek、GPT-4o）的交互功能，
主要用于金融新闻情绪分析。

Performance Notes
-----------------
- 使用 tenacity 库实现指数退避重试
- 实现熔断器（Circuit Breaker）机制：连续失败超过阈值时抛出异常停止交易
- 支持从环境变量或配置文件读取 API 密钥
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    """
    score: float
    confidence: float
    summary: str = ""


class LLMCircuitBreakerError(RuntimeError):
    """
    LLM 熔断器触发异常
    
    当 LLM API 连续失败次数超过阈值时抛出此异常，
    用于停止交易以避免在 API 不可用时继续操作。
    """
    pass


class LLMClient:
    """
    LLM 客户端类
    
    封装 OpenAI 兼容 API 的调用，用于金融新闻情绪分析。
    支持 DeepSeek、OpenAI 等兼容接口。
    
    包含熔断器（Circuit Breaker）机制：当连续 API 调用失败次数超过阈值时，
    抛出 LLMCircuitBreakerError 异常，停止交易操作。
    
    Parameters
    ----------
    api_key : Optional[str]
        API 密钥。如果为 None，尝试从环境变量读取：
        - DEEPSEEK_API_KEY（优先）
        - OPENAI_API_KEY
    base_url : Optional[str]
        API Base URL。如果为 None，尝试从环境变量 OPENAI_BASE_URL 读取，
        默认使用 DeepSeek 的端点
    model : str
        模型名称，默认 "deepseek-chat"
    timeout : int
        请求超时时间（秒），默认 30
    max_retries : int
        最大重试次数，默认 3
    max_consecutive_failures : int
        熔断器阈值：连续失败次数超过此值时触发熔断，默认 5
    
    Attributes
    ----------
    client : OpenAI
        OpenAI 客户端实例
    model : str
        使用的模型名称
    
    Examples
    --------
    >>> client = LLMClient()
    >>> result = client.get_sentiment_score("公司业绩大幅增长，净利润同比增长50%")
    >>> print(result.score, result.confidence)  # 0.8, 0.9
    
    >>> # 使用自定义配置
    >>> client = LLMClient(
    ...     api_key="sk-xxx",
    ...     base_url="https://api.openai.com/v1",
    ...     model="gpt-4o",
    ...     max_consecutive_failures=5
    ... )
    
    Notes
    -----
    - 熔断器触发时抛出 LLMCircuitBreakerError，不会静默失败
    - 使用指数退避重试策略应对临时网络问题
    - 日志记录所有 API 调用和错误
    """
    
    # 默认 DeepSeek API 端点
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    
    # 系统提示词
    SYSTEM_PROMPT = "You are a financial risk control expert."
    
    # 用户提示词模板（包含置信度和摘要）
    USER_PROMPT_TEMPLATE = (
        "Analyze the following news for stock {symbol}: {news_content}. "
        "Output a valid JSON object with the following fields: "
        "'score' (float, -1.0 to 1.0, where -1.0 = Bankruptcy/Investigation/Delisting, "
        "0.0 = Neutral, 1.0 = Major Breakthrough/Huge Profit), "
        "'confidence' (float, 0.0 to 1.0, how confident you are in this sentiment assessment), "
        "'summary' (string, one-sentence reason for the score). "
        "Output only JSON."
    )
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        timeout: int = 30,
        max_retries: int = 3,
        max_consecutive_failures: int = 5
    ) -> None:
        """
        初始化 LLM 客户端
        
        Parameters
        ----------
        api_key : Optional[str]
            API 密钥
        base_url : Optional[str]
            API Base URL
        model : str
            模型名称
        timeout : int
            请求超时时间
        max_retries : int
            最大重试次数
        max_consecutive_failures : int
            熔断器阈值，默认 5
        """
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[Any] = None
        
        # 熔断器状态
        self._consecutive_failures: int = 0
        self._max_consecutive_failures: int = max_consecutive_failures
        
        # 检查依赖
        if not OPENAI_AVAILABLE:
            logger.warning(
                "openai 库未安装，LLM 功能不可用。"
                "安装命令: pip install openai"
            )
            return
        
        # 获取 API Key
        resolved_api_key = api_key or self._get_api_key_from_env()
        if not resolved_api_key:
            logger.warning(
                "未配置 API Key，LLM 功能不可用。"
                "请设置环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
            )
            return
        
        # 获取 Base URL
        resolved_base_url = base_url or self._get_base_url_from_env()
        
        # 初始化客户端
        try:
            self._client = OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                timeout=timeout
            )
            logger.info(
                f"LLM 客户端初始化成功: model={model}, base_url={resolved_base_url}, "
                f"max_consecutive_failures={max_consecutive_failures}"
            )
        except Exception as e:
            logger.error(f"LLM 客户端初始化失败: {e}")
            self._client = None
    
    @staticmethod
    def _get_api_key_from_env() -> Optional[str]:
        """
        从环境变量获取 API Key
        
        Returns
        -------
        Optional[str]
            API Key，如果未找到则返回 None
        
        Notes
        -----
        按以下顺序查找：
        1. DEEPSEEK_API_KEY
        2. OPENAI_API_KEY
        """
        return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    def _get_base_url_from_env(self) -> str:
        """
        从环境变量获取 Base URL
        
        Returns
        -------
        str
            Base URL，默认使用 DeepSeek 端点
        """
        return os.environ.get("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)
    
    @property
    def is_available(self) -> bool:
        """
        检查 LLM 客户端是否可用
        
        Returns
        -------
        bool
            客户端是否可用
        """
        return self._client is not None
    
    @property
    def consecutive_failures(self) -> int:
        """
        获取当前连续失败次数
        
        Returns
        -------
        int
            连续失败次数
        """
        return self._consecutive_failures
    
    @property
    def is_circuit_breaker_open(self) -> bool:
        """
        检查熔断器是否触发
        
        Returns
        -------
        bool
            熔断器是否已触发
        """
        return self._consecutive_failures >= self._max_consecutive_failures
    
    def reset_circuit_breaker(self) -> None:
        """
        手动重置熔断器状态
        
        用于在问题解决后恢复 API 调用功能。
        """
        self._consecutive_failures = 0
        logger.info("LLM 熔断器已手动重置")
    
    def _check_circuit_breaker(self) -> None:
        """
        检查熔断器状态
        
        Raises
        ------
        LLMCircuitBreakerError
            当连续失败次数超过阈值时抛出
        """
        if self._consecutive_failures >= self._max_consecutive_failures:
            error_msg = (
                f"LLM Circuit Breaker Triggered: {self._consecutive_failures} "
                f"consecutive failures (threshold: {self._max_consecutive_failures}). "
                f"Trading halted. Manual intervention required."
            )
            logger.critical(error_msg)
            raise LLMCircuitBreakerError(error_msg)
    
    def get_sentiment_score(
        self,
        news_content: str,
        symbol: str = "UNKNOWN"
    ) -> SentimentResult:
        """
        获取新闻情绪分数
        
        分析给定的新闻内容，返回情绪分数和置信度。
        
        Parameters
        ----------
        news_content : str
            新闻内容文本
        symbol : str, optional
            股票代码，用于日志和提示词，默认 "UNKNOWN"
        
        Returns
        -------
        SentimentResult
            包含 score, confidence, summary 的结果对象
            - score: 情绪分数，范围 [-1.0, 1.0]
            - confidence: 置信度，范围 [0.0, 1.0]
            - summary: 分析摘要
        
        Raises
        ------
        LLMCircuitBreakerError
            当熔断器触发时抛出（连续失败次数超过阈值）
        
        Notes
        -----
        - 如果客户端不可用，返回中性结果 (score=0.0, confidence=0.0)
        - 如果新闻内容为空，返回中性结果
        - 熔断器触发时不会静默失败，而是抛出异常
        
        Examples
        --------
        >>> result = client.get_sentiment_score(
        ...     "公司因财务造假被证监会立案调查",
        ...     symbol="000001"
        ... )
        >>> print(result.score, result.confidence)  # -0.9, 0.95
        """
        # 检查熔断器
        self._check_circuit_breaker()
        
        # 检查客户端可用性
        if not self.is_available:
            logger.debug(f"LLM 客户端不可用，返回中性分数: symbol={symbol}")
            return SentimentResult(score=0.0, confidence=0.0, summary="LLM client unavailable")
        
        # 检查输入
        if not news_content or not news_content.strip():
            logger.debug(f"新闻内容为空，返回中性分数: symbol={symbol}")
            return SentimentResult(score=0.0, confidence=0.0, summary="No news content")
        
        # 构建提示词
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            symbol=symbol,
            news_content=news_content[:2000]  # 限制长度避免超出 token 限制
        )
        
        # 调用 API
        try:
            result = self._call_api_with_retry(user_prompt, symbol)
            # 成功：重置熔断器
            self._consecutive_failures = 0
            return result
        except LLMCircuitBreakerError:
            # 熔断器异常直接抛出
            raise
        except Exception as e:
            # 失败：递增熔断器计数
            self._consecutive_failures += 1
            logger.warning(
                f"获取情绪分数失败 (symbol={symbol}): {e}. "
                f"连续失败次数: {self._consecutive_failures}/{self._max_consecutive_failures}"
            )
            
            # 检查是否需要触发熔断
            self._check_circuit_breaker()
            
            # 未触发熔断时返回中性结果
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                summary=f"API call failed: {str(e)[:100]}"
            )
    
    def _call_api_with_retry(self, user_prompt: str, symbol: str) -> SentimentResult:
        """
        带重试的 API 调用
        
        Parameters
        ----------
        user_prompt : str
            用户提示词
        symbol : str
            股票代码（用于日志）
        
        Returns
        -------
        SentimentResult
            情绪分析结果
        """
        if TENACITY_AVAILABLE:
            return self._call_api_with_tenacity(user_prompt, symbol)
        else:
            return self._call_api_simple_retry(user_prompt, symbol)
    
    def _call_api_with_tenacity(self, user_prompt: str, symbol: str) -> SentimentResult:
        """
        使用 tenacity 的重试逻辑
        
        Parameters
        ----------
        user_prompt : str
            用户提示词
        symbol : str
            股票代码
        
        Returns
        -------
        SentimentResult
            情绪分析结果
        """
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((Exception,)),
            reraise=True
        )
        def _do_call() -> SentimentResult:
            return self._single_api_call(user_prompt, symbol)
        
        return _do_call()
    
    def _call_api_simple_retry(self, user_prompt: str, symbol: str) -> SentimentResult:
        """
        简单重试逻辑（无 tenacity 时使用）
        
        Parameters
        ----------
        user_prompt : str
            用户提示词
        symbol : str
            股票代码
        
        Returns
        -------
        SentimentResult
            情绪分析结果
        """
        import time
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self._single_api_call(user_prompt, symbol)
            except Exception as e:
                last_exception = e
                wait_time = min(2 ** attempt, 10)
                logger.debug(
                    f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}, "
                    f"等待 {wait_time} 秒后重试"
                )
                time.sleep(wait_time)
        
        raise last_exception if last_exception else RuntimeError("API 调用失败")
    
    def _single_api_call(self, user_prompt: str, symbol: str) -> SentimentResult:
        """
        单次 API 调用
        
        Parameters
        ----------
        user_prompt : str
            用户提示词
        symbol : str
            股票代码
        
        Returns
        -------
        SentimentResult
            情绪分析结果
        
        Raises
        ------
        Exception
            API 调用或解析失败时抛出
        """
        logger.debug(f"调用 LLM API: symbol={symbol}, model={self.model}")
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # 低温度以获得更一致的输出
            max_tokens=150,   # 增加 token 限制以容纳 summary
            response_format={"type": "json_object"}  # 强制 JSON 输出
        )
        
        # 解析响应
        content = response.choices[0].message.content
        result = self._parse_score_from_response(content)
        
        logger.debug(
            f"LLM 返回情绪分数: symbol={symbol}, "
            f"score={result.score}, confidence={result.confidence}"
        )
        return result
    
    @staticmethod
    def _parse_score_from_response(content: str) -> SentimentResult:
        """
        从 LLM 响应中解析分数和置信度
        
        Parameters
        ----------
        content : str
            LLM 响应内容
        
        Returns
        -------
        SentimentResult
            解析出的结果，包含 score, confidence, summary
        
        Raises
        ------
        ValueError
            无法解析响应时抛出
        
        Notes
        -----
        - score 范围限制在 [-1.0, 1.0]
        - confidence 范围限制在 [0.0, 1.0]，缺失时默认 0.5
        - summary 缺失时默认为空字符串
        """
        if not content:
            raise ValueError("LLM 返回空响应")
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 JSON 响应: {content}") from e
        
        if "score" not in data:
            raise ValueError(f"响应中缺少 'score' 字段: {content}")
        
        # 解析 score
        score = float(data["score"])
        score = max(-1.0, min(1.0, score))  # 限制范围
        
        # 解析 confidence（可选，默认 0.5）
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # 限制范围
        
        # 解析 summary（可选，默认空字符串）
        summary = str(data.get("summary", ""))
        
        return SentimentResult(score=score, confidence=confidence, summary=summary)
    
    def analyze_batch(
        self,
        news_dict: Dict[str, str],
        parallel: bool = False
    ) -> Dict[str, SentimentResult]:
        """
        批量分析多条新闻
        
        Parameters
        ----------
        news_dict : Dict[str, str]
            股票代码到新闻内容的映射
        parallel : bool, optional
            是否并行处理（暂不支持），默认 False
        
        Returns
        -------
        Dict[str, SentimentResult]
            股票代码到情绪分析结果的映射
        
        Raises
        ------
        LLMCircuitBreakerError
            当熔断器触发时抛出
        
        Examples
        --------
        >>> news = {
        ...     "000001": "公司业绩大幅增长",
        ...     "000002": "被监管部门处罚"
        ... }
        >>> results = client.analyze_batch(news)
        >>> print(results["000001"].score)
        0.7
        """
        results: Dict[str, SentimentResult] = {}
        
        for symbol, news_content in news_dict.items():
            result = self.get_sentiment_score(news_content, symbol)
            results[symbol] = result
        
        return results


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
    
    Examples
    --------
    >>> config = load_config("config/strategy_config.yaml")
    >>> client = create_llm_client_from_config(config)
    """
    llm_config = config.get("llm", {})
    
    return LLMClient(
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        model=llm_config.get("model", "deepseek-chat"),
        timeout=llm_config.get("request_timeout", 30),
        max_retries=llm_config.get("max_retries", 3),
        max_consecutive_failures=llm_config.get("max_consecutive_failures", 5)
    )
