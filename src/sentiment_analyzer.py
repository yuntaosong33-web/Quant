"""
金融情感分析模块

使用 ProsusAI/finbert 预训练模型进行金融新闻情感分析。
支持 GPU 加速推理、异步新闻获取和批量处理。

Performance Notes
-----------------
- 使用 torch.utils.data.DataLoader 进行高效批处理
- GPU 推理与网络 I/O 分离，使用 asyncio 异步获取新闻
- 支持多线程并行分析以提高吞吐量
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
import threading
import hashlib
import json
import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class SentimentResult:
    """
    情感分析结果数据类
    
    Attributes
    ----------
    text : str
        分析的文本内容
    label : str
        情感标签 ('positive', 'negative', 'neutral')
    score : float
        情感分数 [-1.0, 1.0]，正值表示积极，负值表示消极
    confidence : float
        模型置信度 [0.0, 1.0]
    probabilities : Dict[str, float]
        各标签的概率分布
    """
    text: str
    label: str
    score: float
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class DailySentiment:
    """
    每日股票情感汇总数据类
    
    Attributes
    ----------
    stock_code : str
        股票代码
    date : str
        日期 (YYYY-MM-DD)
    sentiment_score : float
        当日加权情感分数 [-1.0, 1.0]
    sentiment_ma3 : float
        3日情感分数移动平均
    news_count : int
        当日新闻数量
    avg_confidence : float
        平均置信度
    """
    stock_code: str
    date: str
    sentiment_score: float
    sentiment_ma3: float = 0.0
    news_count: int = 0
    avg_confidence: float = 0.0


# ============================================================================
# 文本数据集
# ============================================================================

class TextDataset(Dataset):
    """
    文本数据集，用于 DataLoader 批量处理
    
    Parameters
    ----------
    texts : List[str]
        文本列表
    max_length : int
        最大序列长度
    
    Examples
    --------
    >>> dataset = TextDataset(["新闻1", "新闻2"], max_length=512)
    >>> loader = DataLoader(dataset, batch_size=8)
    """
    
    def __init__(self, texts: List[str], max_length: int = 512) -> None:
        """
        初始化数据集
        
        Parameters
        ----------
        texts : List[str]
            文本列表
        max_length : int
            最大序列长度，默认 512
        """
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> str:
        """
        获取单个样本
        
        Parameters
        ----------
        idx : int
            样本索引
        
        Returns
        -------
        str
            文本内容
        """
        text = self.texts[idx]
        # 截断超长文本
        if len(text) > self.max_length * 4:  # 估算 token 长度
            text = text[:self.max_length * 4]
        return text


# ============================================================================
# 情感分析器核心类
# ============================================================================

class SentimentAnalyzer:
    """
    金融情感分析器
    
    使用 ProsusAI/finbert 预训练模型进行金融文本情感分析。
    支持 GPU 加速推理和批量处理。
    
    Parameters
    ----------
    model_name : str
        Hugging Face 模型名称，默认 'ProsusAI/finbert'
    device : Optional[str]
        计算设备 ('cuda', 'cpu', 'auto')，默认 'auto' 自动检测
    batch_size : int
        推理批次大小，默认 16
    max_length : int
        最大序列长度，默认 512
    cache_dir : Optional[str]
        模型缓存目录
    
    Attributes
    ----------
    model : transformers.PreTrainedModel
        FinBERT 模型
    tokenizer : transformers.PreTrainedTokenizer
        分词器
    device : torch.device
        计算设备
    
    Examples
    --------
    >>> analyzer = SentimentAnalyzer(device='cuda')
    >>> 
    >>> # 单条分析
    >>> result = analyzer.analyze("公司业绩大幅增长")
    >>> print(f"情感: {result.label}, 分数: {result.score:.2f}")
    >>> 
    >>> # 批量分析
    >>> texts = ["利润增长50%", "股价暴跌", "业务稳健发展"]
    >>> results = analyzer.batch_score(texts)
    >>> for r in results:
    ...     print(f"{r.label}: {r.score:.2f}")
    
    Notes
    -----
    - FinBERT 专为金融文本设计，对财经新闻有较好的识别效果
    - 模型首次加载需要从 Hugging Face 下载（约 500MB）
    - GPU 推理速度比 CPU 快 10-20 倍
    """
    
    # 标签映射
    LABEL_MAPPING = {
        'positive': 1.0,
        'negative': -1.0,
        'neutral': 0.0,
        0: 'positive',
        1: 'negative', 
        2: 'neutral',
    }
    
    # FinBERT 特定标签映射
    FINBERT_LABEL_MAPPING = {
        'LABEL_0': 'positive',
        'LABEL_1': 'negative',
        'LABEL_2': 'neutral',
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
    }
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        初始化情感分析器
        
        Parameters
        ----------
        model_name : str
            模型名称
        device : Optional[str]
            计算设备
        batch_size : int
            批次大小
        max_length : int
            最大序列长度
        cache_dir : Optional[str]
            缓存目录
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # 确定设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"初始化 SentimentAnalyzer: device={self.device}, batch_size={batch_size}")
        
        # 加载模型和分词器
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # 线程锁（用于多线程安全）
        self._lock = threading.Lock()
    
    def _load_model(self) -> None:
        """
        加载预训练模型和分词器
        
        Raises
        ------
        ImportError
            当 transformers 未安装时
        RuntimeError
            当模型加载失败时
        """
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                logging as transformers_logging
            )
            # 降低 transformers 日志级别
            transformers_logging.set_verbosity_error()
        except ImportError:
            raise ImportError(
                "请安装 transformers 库: pip install transformers"
            )
        
        logger.info(f"正在加载模型: {self.model_name}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # 移动到指定设备
            self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info(f"模型加载成功: {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载模型 {self.model_name}: {e}")
    
    def analyze(self, text: str) -> SentimentResult:
        """
        分析单条文本的情感
        
        Parameters
        ----------
        text : str
            待分析的文本
        
        Returns
        -------
        SentimentResult
            情感分析结果
        
        Examples
        --------
        >>> result = analyzer.analyze("公司股价创新高")
        >>> print(result.label, result.score)
        positive 0.85
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                label="neutral",
                score=0.0,
                confidence=0.0,
                probabilities={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            )
        
        results = self.batch_score([text])
        return results[0] if results else SentimentResult(
            text=text,
            label="neutral",
            score=0.0,
            confidence=0.0
        )
    
    def batch_score(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[SentimentResult]:
        """
        批量分析文本情感
        
        使用 DataLoader 进行高效批处理，在 GPU 上进行推理。
        
        Parameters
        ----------
        texts : List[str]
            待分析的文本列表
        show_progress : bool
            是否显示进度条，默认 False
        
        Returns
        -------
        List[SentimentResult]
            情感分析结果列表，与输入顺序一致
        
        Notes
        -----
        - 使用 torch.no_grad() 禁用梯度计算，减少内存占用
        - 空文本会被自动处理为 neutral
        - 支持任意长度的文本列表
        
        Examples
        --------
        >>> texts = ["利好消息", "股价下跌", "市场平稳"]
        >>> results = analyzer.batch_score(texts)
        >>> for text, result in zip(texts, results):
        ...     print(f"{text}: {result.label} ({result.score:.2f})")
        利好消息: positive (0.92)
        股价下跌: negative (-0.85)
        市场平稳: neutral (0.05)
        """
        if not texts:
            return []
        
        # 过滤空文本并记录索引
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text.strip())
        
        # 如果没有有效文本，返回全 neutral
        if not valid_texts:
            return [
                SentimentResult(
                    text=t,
                    label="neutral",
                    score=0.0,
                    confidence=0.0,
                    probabilities={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
                )
                for t in texts
            ]
        
        # 创建数据集和 DataLoader
        dataset = TextDataset(valid_texts, max_length=self.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # 避免多进程问题
            pin_memory=self.device.type == 'cuda'
        )
        
        # 批量推理
        all_results: List[SentimentResult] = []
        
        with self._lock:  # 确保线程安全
            with torch.no_grad():
                for batch_texts in dataloader:
                    # 分词
                    encoding = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    # 移动到设备
                    input_ids = encoding["input_ids"].to(self.device)
                    attention_mask = encoding["attention_mask"].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # 计算概率
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    
                    # 获取预测标签和置信度
                    predictions = torch.argmax(probs, dim=-1)
                    confidences = torch.max(probs, dim=-1).values
                    
                    # 转换为 CPU numpy
                    probs_np = probs.cpu().numpy()
                    predictions_np = predictions.cpu().numpy()
                    confidences_np = confidences.cpu().numpy()
                    
                    # 构建结果
                    for i, text in enumerate(batch_texts):
                        # 获取标签
                        pred_id = predictions_np[i]
                        label = self._get_label_name(pred_id)
                        
                        # 计算情感分数
                        prob_dict = self._get_prob_dict(probs_np[i])
                        score = self._calculate_score(prob_dict)
                        
                        all_results.append(SentimentResult(
                            text=text,
                            label=label,
                            score=score,
                            confidence=float(confidences_np[i]),
                            probabilities=prob_dict
                        ))
        
        # 将结果填回原始位置
        final_results: List[SentimentResult] = []
        valid_idx = 0
        for i, text in enumerate(texts):
            if i in valid_indices:
                final_results.append(all_results[valid_idx])
                valid_idx += 1
            else:
                # 空文本返回 neutral
                final_results.append(SentimentResult(
                    text=text,
                    label="neutral",
                    score=0.0,
                    confidence=0.0,
                    probabilities={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
                ))
        
        return final_results
    
    def _get_label_name(self, label_id: int) -> str:
        """
        将标签 ID 转换为标签名称
        
        Parameters
        ----------
        label_id : int
            标签 ID
        
        Returns
        -------
        str
            标签名称
        """
        # 尝试从模型配置获取标签
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
            label = self.model.config.id2label.get(label_id, f"LABEL_{label_id}")
            # 标准化标签
            return self.FINBERT_LABEL_MAPPING.get(label, label.lower())
        
        # 默认映射
        return self.LABEL_MAPPING.get(label_id, 'neutral')
    
    def _get_prob_dict(self, probs: np.ndarray) -> Dict[str, float]:
        """
        将概率数组转换为字典
        
        Parameters
        ----------
        probs : np.ndarray
            概率数组
        
        Returns
        -------
        Dict[str, float]
            标签到概率的映射
        """
        # FinBERT 通常是 3 类：positive, negative, neutral
        if len(probs) == 3:
            # 尝试从模型配置获取标签顺序
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                id2label = self.model.config.id2label
                return {
                    self.FINBERT_LABEL_MAPPING.get(id2label[i], id2label[i].lower()): float(probs[i])
                    for i in range(len(probs))
                }
            # 默认顺序
            return {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2])
            }
        
        # 其他情况
        return {f"label_{i}": float(p) for i, p in enumerate(probs)}
    
    def _calculate_score(self, prob_dict: Dict[str, float]) -> float:
        """
        根据概率分布计算情感分数
        
        分数范围 [-1.0, 1.0]，其中：
        - 1.0 表示完全积极
        - -1.0 表示完全消极
        - 0.0 表示中性
        
        Parameters
        ----------
        prob_dict : Dict[str, float]
            标签概率字典
        
        Returns
        -------
        float
            情感分数
        """
        pos_prob = prob_dict.get('positive', 0.0)
        neg_prob = prob_dict.get('negative', 0.0)
        
        # 分数 = 积极概率 - 消极概率
        score = pos_prob - neg_prob
        
        return float(np.clip(score, -1.0, 1.0))
    
    @property
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None and self.tokenizer is not None


# ============================================================================
# 异步新闻获取器
# ============================================================================

class AsyncNewsFetcher:
    """
    异步新闻获取器
    
    使用 asyncio 并发获取多只股票的新闻数据。
    将网络 I/O 与 GPU 推理分离，提高整体效率。
    
    Parameters
    ----------
    max_concurrent : int
        最大并发请求数，默认 10
    timeout : int
        单个请求超时时间（秒），默认 30
    cache_ttl : int
        缓存有效期（秒），默认 3600（1小时）
    
    Attributes
    ----------
    cache : Dict[str, Tuple[str, float]]
        新闻缓存，key 为 "stock_date"，value 为 (新闻内容, 缓存时间戳)
    
    Examples
    --------
    >>> fetcher = AsyncNewsFetcher(max_concurrent=5)
    >>> 
    >>> # 异步获取多只股票新闻
    >>> import asyncio
    >>> news = asyncio.run(fetcher.fetch_batch(
    ...     ["000001", "000002", "600519"],
    ...     "2024-01-15"
    ... ))
    >>> print(news)
    {'000001': '公司业绩...', '000002': '项目进展...', '600519': ''}
    
    Notes
    -----
    - 使用 asyncio.Semaphore 限制并发数
    - 网络错误不会中断其他请求
    - 内置 LRU 缓存避免重复请求
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: int = 30,
        cache_ttl: int = 3600
    ) -> None:
        """
        初始化异步新闻获取器
        
        Parameters
        ----------
        max_concurrent : int
            最大并发数
        timeout : int
            超时时间（秒）
        cache_ttl : int
            缓存有效期（秒）
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        
        # 新闻缓存 {cache_key: (content, timestamp)}
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._cache_lock = threading.Lock()
        
        # 信号量将在异步上下文中创建
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        logger.info(f"AsyncNewsFetcher 初始化: max_concurrent={max_concurrent}")
    
    def _get_cache_key(self, stock_code: str, date: str) -> str:
        """生成缓存键"""
        return f"{stock_code}_{date[:10]}"
    
    def _get_from_cache(self, stock_code: str, date: str) -> Optional[str]:
        """
        从缓存获取新闻
        
        Parameters
        ----------
        stock_code : str
            股票代码
        date : str
            日期
        
        Returns
        -------
        Optional[str]
            缓存的新闻内容，过期或不存在返回 None
        """
        cache_key = self._get_cache_key(stock_code, date)
        
        with self._cache_lock:
            if cache_key in self._cache:
                content, timestamp = self._cache[cache_key]
                # 检查是否过期
                if time.time() - timestamp < self.cache_ttl:
                    return content
                else:
                    # 删除过期缓存
                    del self._cache[cache_key]
        
        return None
    
    def _set_cache(self, stock_code: str, date: str, content: str) -> None:
        """
        设置缓存
        
        Parameters
        ----------
        stock_code : str
            股票代码
        date : str
            日期
        content : str
            新闻内容
        """
        cache_key = self._get_cache_key(stock_code, date)
        
        with self._cache_lock:
            self._cache[cache_key] = (content, time.time())
    
    async def fetch_single(
        self,
        stock_code: str,
        date: str,
        use_cache: bool = True
    ) -> Tuple[str, str]:
        """
        异步获取单只股票新闻
        
        Parameters
        ----------
        stock_code : str
            股票代码
        date : str
            日期
        use_cache : bool
            是否使用缓存
        
        Returns
        -------
        Tuple[str, str]
            (股票代码, 新闻内容)
        """
        # 检查缓存
        if use_cache:
            cached = self._get_from_cache(stock_code, date)
            if cached is not None:
                logger.debug(f"缓存命中: {stock_code}")
                return (stock_code, cached)
        
        # 使用信号量限制并发
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with self._semaphore:
            try:
                # 在线程池中执行同步的 AkShare 调用
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(
                    None,
                    self._fetch_news_sync,
                    stock_code,
                    date
                )
                
                # 缓存结果
                if use_cache:
                    self._set_cache(stock_code, date, content)
                
                return (stock_code, content)
                
            except asyncio.TimeoutError:
                logger.warning(f"获取新闻超时: {stock_code}")
                return (stock_code, "")
            except Exception as e:
                logger.warning(f"获取新闻失败 {stock_code}: {e}")
                return (stock_code, "")
    
    def _fetch_news_sync(self, stock_code: str, date: str) -> str:
        """
        同步获取新闻（在线程池中执行）
        
        Parameters
        ----------
        stock_code : str
            股票代码
        date : str
            日期
        
        Returns
        -------
        str
            新闻内容
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装")
            return ""
        
        try:
            # 标准化股票代码
            clean_code = stock_code.replace(".", "").replace("SZ", "").replace("SH", "")
            if len(clean_code) > 6:
                clean_code = clean_code[:6]
            
            # 获取新闻
            news_df = ak.stock_news_em(symbol=clean_code)
            
            if news_df is None or news_df.empty:
                return ""
            
            # 过滤日期范围（前后3天）
            target_date = pd.to_datetime(date)
            
            if "发布时间" in news_df.columns:
                news_df["发布时间"] = pd.to_datetime(news_df["发布时间"], errors="coerce")
                date_mask = (
                    (news_df["发布时间"] >= target_date - pd.Timedelta(days=3)) &
                    (news_df["发布时间"] <= target_date + pd.Timedelta(days=1))
                )
                filtered = news_df[date_mask]
            else:
                filtered = news_df.head(5)
            
            if filtered.empty:
                filtered = news_df.head(3)
            
            # 提取标题和内容
            texts = []
            title_col = "新闻标题" if "新闻标题" in filtered.columns else None
            content_col = "新闻内容" if "新闻内容" in filtered.columns else None
            
            for _, row in filtered.head(5).iterrows():
                parts = []
                if title_col and pd.notna(row.get(title_col)):
                    parts.append(str(row[title_col]))
                if content_col and pd.notna(row.get(content_col)):
                    parts.append(str(row[content_col])[:200])
                if parts:
                    texts.append("; ".join(parts))
            
            combined = " | ".join(texts)
            
            # 截断
            if len(combined) > 1500:
                combined = combined[:1500] + "..."
            
            logger.debug(f"获取新闻成功: {stock_code}, {len(texts)} 条")
            return combined
            
        except Exception as e:
            logger.warning(f"获取新闻异常 {stock_code}: {e}")
            return ""
    
    async def fetch_batch(
        self,
        stock_list: List[str],
        date: str,
        use_cache: bool = True
    ) -> Dict[str, str]:
        """
        批量异步获取多只股票新闻
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        date : str
            日期
        use_cache : bool
            是否使用缓存
        
        Returns
        -------
        Dict[str, str]
            股票代码到新闻内容的映射
        
        Examples
        --------
        >>> import asyncio
        >>> fetcher = AsyncNewsFetcher()
        >>> news = asyncio.run(fetcher.fetch_batch(
        ...     ["000001", "000002"],
        ...     "2024-01-15"
        ... ))
        """
        if not stock_list:
            return {}
        
        # 创建所有任务
        tasks = [
            self.fetch_single(code, date, use_cache)
            for code in stock_list
        ]
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 构建结果字典
        news_dict = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"获取新闻异常: {result}")
                continue
            if isinstance(result, tuple) and len(result) == 2:
                stock_code, content = result
                news_dict[stock_code] = content
        
        logger.info(
            f"批量获取新闻完成: 请求 {len(stock_list)}, "
            f"成功 {len(news_dict)}"
        )
        
        return news_dict
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("新闻缓存已清空")


# ============================================================================
# 情感分析管道
# ============================================================================

class SentimentPipeline:
    """
    情感分析管道
    
    整合异步新闻获取、GPU 推理和结果聚合的完整管道。
    支持按股票代码聚合每日情感分数，并计算移动平均。
    
    Parameters
    ----------
    model_name : str
        模型名称，默认 'ProsusAI/finbert'
    device : str
        计算设备，默认 'auto'
    batch_size : int
        推理批次大小，默认 16
    max_concurrent_fetch : int
        最大并发新闻获取数，默认 10
    cache_dir : Optional[str]
        缓存目录
    
    Attributes
    ----------
    analyzer : SentimentAnalyzer
        情感分析器
    fetcher : AsyncNewsFetcher
        异步新闻获取器
    
    Examples
    --------
    >>> pipeline = SentimentPipeline(device='cuda')
    >>> 
    >>> # 分析单日情感
    >>> result = pipeline.analyze_daily(
    ...     stock_list=["000001", "000002", "600519"],
    ...     date="2024-01-15"
    ... )
    >>> print(result)
    
    >>> # 分析多日并计算移动平均
    >>> result_with_ma = pipeline.analyze_with_moving_average(
    ...     stock_list=["000001", "000002"],
    ...     start_date="2024-01-10",
    ...     end_date="2024-01-15",
    ...     ma_window=3
    ... )
    
    Notes
    -----
    - 网络 I/O（新闻获取）和 GPU 推理在不同线程中执行
    - 使用 asyncio 实现异步新闻获取，不阻塞主线程
    - 支持历史数据的移动平均计算
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "auto",
        batch_size: int = 16,
        max_concurrent_fetch: int = 10,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        初始化情感分析管道
        
        Parameters
        ----------
        model_name : str
            模型名称
        device : str
            计算设备
        batch_size : int
            批次大小
        max_concurrent_fetch : int
            最大并发新闻获取数
        cache_dir : Optional[str]
            缓存目录
        """
        # 初始化组件
        self.analyzer = SentimentAnalyzer(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache_dir=cache_dir
        )
        
        self.fetcher = AsyncNewsFetcher(
            max_concurrent=max_concurrent_fetch
        )
        
        # 历史分数缓存（用于移动平均计算）
        self._history_cache: Dict[str, List[Tuple[str, float]]] = {}
        
        logger.info(
            f"SentimentPipeline 初始化完成: "
            f"model={model_name}, device={device}"
        )
    
    def analyze_daily(
        self,
        stock_list: List[str],
        date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        分析单日股票情感
        
        异步获取新闻后，使用 GPU 批量推理计算情感分数。
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        date : str
            日期 (YYYY-MM-DD)
        use_cache : bool
            是否使用缓存
        
        Returns
        -------
        pd.DataFrame
            情感分析结果，包含以下列：
            - stock_code: 股票代码
            - date: 日期
            - sentiment_score: 情感分数 [-1.0, 1.0]
            - confidence: 置信度
            - news_count: 新闻数量
            - label: 情感标签
        
        Examples
        --------
        >>> result = pipeline.analyze_daily(
        ...     ["000001", "000002"],
        ...     "2024-01-15"
        ... )
        >>> print(result)
           stock_code        date  sentiment_score  confidence  news_count   label
        0      000001  2024-01-15             0.45        0.82           3  positive
        1      000002  2024-01-15            -0.12        0.65           2  neutral
        """
        if not stock_list:
            return pd.DataFrame(columns=[
                "stock_code", "date", "sentiment_score", 
                "confidence", "news_count", "label"
            ])
        
        # Step 1: 异步获取新闻（网络 I/O）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            news_dict = loop.run_until_complete(
                self.fetcher.fetch_batch(stock_list, date, use_cache)
            )
        finally:
            loop.close()
        
        logger.info(f"新闻获取完成: {len(news_dict)}/{len(stock_list)} 只股票有新闻")
        
        # Step 2: GPU 批量推理
        # 准备有新闻的股票列表
        stocks_with_news = []
        texts_to_analyze = []
        
        for stock_code in stock_list:
            news = news_dict.get(stock_code, "")
            if news:
                stocks_with_news.append(stock_code)
                texts_to_analyze.append(news)
        
        # 执行推理
        if texts_to_analyze:
            results = self.analyzer.batch_score(texts_to_analyze)
        else:
            results = []
        
        # Step 3: 构建结果 DataFrame
        records = []
        result_idx = 0
        
        for stock_code in stock_list:
            if stock_code in stocks_with_news:
                result = results[result_idx]
                result_idx += 1
                
                records.append({
                    "stock_code": stock_code,
                    "date": date,
                    "sentiment_score": result.score,
                    "confidence": result.confidence,
                    "news_count": 1,  # 简化处理
                    "label": result.label
                })
            else:
                # 无新闻的股票
                records.append({
                    "stock_code": stock_code,
                    "date": date,
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "news_count": 0,
                    "label": "neutral"
                })
        
        df = pd.DataFrame(records)
        
        logger.info(
            f"情感分析完成: date={date}, "
            f"stocks={len(stock_list)}, "
            f"with_news={len(stocks_with_news)}"
        )
        
        return df
    
    def aggregate_daily_sentiment(
        self,
        sentiment_results: List[SentimentResult],
        stock_code: str,
        date: str
    ) -> DailySentiment:
        """
        聚合单只股票当日的多条新闻情感
        
        使用置信度加权平均计算综合情感分数。
        
        Parameters
        ----------
        sentiment_results : List[SentimentResult]
            单只股票多条新闻的情感分析结果
        stock_code : str
            股票代码
        date : str
            日期
        
        Returns
        -------
        DailySentiment
            聚合后的每日情感数据
        
        Notes
        -----
        加权公式: score = sum(result.score * result.confidence) / sum(confidence)
        """
        if not sentiment_results:
            return DailySentiment(
                stock_code=stock_code,
                date=date,
                sentiment_score=0.0,
                news_count=0,
                avg_confidence=0.0
            )
        
        # 置信度加权平均
        total_weight = 0.0
        weighted_score = 0.0
        
        for result in sentiment_results:
            weight = result.confidence
            weighted_score += result.score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0
        
        avg_confidence = np.mean([r.confidence for r in sentiment_results])
        
        return DailySentiment(
            stock_code=stock_code,
            date=date,
            sentiment_score=final_score,
            news_count=len(sentiment_results),
            avg_confidence=avg_confidence
        )
    
    def calculate_moving_average(
        self,
        sentiment_df: pd.DataFrame,
        window: int = 3
    ) -> pd.DataFrame:
        """
        计算情感分数的移动平均
        
        按股票代码分组计算滚动移动平均。
        
        Parameters
        ----------
        sentiment_df : pd.DataFrame
            情感分析结果 DataFrame，需包含 stock_code, date, sentiment_score 列
        window : int
            移动平均窗口大小，默认 3（天）
        
        Returns
        -------
        pd.DataFrame
            添加了 sentiment_ma{window} 列的 DataFrame
        
        Examples
        --------
        >>> df = pipeline.analyze_daily(["000001"], "2024-01-15")
        >>> df_with_ma = pipeline.calculate_moving_average(df, window=3)
        >>> print(df_with_ma[['stock_code', 'sentiment_score', 'sentiment_ma3']])
        """
        if sentiment_df.empty:
            sentiment_df[f"sentiment_ma{window}"] = np.nan
            return sentiment_df
        
        df = sentiment_df.copy()
        
        # 确保按日期排序
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["stock_code", "date"])
        
        # 按股票分组计算移动平均
        df[f"sentiment_ma{window}"] = df.groupby("stock_code")["sentiment_score"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        return df
    
    def analyze_with_moving_average(
        self,
        stock_list: List[str],
        start_date: str,
        end_date: str,
        ma_window: int = 3,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        分析多日情感并计算移动平均
        
        遍历日期范围，逐日分析情感，最后计算移动平均。
        
        Parameters
        ----------
        stock_list : List[str]
            股票代码列表
        start_date : str
            开始日期 (YYYY-MM-DD)
        end_date : str
            结束日期 (YYYY-MM-DD)
        ma_window : int
            移动平均窗口，默认 3
        use_cache : bool
            是否使用缓存
        
        Returns
        -------
        pd.DataFrame
            包含每日情感分数和移动平均的 DataFrame
        
        Examples
        --------
        >>> result = pipeline.analyze_with_moving_average(
        ...     stock_list=["000001", "000002"],
        ...     start_date="2024-01-10",
        ...     end_date="2024-01-15",
        ...     ma_window=3
        ... )
        >>> print(result[['stock_code', 'date', 'sentiment_score', 'sentiment_ma3']])
        """
        # 生成日期范围
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='B')  # 工作日
        
        all_results = []
        
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            logger.info(f"分析日期: {date_str}")
            
            daily_df = self.analyze_daily(
                stock_list=stock_list,
                date=date_str,
                use_cache=use_cache
            )
            all_results.append(daily_df)
        
        if not all_results:
            return pd.DataFrame()
        
        # 合并所有日期的结果
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # 计算移动平均
        result_df = self.calculate_moving_average(combined_df, window=ma_window)
        
        logger.info(
            f"多日分析完成: "
            f"日期范围 {start_date} ~ {end_date}, "
            f"共 {len(result_df)} 条记录"
        )
        
        return result_df
    
    def get_latest_sentiment(
        self,
        stock_code: str,
        date: str,
        lookback_days: int = 3
    ) -> Optional[DailySentiment]:
        """
        获取股票最新情感分数（含移动平均）
        
        Parameters
        ----------
        stock_code : str
            股票代码
        date : str
            基准日期
        lookback_days : int
            回溯天数（用于移动平均）
        
        Returns
        -------
        Optional[DailySentiment]
            最新情感数据，分析失败返回 None
        """
        # 生成日期范围
        end = pd.to_datetime(date)
        start = end - pd.Timedelta(days=lookback_days * 2)  # 多取一些以确保有足够工作日
        
        df = self.analyze_with_moving_average(
            stock_list=[stock_code],
            start_date=start.strftime("%Y-%m-%d"),
            end_date=date,
            ma_window=lookback_days
        )
        
        if df.empty:
            return None
        
        # 获取最新一条
        latest = df[df["date"] == df["date"].max()].iloc[0]
        
        return DailySentiment(
            stock_code=stock_code,
            date=date,
            sentiment_score=latest["sentiment_score"],
            sentiment_ma3=latest.get(f"sentiment_ma{lookback_days}", 0.0),
            news_count=latest["news_count"],
            avg_confidence=latest["confidence"]
        )


# ============================================================================
# 便捷函数
# ============================================================================

def create_sentiment_pipeline(
    device: str = "auto",
    batch_size: int = 16
) -> SentimentPipeline:
    """
    创建情感分析管道的工厂函数
    
    Parameters
    ----------
    device : str
        计算设备，'auto', 'cuda', 或 'cpu'
    batch_size : int
        推理批次大小
    
    Returns
    -------
    SentimentPipeline
        配置好的情感分析管道
    
    Examples
    --------
    >>> pipeline = create_sentiment_pipeline(device='cuda')
    >>> result = pipeline.analyze_daily(["000001"], "2024-01-15")
    """
    return SentimentPipeline(
        model_name="ProsusAI/finbert",
        device=device,
        batch_size=batch_size
    )


async def fetch_and_analyze_async(
    stock_list: List[str],
    date: str,
    pipeline: Optional[SentimentPipeline] = None
) -> pd.DataFrame:
    """
    异步获取新闻并分析情感
    
    适用于需要在异步环境中调用的场景。
    
    Parameters
    ----------
    stock_list : List[str]
        股票代码列表
    date : str
        日期
    pipeline : Optional[SentimentPipeline]
        情感分析管道，如果为 None 则创建新实例
    
    Returns
    -------
    pd.DataFrame
        情感分析结果
    
    Examples
    --------
    >>> import asyncio
    >>> result = asyncio.run(fetch_and_analyze_async(
    ...     ["000001", "000002"],
    ...     "2024-01-15"
    ... ))
    """
    if pipeline is None:
        pipeline = create_sentiment_pipeline()
    
    # 异步获取新闻
    news_dict = await pipeline.fetcher.fetch_batch(stock_list, date)
    
    # GPU 推理在主线程中执行
    texts = [news_dict.get(code, "") for code in stock_list]
    results = pipeline.analyzer.batch_score(texts)
    
    # 构建结果
    records = []
    for i, code in enumerate(stock_list):
        result = results[i]
        records.append({
            "stock_code": code,
            "date": date,
            "sentiment_score": result.score,
            "confidence": result.confidence,
            "label": result.label
        })
    
    return pd.DataFrame(records)


# ============================================================================
# 主函数（用于独立运行测试）
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    # 测试代码
    print("=" * 60)
    print("情感分析模块测试")
    print("=" * 60)
    
    # 1. 测试单条分析
    print("\n1. 测试单条分析")
    analyzer = SentimentAnalyzer(device="cpu", batch_size=4)
    
    test_texts = [
        "公司业绩大幅增长，股价创新高",
        "股价暴跌，投资者损失惨重",
        "市场行情平稳，交投清淡"
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"  文本: {text[:20]}...")
        print(f"    标签: {result.label}, 分数: {result.score:.2f}, 置信度: {result.confidence:.2f}")
    
    # 2. 测试批量分析
    print("\n2. 测试批量分析")
    batch_results = analyzer.batch_score(test_texts)
    for i, result in enumerate(batch_results):
        print(f"  [{i+1}] {result.label}: {result.score:.2f}")
    
    # 3. 测试完整管道（如果安装了 akshare）
    print("\n3. 测试完整管道")
    try:
        pipeline = create_sentiment_pipeline(device="cpu")
        
        # 使用真实股票代码测试
        test_stocks = ["000001", "600519"]
        test_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"  分析日期: {test_date}")
        print(f"  股票列表: {test_stocks}")
        
        result_df = pipeline.analyze_daily(test_stocks, test_date)
        print("\n  分析结果:")
        print(result_df.to_string(index=False))
        
    except ImportError:
        print("  跳过管道测试（需要安装 akshare）")
    except Exception as e:
        print(f"  管道测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

