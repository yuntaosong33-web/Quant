"""
Tushare 新闻资讯模块

该模块提供新闻数据获取功能，作为 Mixin 类混入主类。
支持 Tushare 和 AKShare 双数据源。

Features
--------
- Tushare Pro 新闻接口
- AKShare 财联社电报（免费无限制）
- 多数据源自动切换
- 新闻缓存机制
"""

from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)

# 全局变量：追踪新闻 API 最后调用时间（跨实例共享）
_GLOBAL_NEWS_API_LAST_CALL = 0.0
_GLOBAL_NEWS_RATE_LIMIT_COUNT = 0
# 全局新闻缓存（跨实例共享，避免重复调用 API）
_GLOBAL_NEWS_CACHE: Dict[str, pd.DataFrame] = {}


class TushareNewsMixin:
    """
    Tushare 新闻资讯 Mixin
    
    提供新闻数据获取功能，需要与 TushareDataLoaderBase 组合使用。
    
    Methods
    -------
    fetch_news(stock_code, start_date, end_date, src)
        获取新闻资讯数据
    fetch_news_multi_source(start_date, end_date, prefer_akshare)
        多数据源新闻获取
    fetch_stock_news(stock_code, days_back)
        获取单只股票相关新闻
    """
    
    def fetch_news(
        self,
        stock_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        src: str = "sina"
    ) -> Optional[pd.DataFrame]:
        """
        获取新闻资讯数据
        
        使用 Tushare Pro news 接口获取财经新闻。
        
        Parameters
        ----------
        stock_code : Optional[str]
            股票代码（6位），如果提供则过滤相关新闻
        start_date : Optional[str]
            开始日期，格式 YYYYMMDD
        end_date : Optional[str]
            结束日期，格式 YYYYMMDD
        src : str
            新闻来源，可选：sina(新浪), wallstreetcn(华尔街见闻), 
            10jqka(同花顺), eastmoney(东方财富), yuncaijing(云财经)
            默认 sina
        
        Returns
        -------
        Optional[pd.DataFrame]
            新闻数据，包含 datetime, title, content, channels 等字段
            失败返回 None
        
        Notes
        -----
        - Tushare Pro 新闻接口需要较高积分权限
        - 如果接口不可用，会返回空 DataFrame
        
        Examples
        --------
        >>> loader = TushareDataLoader()
        >>> news = loader.fetch_news(start_date="20240101", end_date="20240115")
        >>> print(news[['datetime', 'title']].head())
        """
        # 标准化日期格式
        if start_date:
            start_date = start_date.replace("-", "")
        if end_date:
            end_date = end_date.replace("-", "")
        
        global _GLOBAL_NEWS_API_LAST_CALL, _GLOBAL_NEWS_RATE_LIMIT_COUNT
        
        # 检查是否在配置中跳过新闻获取
        if getattr(self, '_skip_news', False):
            logger.debug("新闻获取已在配置中禁用 (tushare.skip_news=true)")
            return pd.DataFrame()
        
        # 检查是否应该跳过新闻获取（频率限制保护）
        if _GLOBAL_NEWS_RATE_LIMIT_COUNT >= 3:
            if _GLOBAL_NEWS_RATE_LIMIT_COUNT >= 100:
                logger.debug("新闻接口今日配额已用完，跳过")
            elif _GLOBAL_NEWS_RATE_LIMIT_COUNT >= 10:
                logger.debug("新闻接口本小时配额已用完，跳过")
            else:
                logger.warning("新闻接口频繁触发限制，本次跳过（需要更高积分权限）")
            return pd.DataFrame()
        
        # 尝试缓存（新闻按日期和来源缓存）
        cache_key = f"news_{src}_{start_date}_{end_date}"
        if stock_code:
            cache_key += f"_{stock_code.replace('.', '')[:6]}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if cache_file.exists():
            try:
                # 检查缓存是否在24小时内
                cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.now() - cache_mtime).total_seconds() < 86400:  # 24小时
                    df = pd.read_parquet(cache_file)
                    if not df.empty:
                        logger.info(f"从缓存加载新闻: {len(df)} 条")
                        return df
            except Exception:
                pass
        
        # 新闻接口特殊限流：每分钟最多 1 次（使用全局变量跨实例共享）
        elapsed = time.time() - _GLOBAL_NEWS_API_LAST_CALL
        if elapsed < self.NEWS_API_INTERVAL:
            wait_time = self.NEWS_API_INTERVAL - elapsed
            logger.info(f"⏳ 新闻接口限流（每分钟1次），等待 {wait_time:.0f} 秒...")
            time.sleep(wait_time)
        
        logger.info(f"获取新闻资讯: src={src}, {start_date} ~ {end_date}")
        
        try:
            # 更新全局最后调用时间
            _GLOBAL_NEWS_API_LAST_CALL = time.time()
            
            df = self._fetch_with_retry(
                self.pro.news,
                src=src,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None:
                # _fetch_with_retry 返回 None 说明可能遇到限流
                # 增加计数器避免后续重复尝试
                _GLOBAL_NEWS_RATE_LIMIT_COUNT += 1
                logger.warning(
                    f"新闻接口请求失败 (累计 {_GLOBAL_NEWS_RATE_LIMIT_COUNT} 次)，"
                    "可能触发配额限制，使用缓存或跳过"
                )
                
                # 尝试返回过期缓存作为回退
                if cache_file.exists():
                    try:
                        df = pd.read_parquet(cache_file)
                        if not df.empty:
                            logger.info(f"使用过期缓存回退: {len(df)} 条新闻")
                            return df
                    except Exception:
                        pass
                
                return pd.DataFrame()
            
            if df.empty:
                logger.debug("无新闻数据")
                return pd.DataFrame()
            
            # 成功则重置全局计数器
            _GLOBAL_NEWS_RATE_LIMIT_COUNT = 0
            
            # 如果指定了股票代码，尝试过滤相关新闻
            if stock_code:
                # 在标题或内容中搜索股票代码或名称
                stock_code_clean = stock_code.replace(".", "")[:6]
                mask = (
                    df["title"].str.contains(stock_code_clean, na=False) |
                    df["content"].str.contains(stock_code_clean, na=False)
                )
                df = df[mask]
            
            # 保存缓存
            if not df.empty:
                try:
                    df.to_parquet(cache_file, index=False)
                    logger.debug(f"新闻已缓存: {cache_file.name}")
                except Exception:
                    pass
            
            logger.info(f"获取新闻成功: {len(df)} 条")
            return df
            
        except Exception as e:
            error_msg = str(e)
            # 记录频率限制（使用全局变量，已在函数开头声明）
            if "每天" in error_msg:
                # 每天限制 - 今天不再尝试
                _GLOBAL_NEWS_RATE_LIMIT_COUNT = 100
                logger.warning(f"⚠️ 新闻接口每天配额已用完，今日跳过新闻获取")
                logger.warning(f"   提示：Tushare 新闻接口需要较高积分（2000+）才能解除限制")
                logger.warning(f"   提示：可在配置中设置 llm.enable_sentiment_filter: false 暂时禁用情绪分析")
            elif "每小时" in error_msg:
                # 每小时限制 - 本次会话内不再尝试
                _GLOBAL_NEWS_RATE_LIMIT_COUNT = 10
                logger.warning(f"⚠️ 新闻接口每小时限制已达上限，本次跳过新闻获取")
                logger.warning(f"   提示：可在配置中设置 llm.enable_sentiment_filter: false 暂时禁用情绪分析")
            elif "每分钟" in error_msg or "频率" in error_msg.lower() or "抱歉" in error_msg:
                _GLOBAL_NEWS_RATE_LIMIT_COUNT += 1
                logger.warning(f"新闻接口频率限制 ({_GLOBAL_NEWS_RATE_LIMIT_COUNT}/3): {e}")
            else:
                logger.warning(f"获取新闻失败: {e}")
            return pd.DataFrame()
    
    def _fetch_news_akshare(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        使用 AKShare 获取财经新闻（升级版：财联社电报为核心源）
        
        AKShare 是免费开源的数据接口，无配额限制。
        优先使用财联社电报作为 A 股最快消息源。
        
        Parameters
        ----------
        start_date : Optional[str]
            开始日期，格式 YYYYMMDD
        end_date : Optional[str]
            结束日期，格式 YYYYMMDD
        
        Returns
        -------
        pd.DataFrame
            新闻数据，包含 datetime, title, content, source 列
        
        Notes
        -----
        数据源优先级:
        1. stock_telegraph_cls(): 财联社电报（A股最快，流式接口，约300条）
        2. stock_zh_a_alerts_cls(): 财联社快讯（补充）
        3. stock_news_em(): 东方财富股票新闻（深度报道补充）
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("AKShare 未安装，无法使用备选新闻源。安装: pip install akshare")
            return pd.DataFrame()
        
        all_news = []
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # =====================================================================
        # 核心源：财联社电报 (A股最快消息源)
        # =====================================================================
        try:
            logger.info("⚡ 获取财联社电报 (CLS Telegraph)...")
            df_telegraph = ak.stock_telegraph_cls(symbol="全部")
            if df_telegraph is not None and not df_telegraph.empty:
                # 标准化列名
                col_mapping = {
                    '发布时间': 'time_str',
                    '发布日期': 'date_str', 
                    '标题': 'title',
                    '内容': 'content'
                }
                df_telegraph = df_telegraph.rename(columns={
                    k: v for k, v in col_mapping.items() if k in df_telegraph.columns
                })
                
                # 处理日期时间
                if 'time_str' in df_telegraph.columns:
                    if 'date_str' in df_telegraph.columns:
                        df_telegraph['datetime'] = pd.to_datetime(
                            df_telegraph['date_str'].astype(str) + ' ' + 
                            df_telegraph['time_str'].astype(str),
                            errors='coerce'
                        )
                    else:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        
                        def infer_date(time_val: str) -> str:
                            try:
                                time_str = str(time_val).strip()
                                if time_str > current_time:
                                    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
                                    return yesterday + ' ' + time_str
                                else:
                                    return today_str + ' ' + time_str
                            except Exception:
                                return today_str + ' ' + str(time_val)
                        
                        df_telegraph['datetime'] = df_telegraph['time_str'].apply(infer_date)
                        df_telegraph['datetime'] = pd.to_datetime(
                            df_telegraph['datetime'], errors='coerce'
                        )
                elif 'datetime' not in df_telegraph.columns:
                    for col in df_telegraph.columns:
                        if '时间' in col or '日期' in col:
                            df_telegraph['datetime'] = pd.to_datetime(
                                df_telegraph[col], errors='coerce'
                            )
                            break
                    else:
                        df_telegraph['datetime'] = pd.Timestamp.now()
                
                if 'title' not in df_telegraph.columns:
                    if 'content' in df_telegraph.columns:
                        df_telegraph['title'] = df_telegraph['content'].str[:50] + '...'
                    else:
                        df_telegraph['title'] = ''
                
                if 'content' not in df_telegraph.columns:
                    if 'title' in df_telegraph.columns:
                        df_telegraph['content'] = df_telegraph['title']
                    else:
                        df_telegraph['content'] = ''
                
                df_telegraph['source'] = '财联社电报'
                all_news.append(df_telegraph[['datetime', 'title', 'content', 'source']])
                logger.info(f"✅ 财联社电报获取成功: {len(df_telegraph)} 条")
        except Exception as e:
            logger.debug(f"财联社电报获取失败: {e}")
        
        # =====================================================================
        # 辅助源1：财联社快讯
        # =====================================================================
        try:
            logger.info("📰 获取财联社快讯 (CLS Alerts)...")
            df_cls = ak.stock_zh_a_alerts_cls()
            if df_cls is not None and not df_cls.empty:
                df_cls = df_cls.rename(columns={
                    '时间': 'datetime',
                    '标题': 'title', 
                    '内容': 'content'
                })
                
                if 'title' not in df_cls.columns and 'content' in df_cls.columns:
                    df_cls['title'] = df_cls['content'].str[:50] + '...'
                if 'content' not in df_cls.columns and 'title' in df_cls.columns:
                    df_cls['content'] = df_cls['title']
                
                if 'datetime' in df_cls.columns:
                    df_cls['datetime'] = pd.to_datetime(df_cls['datetime'], errors='coerce')
                
                df_cls['source'] = '财联社快讯'
                all_news.append(df_cls[['datetime', 'title', 'content', 'source']])
                logger.info(f"✅ 财联社快讯获取成功: {len(df_cls)} 条")
        except Exception as e:
            logger.debug(f"财联社快讯获取失败: {e}")
        
        # =====================================================================
        # 辅助源2：东方财富股票新闻
        # =====================================================================
        try:
            logger.info("📰 获取东方财富新闻 (EM)...")
            df_em = ak.stock_news_em(symbol="全部")
            if df_em is not None and not df_em.empty:
                df_em = df_em.rename(columns={
                    '发布时间': 'datetime',
                    '新闻标题': 'title',
                    '新闻内容': 'content',
                    '文章来源': 'source'
                })
                
                if 'source' not in df_em.columns:
                    df_em['source'] = '东方财富'
                
                if 'datetime' in df_em.columns:
                    df_em['datetime'] = pd.to_datetime(df_em['datetime'], errors='coerce')
                    
                all_news.append(df_em[['datetime', 'title', 'content', 'source']])
                logger.info(f"✅ 东方财富新闻获取成功: {len(df_em)} 条")
        except Exception as e:
            logger.debug(f"东方财富新闻获取失败: {e}")
        
        # =====================================================================
        # 合并、去重、过滤
        # =====================================================================
        if not all_news:
            logger.warning("⚠️ AKShare 所有新闻源均获取失败")
            return pd.DataFrame()
        
        result = pd.concat(all_news, ignore_index=True)
        
        # 基于 content 去重（优先保留财联社电报的数据）
        source_priority = {'财联社电报': 0, '财联社快讯': 1, '东方财富': 2}
        result['_priority'] = result['source'].map(source_priority).fillna(99)
        result = result.sort_values('_priority')
        result = result.drop_duplicates(subset=['content'], keep='first')
        result = result.drop(columns=['_priority'])
        
        # 日期过滤
        if 'datetime' in result.columns:
            try:
                result = result.dropna(subset=['datetime'])
                
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    result = result[result['datetime'] >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date) + timedelta(days=1)
                    result = result[result['datetime'] < end_dt]
                
                result = result.sort_values('datetime', ascending=False)
            except Exception as e:
                logger.debug(f"日期过滤异常: {e}")
        
        logger.info(
            f"📊 AKShare 新闻汇总: {len(result)} 条 | "
            f"电报: {len(result[result['source'] == '财联社电报']) if '财联社电报' in result['source'].values else 0}, "
            f"快讯: {len(result[result['source'] == '财联社快讯']) if '财联社快讯' in result['source'].values else 0}, "
            f"东财: {len(result[result['source'] == '东方财富']) if '东方财富' in result['source'].values else 0}"
        )
        return result
    
    def fetch_news_multi_source(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        prefer_akshare: bool = False
    ) -> pd.DataFrame:
        """
        多数据源新闻获取（自动切换）
        
        根据配置和可用性选择数据源：
        - tushare: 仅使用 Tushare（需要积分）
        - akshare: 仅使用 AKShare（免费无限制）
        - auto: 优先 Tushare，限流时自动切换到 AKShare
        
        Parameters
        ----------
        start_date : Optional[str]
            开始日期
        end_date : Optional[str]
            结束日期
        prefer_akshare : bool
            是否优先使用 AKShare，默认 False
        
        Returns
        -------
        pd.DataFrame
            新闻数据
        """
        global _GLOBAL_NEWS_RATE_LIMIT_COUNT
        
        # 读取配置的新闻数据源
        news_source = getattr(self, '_news_source', 'auto')
        
        # 如果配置为仅使用 AKShare
        if news_source == 'akshare':
            logger.info("配置指定使用 AKShare 作为新闻数据源")
            return self._fetch_news_akshare(start_date, end_date)
        
        # 如果配置为仅使用 Tushare
        if news_source == 'tushare':
            return self.fetch_news(start_date=start_date, end_date=end_date)
        
        # 自动模式 (auto)
        if _GLOBAL_NEWS_RATE_LIMIT_COUNT > 0 or prefer_akshare:
            logger.debug(f"Tushare 新闻接口已限流 (计数={_GLOBAL_NEWS_RATE_LIMIT_COUNT})，使用 AKShare")
            return self._fetch_news_akshare(start_date, end_date)
        
        # 尝试 Tushare
        df = self.fetch_news(start_date=start_date, end_date=end_date)
        
        # 如果 Tushare 失败，回退到 AKShare
        if df.empty:
            logger.info("Tushare 新闻获取失败，切换到 AKShare")
            return self._fetch_news_akshare(start_date, end_date)
        
        return df
    
    def fetch_all_news_once(
        self,
        days_back: int = 7,
        src: str = "sina"
    ) -> pd.DataFrame:
        """
        一次性获取所有新闻（优化：避免多次 API 调用）
        
        获取最近几天的所有新闻，缓存后供多只股票使用。
        新闻接口每分钟只能调用1次，因此一次获取全部数据更高效。
        
        Parameters
        ----------
        days_back : int
            回溯天数，默认 7 天
        src : str
            新闻源，默认 sina
        
        Returns
        -------
        pd.DataFrame
            所有新闻数据
        """
        global _GLOBAL_NEWS_CACHE
        
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        
        # 使用全局缓存（跨实例共享）
        cache_key = f"all_news_{start_date}_{end_date}"
        if cache_key in _GLOBAL_NEWS_CACHE:
            cached = _GLOBAL_NEWS_CACHE[cache_key]
            if cached is not None and not cached.empty:
                logger.debug(f"使用全局缓存的新闻数据: {len(cached)} 条")
                return cached
        
        # 获取所有新闻
        df = self.fetch_news_multi_source(
            start_date=start_date,
            end_date=end_date
        )
        
        # 缓存到全局变量
        _GLOBAL_NEWS_CACHE[cache_key] = df if df is not None else pd.DataFrame()
        
        if df is not None and not df.empty:
            logger.info(f"📰 一次性获取新闻完成: {len(df)} 条，可供所有股票使用")
        
        return df if df is not None else pd.DataFrame()
    
    def _load_all_stock_names(self) -> None:
        """
        一次性加载所有股票名称到全局缓存
        
        避免为每只股票单独查询 API。
        """
        global _GLOBAL_NEWS_CACHE
        
        # 检查是否已加载
        if '_stock_names_loaded' in _GLOBAL_NEWS_CACHE:
            return
        
        logger.info("📋 批量加载股票名称...")
        
        try:
            df = self._fetch_with_retry(
                self.pro.stock_basic,
                exchange='',
                list_status='L',
                fields='ts_code,name'
            )
            
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    code = row['ts_code'][:6]
                    name = row['name']
                    _GLOBAL_NEWS_CACHE[f"stock_name_{code}"] = name
                
                logger.info(f"📋 股票名称加载完成: {len(df)} 只")
            
            _GLOBAL_NEWS_CACHE['_stock_names_loaded'] = True
            
        except Exception as e:
            logger.warning(f"批量加载股票名称失败: {e}")
            _GLOBAL_NEWS_CACHE['_stock_names_loaded'] = True
    
    def _get_stock_name(self, stock_code: str) -> Optional[str]:
        """
        获取股票名称（用于新闻匹配）
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位）
        
        Returns
        -------
        Optional[str]
            股票名称，如"贵州茅台"，获取失败返回 None
        """
        global _GLOBAL_NEWS_CACHE
        
        # 确保股票名称已批量加载
        if '_stock_names_loaded' not in _GLOBAL_NEWS_CACHE:
            self._load_all_stock_names()
        
        cache_key = f"stock_name_{stock_code}"
        return _GLOBAL_NEWS_CACHE.get(cache_key)
    
    def fetch_stock_news(
        self,
        stock_code: str,
        days_back: int = 7
    ) -> str:
        """
        获取单只股票相关新闻（用于情感分析）
        
        从缓存的全量新闻中筛选与指定股票相关的新闻。
        优化：只调用一次 API 获取全量新闻，然后本地筛选。
        
        Parameters
        ----------
        stock_code : str
            股票代码（6位）
        days_back : int
            回溯天数，默认 7 天
        
        Returns
        -------
        str
            合并的新闻文本，用于情感分析
            无新闻时返回空字符串
        """
        # 先获取全量新闻（会自动缓存）
        all_news_df = self.fetch_all_news_once(days_back=days_back)
        
        if all_news_df.empty:
            logger.debug(f"无新闻数据可用")
            return ""
        
        # 从全量新闻中筛选
        stock_code_clean = stock_code.replace(".", "")[:6]
        
        # 获取股票名称用于匹配
        stock_name = self._get_stock_name(stock_code_clean)
        
        # 创建 mask
        mask = pd.Series(False, index=all_news_df.index)
        
        # 匹配条件
        search_terms = [stock_code_clean]
        if stock_name:
            search_terms.append(stock_name)
            if len(stock_name) > 2:
                search_terms.append(stock_name[-2:])
        
        for term in search_terms:
            if "title" in all_news_df.columns:
                mask = mask | all_news_df["title"].str.contains(term, na=False, regex=False)
            if "content" in all_news_df.columns:
                mask = mask | all_news_df["content"].str.contains(term, na=False, regex=False)
        
        filtered_df = all_news_df.loc[mask]
        
        if filtered_df.empty:
            logger.debug(f"股票 {stock_code} ({stock_name or '未知'}) 无相关新闻")
            return ""
        
        # 提取标题和内容
        all_news = []
        for _, row in filtered_df.head(5).iterrows():
            title = row.get("title", "")
            content = row.get("content", "")
            if title:
                all_news.append(str(title))
            if content and len(str(content)) < 500:
                all_news.append(str(content)[:200])
        
        if not all_news:
            return ""
        
        # 合并新闻文本
        combined = " | ".join(all_news)
        
        # 截断
        if len(combined) > 1500:
            combined = combined[:1500] + "..."
        
        logger.debug(f"获取股票新闻成功: {stock_code}, {len(all_news)} 条")
        return combined

