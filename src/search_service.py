# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 搜索服务模块
===================================

职责：
1. 提供统一的新闻搜索接口
2. 支持 Bocha、Tavily、Brave、SerpAPI、SearXNG 多种搜索引擎
3. 多 Key 负载均衡和故障转移
4. 搜索结果缓存和格式化
"""

import logging
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import StringIO
from typing import List, Dict, Any, Optional, Tuple
from itertools import cycle
import requests
import pandas as pd
try:
    from newspaper import Article, Config
except Exception:  # pragma: no cover - optional dependency
    Article = None
    Config = None
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from data_provider.fundamental_adapter import UsSecFundamentalAdapter
from data_provider.us_index_mapping import is_us_index_code
from src.config import (
    NEWS_STRATEGY_WINDOWS,
    normalize_news_strategy_profile,
    resolve_news_window_days,
)
from src.core.trading_calendar import get_market_for_stock

logger = logging.getLogger(__name__)

_BACKGROUND_SOURCE_EXCLUSIONS = "-site:wikipedia.org -site:wikiwand.com -site:wikidata.org"

# Transient network errors (retryable)
_SEARCH_TRANSIENT_EXCEPTIONS = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_SEARCH_TRANSIENT_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _post_with_retry(url: str, *, headers: Dict[str, str], json: Dict[str, Any], timeout: int) -> requests.Response:
    """POST with retry on transient SSL/network errors."""
    return requests.post(url, headers=headers, json=json, timeout=timeout)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_SEARCH_TRANSIENT_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _get_with_retry(
    url: str, *, headers: Dict[str, str], params: Dict[str, Any], timeout: int
) -> requests.Response:
    """GET with retry on transient SSL/network errors."""
    return requests.get(url, headers=headers, params=params, timeout=timeout)


def fetch_url_content(url: str, timeout: int = 5) -> str:
    """
    获取 URL 网页正文内容 (使用 newspaper3k)
    """
    if Article is None or Config is None:
        logger.debug("newspaper3k not installed; skip fetching URL content")
        return ""
    try:
        # 配置 newspaper3k
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        config.request_timeout = timeout
        config.fetch_images = False  # 不下载图片
        config.memoize_articles = False # 不缓存

        article = Article(url, config=config, language='zh') # 默认中文，但也支持其他
        article.download()
        article.parse()

        # 获取正文
        text = article.text.strip()

        # 简单的后处理，去除空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        return text[:1500]  # 限制返回长度（比 bs4 稍微多一点，因为 newspaper 解析更干净）
    except Exception as e:
        logger.debug(f"Fetch content failed for {url}: {e}")

    return ""


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    snippet: str  # 摘要
    url: str
    source: str  # 来源网站
    published_date: Optional[str] = None
    
    def to_text(self) -> str:
        """转换为文本格式"""
        date_str = f" ({self.published_date})" if self.published_date else ""
        return f"【{self.source}】{self.title}{date_str}\n{self.snippet}"


@dataclass 
class SearchResponse:
    """搜索响应"""
    query: str
    results: List[SearchResult]
    provider: str  # 使用的搜索引擎
    success: bool = True
    error_message: Optional[str] = None
    search_time: float = 0.0  # 搜索耗时（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_context(self, max_results: int = 5) -> str:
        """将搜索结果转换为可用于 AI 分析的上下文"""
        if not self.success or not self.results:
            return f"搜索 '{self.query}' 未找到相关结果。"
        
        lines = [f"【{self.query} 搜索结果】（来源：{self.provider}）"]
        for i, result in enumerate(self.results[:max_results], 1):
            lines.append(f"\n{i}. {result.to_text()}")
        
        return "\n".join(lines)


class BaseSearchProvider(ABC):
    """搜索引擎基类"""
    
    def __init__(self, api_keys: List[str], name: str):
        """
        初始化搜索引擎
        
        Args:
            api_keys: API Key 列表（支持多个 key 负载均衡）
            name: 搜索引擎名称
        """
        self._api_keys = api_keys
        self._name = name
        self._key_cycle = cycle(api_keys) if api_keys else None
        self._key_usage: Dict[str, int] = {key: 0 for key in api_keys}
        self._key_errors: Dict[str, int] = {key: 0 for key in api_keys}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_available(self) -> bool:
        """检查是否有可用的 API Key"""
        return bool(self._api_keys)
    
    def _get_next_key(self) -> Optional[str]:
        """
        获取下一个可用的 API Key（负载均衡）
        
        策略：轮询 + 跳过错误过多的 key
        """
        if not self._key_cycle:
            return None
        
        # 最多尝试所有 key
        for _ in range(len(self._api_keys)):
            key = next(self._key_cycle)
            # 跳过错误次数过多的 key（超过 3 次）
            if self._key_errors.get(key, 0) < 3:
                return key
        
        # 所有 key 都有问题，重置错误计数并返回第一个
        logger.warning(f"[{self._name}] 所有 API Key 都有错误记录，重置错误计数")
        self._key_errors = {key: 0 for key in self._api_keys}
        return self._api_keys[0] if self._api_keys else None
    
    def _record_success(self, key: str) -> None:
        """记录成功使用"""
        self._key_usage[key] = self._key_usage.get(key, 0) + 1
        # 成功后减少错误计数
        if key in self._key_errors and self._key_errors[key] > 0:
            self._key_errors[key] -= 1
    
    def _record_error(self, key: str) -> None:
        """记录错误"""
        self._key_errors[key] = self._key_errors.get(key, 0) + 1
        logger.warning(f"[{self._name}] API Key {key[:8]}... 错误计数: {self._key_errors[key]}")
    
    @abstractmethod
    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行搜索（子类实现）"""
        pass
    
    def search(self, query: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        """
        执行搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大返回结果数
            days: 搜索最近几天的时间范围（默认7天）
            
        Returns:
            SearchResponse 对象
        """
        api_key = self._get_next_key()
        if not api_key:
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=f"{self._name} 未配置 API Key"
            )
        
        start_time = time.time()
        try:
            response = self._do_search(query, api_key, max_results, days=days)
            response.search_time = time.time() - start_time
            
            if response.success:
                self._record_success(api_key)
                logger.info(f"[{self._name}] 搜索 '{query}' 成功，返回 {len(response.results)} 条结果，耗时 {response.search_time:.2f}s")
            else:
                self._record_error(api_key)
                logger.warning(
                    f"[{self._name}] 搜索 '{query}' 失败: {response.error_message or '未知错误'}"
                )

            return response
            
        except Exception as e:
            self._record_error(api_key)
            elapsed = time.time() - start_time
            logger.error(f"[{self._name}] 搜索 '{query}' 失败: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=str(e),
                search_time=elapsed
            )


class TavilySearchProvider(BaseSearchProvider):
    """
    Tavily 搜索引擎
    
    特点：
    - 专为 AI/LLM 优化的搜索 API
    - 免费版每月 1000 次请求
    - 返回结构化的搜索结果
    
    文档：https://docs.tavily.com/
    """
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Tavily")
    
    def _do_search(
        self,
        query: str,
        api_key: str,
        max_results: int,
        days: int = 7,
        topic: Optional[str] = None,
    ) -> SearchResponse:
        """执行 Tavily 搜索"""
        try:
            from tavily import TavilyClient
        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="tavily-python 未安装，请运行: pip install tavily-python"
            )
        
        try:
            client = TavilyClient(api_key=api_key)
            
            # 执行搜索（优化：使用advanced深度、限制最近几天）
            search_kwargs: Dict[str, Any] = {
                "query": query,
                "search_depth": "advanced",  # advanced 获取更多结果
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "days": days,  # 搜索最近天数的内容
            }
            if topic is not None:
                search_kwargs["topic"] = topic

            response = client.search(
                **search_kwargs,
            )
            
            # 记录原始响应到日志
            logger.info(f"[Tavily] 搜索完成，query='{query}', 返回 {len(response.get('results', []))} 条结果")
            logger.debug(f"[Tavily] 原始响应: {response}")
            
            # 解析结果
            results = []
            for item in response.get('results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=item.get('content', '')[:500],  # 截取前500字
                    url=item.get('url', ''),
                    source=self._extract_domain(item.get('url', '')),
                    published_date=item.get('published_date') or item.get('publishedDate'),
                ))
            
            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )
            
        except Exception as e:
            error_msg = str(e)
            # 检查是否是配额问题
            if 'rate limit' in error_msg.lower() or 'quota' in error_msg.lower():
                error_msg = f"API 配额已用尽: {error_msg}"
            
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )

    def search(
        self,
        query: str,
        max_results: int = 5,
        days: int = 7,
        topic: Optional[str] = None,
    ) -> SearchResponse:
        """执行 Tavily 搜索，可按调用方选择是否启用新闻 topic。"""
        if topic is None:
            return super().search(query, max_results=max_results, days=days)

        api_key = self._get_next_key()
        if not api_key:
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=f"{self._name} 未配置 API Key"
            )

        start_time = time.time()
        try:
            response = self._do_search(query, api_key, max_results, days=days, topic=topic)
            response.search_time = time.time() - start_time

            if response.success:
                self._record_success(api_key)
                logger.info(f"[{self._name}] 搜索 '{query}' 成功，返回 {len(response.results)} 条结果，耗时 {response.search_time:.2f}s")
            else:
                self._record_error(api_key)
                logger.warning(
                    f"[{self._name}] 搜索 '{query}' 失败: {response.error_message or '未知错误'}"
                )

            return response

        except Exception as e:
            self._record_error(api_key)
            elapsed = time.time() - start_time
            logger.error(f"[{self._name}] 搜索 '{query}' 失败: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self._name,
                success=False,
                error_message=str(e),
                search_time=elapsed
            )
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名作为来源"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain or '未知来源'
        except Exception:
            return '未知来源'


class SerpAPISearchProvider(BaseSearchProvider):
    """
    SerpAPI 搜索引擎
    
    特点：
    - 支持 Google、Bing、百度等多种搜索引擎
    - 免费版每月 100 次请求
    - 返回真实的搜索结果
    
    文档：https://serpapi.com/baidu-search-api?utm_source=github_daily_stock_analysis
    """
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "SerpAPI")
    
    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行 SerpAPI 搜索"""
        try:
            from serpapi import GoogleSearch
        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="google-search-results 未安装，请运行: pip install google-search-results"
            )
        
        try:
            # 确定时间范围参数 tbs
            tbs = "qdr:w"  # 默认一周
            if days <= 1:
                tbs = "qdr:d"  # 过去24小时
            elif days <= 7:
                tbs = "qdr:w"  # 过去一周
            elif days <= 30:
                tbs = "qdr:m"  # 过去一月
            else:
                tbs = "qdr:y"  # 过去一年

            # 使用 Google 搜索 (获取 Knowledge Graph, Answer Box 等)
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "google_domain": "google.com.hk", # 使用香港谷歌，中文支持较好
                "hl": "zh-cn",  # 中文界面
                "gl": "cn",     # 中国地区偏好
                "tbs": tbs,     # 时间范围限制
                "num": max_results # 请求的结果数量，注意：Google API有时不严格遵守
            }
            
            search = GoogleSearch(params)
            response = search.get_dict()
            
            # 记录原始响应到日志
            logger.debug(f"[SerpAPI] 原始响应 keys: {response.keys()}")
            
            # 解析结果
            results = []
            
            # 1. 解析 Knowledge Graph (知识图谱)
            kg = response.get('knowledge_graph', {})
            if kg:
                title = kg.get('title', '知识图谱')
                desc = kg.get('description', '')
                
                # 提取额外属性
                details = []
                for key in ['type', 'founded', 'headquarters', 'employees', 'ceo']:
                    val = kg.get(key)
                    if val:
                        details.append(f"{key}: {val}")
                        
                snippet = f"{desc}\n" + " | ".join(details) if details else desc
                
                results.append(SearchResult(
                    title=f"[知识图谱] {title}",
                    snippet=snippet,
                    url=kg.get('source', {}).get('link', ''),
                    source="Google Knowledge Graph"
                ))
                
            # 2. 解析 Answer Box (精选回答/行情卡片)
            ab = response.get('answer_box', {})
            if ab:
                ab_title = ab.get('title', '精选回答')
                ab_snippet = ""
                
                # 财经类回答
                if ab.get('type') == 'finance_results':
                    stock = ab.get('stock', '')
                    price = ab.get('price', '')
                    currency = ab.get('currency', '')
                    movement = ab.get('price_movement', {})
                    mv_val = movement.get('percentage', 0)
                    mv_dir = movement.get('movement', '')
                    
                    ab_title = f"[行情卡片] {stock}"
                    ab_snippet = f"价格: {price} {currency}\n涨跌: {mv_dir} {mv_val}%"
                    
                    # 提取表格数据
                    if 'table' in ab:
                        table_data = []
                        for row in ab['table']:
                            if 'name' in row and 'value' in row:
                                table_data.append(f"{row['name']}: {row['value']}")
                        if table_data:
                            ab_snippet += "\n" + "; ".join(table_data)
                            
                # 普通文本回答
                elif 'snippet' in ab:
                    ab_snippet = ab.get('snippet', '')
                    list_items = ab.get('list', [])
                    if list_items:
                        ab_snippet += "\n" + "\n".join([f"- {item}" for item in list_items])
                
                elif 'answer' in ab:
                    ab_snippet = ab.get('answer', '')
                    
                if ab_snippet:
                    results.append(SearchResult(
                        title=f"[精选回答] {ab_title}",
                        snippet=ab_snippet,
                        url=ab.get('link', '') or ab.get('displayed_link', ''),
                        source="Google Answer Box"
                    ))

            # 3. 解析 Related Questions (相关问题)
            rqs = response.get('related_questions', [])
            for rq in rqs[:3]: # 取前3个
                question = rq.get('question', '')
                snippet = rq.get('snippet', '')
                link = rq.get('link', '')
                
                if question and snippet:
                     results.append(SearchResult(
                        title=f"[相关问题] {question}",
                        snippet=snippet,
                        url=link,
                        source="Google Related Questions"
                     ))

            # 4. 解析 Organic Results (自然搜索结果)
            organic_results = response.get('organic_results', [])

            for item in organic_results[:max_results]:
                link = item.get('link', '')
                snippet = item.get('snippet', '')

                # 增强：如果需要，解析网页正文
                # 策略：如果摘要太短，或者为了获取更多信息，可以请求网页
                # 这里我们对所有结果尝试获取正文，但为了性能，仅获取前1000字符
                content = ""
                if link:
                   try:
                       fetched_content = fetch_url_content(link, timeout=5)
                       if fetched_content:
                           # 如果获取到了正文，将其拼接到 snippet 中，或者替换 snippet
                           # 这里选择拼接，保留原摘要
                           content = fetched_content
                           if len(content) > 500:
                               snippet = f"{snippet}\n\n【网页详情】\n{content[:500]}..."
                           else:
                               snippet = f"{snippet}\n\n【网页详情】\n{content}"
                   except Exception as e:
                       logger.debug(f"[SerpAPI] Fetch content failed: {e}")

                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=snippet[:1000], # 限制总长度
                    url=link,
                    source=item.get('source', self._extract_domain(link)),
                    published_date=item.get('date'),
                ))

            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )
            
        except Exception as e:
            error_msg = str(e)
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '') or '未知来源'
        except Exception:
            return '未知来源'


class BochaSearchProvider(BaseSearchProvider):
    """
    博查搜索引擎
    
    特点：
    - 专为AI优化的中文搜索API
    - 结果准确、摘要完整
    - 支持时间范围过滤和AI摘要
    - 兼容Bing Search API格式
    
    文档：https://bocha-ai.feishu.cn/wiki/RXEOw02rFiwzGSkd9mUcqoeAnNK
    """
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Bocha")
    
    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行博查搜索"""
        try:
            import requests
        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="requests 未安装，请运行: pip install requests"
            )
        
        try:
            # API 端点
            url = "https://api.bocha.cn/v1/web-search"
            
            # 请求头
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # 确定时间范围
            freshness = "oneWeek"
            if days <= 1:
                freshness = "oneDay"
            elif days <= 7:
                freshness = "oneWeek"
            elif days <= 30:
                freshness = "oneMonth"
            else:
                freshness = "oneYear"

            # 请求参数（严格按照API文档）
            payload = {
                "query": query,
                "freshness": freshness,  # 动态时间范围
                "summary": True,  # 启用AI摘要
                "count": min(max_results, 50)  # 最大50条
            }
            
            # 执行搜索（带瞬时 SSL/网络错误重试）
            response = _post_with_retry(url, headers=headers, json=payload, timeout=10)
            
            # 检查HTTP状态码
            if response.status_code != 200:
                # 尝试解析错误信息
                try:
                    if response.headers.get('content-type', '').startswith('application/json'):
                        error_data = response.json()
                        error_message = error_data.get('message', response.text)
                    else:
                        error_message = response.text
                except Exception:
                    error_message = response.text
                
                # 根据错误码处理
                if response.status_code == 403:
                    error_msg = f"余额不足: {error_message}"
                elif response.status_code == 401:
                    error_msg = f"API KEY无效: {error_message}"
                elif response.status_code == 400:
                    error_msg = f"请求参数错误: {error_message}"
                elif response.status_code == 429:
                    error_msg = f"请求频率达到限制: {error_message}"
                else:
                    error_msg = f"HTTP {response.status_code}: {error_message}"
                
                logger.warning(f"[Bocha] 搜索失败: {error_msg}")
                
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )
            
            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"响应JSON解析失败: {str(e)}"
                logger.error(f"[Bocha] {error_msg}")
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )
            
            # 检查响应code
            if data.get('code') != 200:
                error_msg = data.get('msg') or f"API返回错误码: {data.get('code')}"
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )
            
            # 记录原始响应到日志
            logger.info(f"[Bocha] 搜索完成，query='{query}'")
            logger.debug(f"[Bocha] 原始响应: {data}")
            
            # 解析搜索结果
            results = []
            web_pages = data.get('data', {}).get('webPages', {})
            value_list = web_pages.get('value', [])
            
            for item in value_list[:max_results]:
                # 优先使用summary（AI摘要），fallback到snippet
                snippet = item.get('summary') or item.get('snippet', '')
                
                # 截取摘要长度
                if snippet:
                    snippet = snippet[:500]
                
                results.append(SearchResult(
                    title=item.get('name', ''),
                    snippet=snippet,
                    url=item.get('url', ''),
                    source=item.get('siteName') or self._extract_domain(item.get('url', '')),
                    published_date=item.get('datePublished'),  # UTC+8格式，无需转换
                ))
            
            logger.info(f"[Bocha] 成功解析 {len(results)} 条结果")
            
            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )
            
        except requests.exceptions.Timeout:
            error_msg = "请求超时"
            logger.error(f"[Bocha] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"[Bocha] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"[Bocha] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名作为来源"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain or '未知来源'
        except Exception:
            return '未知来源'


class MiniMaxSearchProvider(BaseSearchProvider):
    """
    MiniMax Web Search (Coding Plan API)

    Features:
    - Backed by MiniMax Coding Plan subscription
    - Returns structured organic results with title/link/snippet/date
    - No native time-range parameter; time filtering is done via query
      augmentation and client-side date filtering
    - Circuit-breaker protection: 3 consecutive failures -> 300s cooldown

    API endpoint: POST https://api.minimaxi.com/v1/coding_plan/search
    """

    API_ENDPOINT = "https://api.minimaxi.com/v1/coding_plan/search"

    # Circuit-breaker settings
    _CB_FAILURE_THRESHOLD = 3
    _CB_COOLDOWN_SECONDS = 300  # 5 minutes

    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "MiniMax")
        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until: float = 0.0

    @property
    def is_available(self) -> bool:
        """Check availability considering circuit breaker state."""
        if not super().is_available:
            return False
        if self._consecutive_failures >= self._CB_FAILURE_THRESHOLD:
            if time.time() < self._circuit_open_until:
                return False
            # Cooldown expired -> half-open, allow one probe
        return True

    def _record_success(self, key: str) -> None:
        super()._record_success(key)
        # Reset circuit breaker on success
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_error(self, key: str) -> None:
        super()._record_error(key)
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._CB_FAILURE_THRESHOLD:
            self._circuit_open_until = time.time() + self._CB_COOLDOWN_SECONDS
            logger.warning(
                f"[MiniMax] Circuit breaker OPEN – "
                f"{self._consecutive_failures} consecutive failures, "
                f"cooldown {self._CB_COOLDOWN_SECONDS}s"
            )

    # ------------------------------------------------------------------
    # Time-range helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_hint(days: int, is_chinese: bool = True) -> str:
        """Build a time-hint string to append to the search query."""
        if is_chinese:
            if days <= 1:
                return "今天"
            elif days <= 3:
                return "最近三天"
            elif days <= 7:
                return "最近一周"
            else:
                return "最近一个月"
        else:
            if days <= 1:
                return "today"
            elif days <= 3:
                return "past 3 days"
            elif days <= 7:
                return "past week"
            else:
                return "past month"

    @staticmethod
    def _is_within_days(date_str: Optional[str], days: int) -> bool:
        """Check whether *date_str* falls within the last *days* days.

        Accepts common formats: ``2025-06-01``, ``2025/06/01``,
        ``Jun 1, 2025``, ISO-8601 with timezone, etc.
        Returns True when date_str is None or unparseable (keep the result).
        """
        if not date_str:
            return True
        try:
            from dateutil import parser as dateutil_parser
            dt = dateutil_parser.parse(date_str, fuzzy=True)
            from datetime import timedelta, timezone
            now = datetime.now(timezone.utc) if dt.tzinfo else datetime.now()
            return (now - dt) <= timedelta(days=days + 1)  # +1 buffer
        except Exception:
            return True  # Keep result when date is unparseable

    # ------------------------------------------------------------------

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """Execute MiniMax web search."""
        try:
            # Detect language hint from query (simple heuristic)
            has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in query)
            time_hint = self._time_hint(days, is_chinese=has_cjk)
            augmented_query = f"{query} {time_hint}"

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'MM-API-Source': 'Minimax-MCP',
            }
            payload = {"q": augmented_query}

            response = _post_with_retry(
                self.API_ENDPOINT, headers=headers, json=payload, timeout=15
            )

            # HTTP error handling
            if response.status_code != 200:
                error_msg = self._parse_http_error(response)
                logger.warning(f"[MiniMax] Search failed: {error_msg}")
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg,
                )

            data = response.json()

            # Check base_resp status
            base_resp = data.get('base_resp', {})
            if base_resp.get('status_code', 0) != 0:
                error_msg = base_resp.get('status_msg', 'Unknown API error')
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg,
                )

            logger.info(f"[MiniMax] Search done, query='{query}'")
            logger.debug(f"[MiniMax] Raw response keys: {list(data.keys())}")

            # Parse organic results
            results: List[SearchResult] = []
            for item in data.get('organic', []):
                date_val = item.get('date')

                # Client-side time filtering
                if not self._is_within_days(date_val, days):
                    continue

                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=(item.get('snippet', '') or '')[:500],
                    url=item.get('link', ''),
                    source=self._extract_domain(item.get('link', '')),
                    published_date=date_val,
                ))

                if len(results) >= max_results:
                    break

            logger.info(f"[MiniMax] Parsed {len(results)} results (after time filter)")

            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )

        except requests.exceptions.Timeout:
            error_msg = "Request timeout"
            logger.error(f"[MiniMax] {error_msg}")
            return SearchResponse(
                query=query, results=[], provider=self.name,
                success=False, error_message=error_msg,
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {e}"
            logger.error(f"[MiniMax] {error_msg}")
            return SearchResponse(
                query=query, results=[], provider=self.name,
                success=False, error_message=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"[MiniMax] {error_msg}")
            return SearchResponse(
                query=query, results=[], provider=self.name,
                success=False, error_message=error_msg,
            )

    @staticmethod
    def _parse_http_error(response) -> str:
        """Parse HTTP error response from MiniMax API."""
        try:
            ct = response.headers.get('content-type', '')
            if 'json' in ct:
                err = response.json()
                base_resp = err.get('base_resp', {})
                msg = base_resp.get('status_msg') or err.get('message') or str(err)
                return msg
            return response.text[:200]
        except Exception:
            return f"HTTP {response.status_code}: {response.text[:200]}"

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL as source label."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain or '未知来源'
        except Exception:
            return '未知来源'


class BraveSearchProvider(BaseSearchProvider):
    """
    Brave Search 搜索引擎

    特点：
    - 隐私优先的独立搜索引擎
    - 索引超过300亿页面
    - 免费层可用
    - 支持时间范围过滤

    文档：https://brave.com/search/api/
    """

    API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Brave")

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行 Brave 搜索"""
        try:
            # 请求头
            headers = {
                'X-Subscription-Token': api_key,
                'Accept': 'application/json'
            }

            # 确定时间范围（freshness 参数）
            if days <= 1:
                freshness = "pd"  # Past day (24小时)
            elif days <= 7:
                freshness = "pw"  # Past week
            elif days <= 30:
                freshness = "pm"  # Past month
            else:
                freshness = "py"  # Past year

            # 请求参数
            params = {
                "q": query,
                "count": min(max_results, 20),  # Brave 最大支持20条
                "freshness": freshness,
                "search_lang": "en",  # 英文内容（US股票优先）
                "country": "US",  # 美国区域偏好
                "safesearch": "moderate"
            }

            # 执行搜索（GET 请求）
            response = requests.get(
                self.API_ENDPOINT,
                headers=headers,
                params=params,
                timeout=10
            )

            # 检查HTTP状态码
            if response.status_code != 200:
                error_msg = self._parse_error(response)
                logger.warning(f"[Brave] 搜索失败: {error_msg}")
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )

            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"响应JSON解析失败: {str(e)}"
                logger.error(f"[Brave] {error_msg}")
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )

            logger.info(f"[Brave] 搜索完成，query='{query}'")
            logger.debug(f"[Brave] 原始响应: {data}")

            # 解析搜索结果
            results = []
            web_data = data.get('web', {})
            web_results = web_data.get('results', [])

            for item in web_results[:max_results]:
                # 解析发布日期（ISO 8601 格式）
                published_date = None
                age = item.get('age') or item.get('page_age')
                if age:
                    try:
                        # 转换 ISO 格式为简单日期字符串
                        dt = datetime.fromisoformat(age.replace('Z', '+00:00'))
                        published_date = dt.strftime('%Y-%m-%d')
                    except (ValueError, AttributeError):
                        published_date = age  # 解析失败时使用原始值

                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=item.get('description', '')[:500],  # 截取到500字符
                    url=item.get('url', ''),
                    source=self._extract_domain(item.get('url', '')),
                    published_date=published_date
                ))

            logger.info(f"[Brave] 成功解析 {len(results)} 条结果")

            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True
            )

        except requests.exceptions.Timeout:
            error_msg = "请求超时"
            logger.error(f"[Brave] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败: {str(e)}"
            logger.error(f"[Brave] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"[Brave] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )

    def _parse_error(self, response) -> str:
        """解析错误响应"""
        try:
            if response.headers.get('content-type', '').startswith('application/json'):
                error_data = response.json()
                # Brave API 返回的错误格式
                if 'message' in error_data:
                    return error_data['message']
                if 'error' in error_data:
                    return error_data['error']
                return str(error_data)
            return response.text[:200]
        except Exception:
            return f"HTTP {response.status_code}: {response.text[:200]}"

    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名作为来源"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain or '未知来源'
        except Exception:
            return '未知来源'


class SearXNGSearchProvider(BaseSearchProvider):
    """
    SearXNG search engine (self-hosted, no quota).

    Self-hosted instances are used when explicitly configured.
    Otherwise, the provider can lazily discover public instances from
    searx.space and rotate across them with per-request failover.
    """

    PUBLIC_INSTANCES_URL = "https://searx.space/data/instances.json"
    PUBLIC_INSTANCES_CACHE_TTL_SECONDS = 3600
    PUBLIC_INSTANCES_STALE_REFRESH_BACKOFF_SECONDS = 60
    PUBLIC_INSTANCES_POOL_LIMIT = 20
    PUBLIC_INSTANCES_MAX_ATTEMPTS = 3
    PUBLIC_INSTANCES_TIMEOUT_SECONDS = 5
    SELF_HOSTED_TIMEOUT_SECONDS = 10

    _public_instances_cache: Optional[Tuple[float, List[str]]] = None
    _public_instances_stale_retry_after: float = 0.0
    _public_instances_lock = threading.Lock()

    def __init__(self, base_urls: Optional[List[str]] = None, *, use_public_instances: bool = False):
        normalized_base_urls = [url.rstrip("/") for url in (base_urls or []) if url.strip()]
        super().__init__(normalized_base_urls, "SearXNG")
        self._base_urls = normalized_base_urls
        self._use_public_instances = bool(use_public_instances and not self._base_urls)
        self._cursor = 0
        self._cursor_lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return bool(self._base_urls) or self._use_public_instances

    @classmethod
    def reset_public_instance_cache(cls) -> None:
        """Reset the shared searx.space cache (used by tests)."""
        with cls._public_instances_lock:
            cls._public_instances_cache = None
            cls._public_instances_stale_retry_after = 0.0

    @staticmethod
    def _parse_http_error(response) -> str:
        """Parse HTTP error details for easier diagnostics."""
        try:
            raw_content_type = response.headers.get("content-type", "")
            content_type = raw_content_type if isinstance(raw_content_type, str) else ""
            if "json" in content_type:
                error_data = response.json()
                if isinstance(error_data, dict):
                    message = error_data.get("error") or error_data.get("message")
                    if message:
                        return str(message)
                return str(error_data)
            raw_text = getattr(response, "text", "")
            body = raw_text.strip() if isinstance(raw_text, str) else ""
            return body[:200] if body else f"HTTP {response.status_code}"
        except Exception:
            raw_text = getattr(response, "text", "")
            body = raw_text if isinstance(raw_text, str) else ""
            return f"HTTP {response.status_code}: {body[:200]}"

    @staticmethod
    def _time_range(days: int) -> str:
        if days <= 1:
            return "day"
        if days <= 7:
            return "week"
        if days <= 30:
            return "month"
        return "year"

    @classmethod
    def _search_latency_seconds(cls, instance_data: Dict[str, Any]) -> float:
        timing = (instance_data.get("timing") or {}).get("search") or {}
        all_timing = timing.get("all")
        if isinstance(all_timing, dict):
            for key in ("mean", "median"):
                value = all_timing.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return float("inf")

    @classmethod
    def _extract_public_instances(cls, payload: Any) -> List[str]:
        if not isinstance(payload, dict):
            return []

        instances = payload.get("instances")
        if not isinstance(instances, dict):
            return []

        ranked: List[Tuple[float, float, str]] = []
        for raw_url, item in instances.items():
            if not isinstance(raw_url, str) or not isinstance(item, dict):
                continue
            if item.get("network_type") != "normal":
                continue
            http_status = (item.get("http") or {}).get("status_code")
            if http_status != 200:
                continue
            timing = (item.get("timing") or {}).get("search") or {}
            uptime = timing.get("success_percentage")
            if not isinstance(uptime, (int, float)) or float(uptime) <= 0:
                continue

            ranked.append(
                (
                    float(uptime),
                    cls._search_latency_seconds(item),
                    raw_url.rstrip("/"),
                )
            )

        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [url for _, _, url in ranked[: cls.PUBLIC_INSTANCES_POOL_LIMIT]]

    @classmethod
    def _get_public_instances(cls) -> List[str]:
        now = time.time()
        with cls._public_instances_lock:
            stale_urls: List[str] = []
            if cls._public_instances_cache is None and cls._public_instances_stale_retry_after > now:
                logger.debug(
                    "[SearXNG] 公共实例冷启动刷新退避中，剩余 %.0fs",
                    cls._public_instances_stale_retry_after - now,
                )
                return []
            if cls._public_instances_cache is not None:
                cached_at, cached_urls = cls._public_instances_cache
                if now - cached_at < cls.PUBLIC_INSTANCES_CACHE_TTL_SECONDS:
                    return list(cached_urls)
                stale_urls = list(cached_urls)
                if cls._public_instances_stale_retry_after > now:
                    logger.debug(
                        "[SearXNG] 公共实例刷新退避中，继续使用过期缓存，剩余 %.0fs",
                        cls._public_instances_stale_retry_after - now,
                    )
                    return stale_urls

            try:
                response = requests.get(
                    cls.PUBLIC_INSTANCES_URL,
                    timeout=cls.PUBLIC_INSTANCES_TIMEOUT_SECONDS,
                )
                if response.status_code != 200:
                    logger.warning(
                        "[SearXNG] 拉取公共实例列表失败: HTTP %s",
                        response.status_code,
                    )
                else:
                    urls = cls._extract_public_instances(response.json())
                    if urls:
                        cls._public_instances_cache = (now, list(urls))
                        cls._public_instances_stale_retry_after = 0.0
                        logger.info("[SearXNG] 已刷新公共实例池，共 %s 个候选实例", len(urls))
                        return list(urls)
                    logger.warning("[SearXNG] searx.space 未返回可用公共实例，保留已有缓存")
            except Exception as exc:
                logger.warning("[SearXNG] 拉取公共实例列表失败: %s", exc)

            if stale_urls:
                cls._public_instances_stale_retry_after = (
                    now + cls.PUBLIC_INSTANCES_STALE_REFRESH_BACKOFF_SECONDS
                )
                logger.warning(
                    "[SearXNG] 公共实例刷新失败，继续使用过期缓存，共 %s 个候选实例；"
                    "%.0fs 内不再刷新",
                    len(stale_urls),
                    cls.PUBLIC_INSTANCES_STALE_REFRESH_BACKOFF_SECONDS,
                )
                return stale_urls
            cls._public_instances_stale_retry_after = (
                now + cls.PUBLIC_INSTANCES_STALE_REFRESH_BACKOFF_SECONDS
            )
            logger.warning(
                "[SearXNG] 公共实例冷启动刷新失败，%.0fs 内不再刷新",
                cls.PUBLIC_INSTANCES_STALE_REFRESH_BACKOFF_SECONDS,
            )
            return []

    def _rotate_candidates(self, pool: List[str], *, max_attempts: int) -> List[str]:
        if not pool or max_attempts <= 0:
            return []
        with self._cursor_lock:
            start = self._cursor % len(pool)
            self._cursor = (self._cursor + 1) % len(pool)
        ordered = pool[start:] + pool[:start]
        return ordered[:max_attempts]

    def _do_search(  # type: ignore[override]
        self,
        query: str,
        base_url: str,
        max_results: int,
        days: int = 7,
        *,
        timeout: int,
        retry_enabled: bool,
    ) -> SearchResponse:
        """Execute one SearXNG search against a specific instance."""
        try:
            base = base_url.rstrip("/")
            search_url = base if base.endswith("/search") else base + "/search"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            params = {
                "q": query,
                "format": "json",
                "time_range": self._time_range(days),
                "pageno": 1,
            }

            request_get = _get_with_retry if retry_enabled else requests.get
            response = request_get(search_url, headers=headers, params=params, timeout=timeout)

            if response.status_code != 200:
                error_msg = self._parse_http_error(response)
                if response.status_code == 403:
                    error_msg = (
                        f"{error_msg}；SearXNG 实例可能未启用 JSON 输出（请检查 settings.yml），"
                        "或实例/代理拒绝了本次访问"
                    )
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg,
                )

            try:
                data = response.json()
            except Exception:
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message="响应JSON解析失败",
                )

            if not isinstance(data, dict):
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message="响应格式无效",
                )

            raw = data.get("results", [])
            if not isinstance(raw, list):
                raw = []

            results = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                url_val = item.get("url")
                if not url_val:
                    continue
                raw_published_date = item.get("publishedDate")

                snippet = (item.get("content") or item.get("description") or "")[:500]
                published_date = None
                if raw_published_date:
                    try:
                        dt = datetime.fromisoformat(raw_published_date.replace("Z", "+00:00"))
                        published_date = dt.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        published_date = raw_published_date

                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        snippet=snippet,
                        url=url_val,
                        source=self._extract_domain(url_val),
                        published_date=published_date,
                    )
                )
                if len(results) >= max_results:
                    break

            return SearchResponse(query=query, results=results, provider=self.name, success=True)

        except requests.exceptions.Timeout:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="请求超时",
            )
        except requests.exceptions.RequestException as e:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=f"网络请求失败: {e}",
            )
        except Exception as e:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=f"未知错误: {e}",
            )

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL as source label."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain or "未知来源"
        except Exception:
            return "未知来源"

    def search(self, query: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        """Execute SearXNG search with instance rotation and per-request failover."""
        start_time = time.time()
        if self._base_urls:
            candidates = self._rotate_candidates(
                self._base_urls,
                max_attempts=len(self._base_urls),
            )
            retry_enabled = True
            timeout = self.SELF_HOSTED_TIMEOUT_SECONDS
            empty_error = "SearXNG 未配置可用实例"
        elif self._use_public_instances:
            public_instances = self._get_public_instances()
            candidates = self._rotate_candidates(
                public_instances,
                max_attempts=min(len(public_instances), self.PUBLIC_INSTANCES_MAX_ATTEMPTS),
            )
            retry_enabled = False
            timeout = self.PUBLIC_INSTANCES_TIMEOUT_SECONDS
            empty_error = "未获取到可用的公共 SearXNG 实例"
        else:
            candidates = []
            retry_enabled = False
            timeout = self.PUBLIC_INSTANCES_TIMEOUT_SECONDS
            empty_error = "SearXNG 未配置可用实例"

        if not candidates:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=empty_error,
                search_time=time.time() - start_time,
            )

        errors: List[str] = []
        for base_url in candidates:
            response = self._do_search(
                query,
                base_url,
                max_results,
                days=days,
                timeout=timeout,
                retry_enabled=retry_enabled,
            )
            response.search_time = time.time() - start_time
            if response.success:
                logger.info(
                    "[%s] 搜索 '%s' 成功，实例=%s，返回 %s 条结果，耗时 %.2fs",
                    self.name,
                    query,
                    base_url,
                    len(response.results),
                    response.search_time,
                )
                return response

            errors.append(f"{base_url}: {response.error_message or '未知错误'}")
            logger.warning("[%s] 实例 %s 搜索失败: %s", self.name, base_url, response.error_message)

        elapsed = time.time() - start_time
        return SearchResponse(
            query=query,
            results=[],
            provider=self.name,
            success=False,
            error_message="；".join(errors[:3]) if errors else empty_error,
            search_time=elapsed,
        )


class AnspireSearchProvider(BaseSearchProvider):
    """
    Anspire Search 搜索引擎

    特点：
    - 面向AI生态的下一代实时智能搜索引擎
    - 结果精准、响应快速
    - 适用于股票新闻和市场情报搜索

    文档: https://open.anspire.cn/document/docs/searchApi/
    """

    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Anspire")

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        """执行 Anspire 搜索"""
        try:
            # API 端点
            url = "https://plugin.anspire.cn/api/ntsearch/search"

            # 请求头
            headers = {
                'Authorization': f'Bearer {api_key}'
            }

            # 请求参数
            payload = {
                "query": query,
                "top_k": min(max_results, 50),
                "FromTime": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S"),
                "ToTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            response = requests.post(url, headers=headers, json=payload, timeout=15)

            if response.status_code != 200:
                error_message = ""
                try:
                    error_data = response.json()
                    error_message = error_data.get('msg', '') or str(error_data)
                except Exception:
                    error_message = response.text[:200]

                if response.status_code == 401:
                    error_msg = f"API KEY 无效：{error_message}"
                elif response.status_code == 400:
                    error_msg = f"请求参数错误：{error_message}"
                else:
                    error_msg = f"HTTP {response.status_code}: {error_message}"

                logger.warning(f"[Anspire] 搜索失败：{error_msg}")

                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )

            # 解析响应
            try:
                data = response.json()
            except ValueError as e:
                error_msg = f"响应 JSON 解析失败：{str(e)}"
                logger.error(f"[Anspire] {error_msg}")
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )

            if 'code' in data and data.get('code') != 200:
                error_msg = data.get('msg') or f"API 返回错误码：{data.get('code')}"
                logger.warning(f"[Anspire] 搜索失败：{error_msg}")
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )

            if 'results' not in data:
                error_msg = "响应中缺少 results 字段"
                logger.error(f"[Anspire] {error_msg}，原始响应：{data}")
                return SearchResponse(
                    query=query,
                    results=[],
                    provider=self.name,
                    success=False,
                    error_message=error_msg
                )

            logger.info(f"[Anspire] 搜索完成，query='{query}'")
            logger.debug(f"[Anspire] 原始响应：{data}")

            results = []
            value_list = data.get('results', [])

            for item in value_list[:max_results]:
                snippet = item.get('content')
                if snippet and isinstance(snippet, str) and len(snippet) > 500:
                    snippet = snippet[:500] + "..."

                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=snippet,
                    url=item.get('url', ''),
                    source=self._extract_domain(item.get('url', '')),
                    published_date=item.get('date', '')
                ))

            logger.info(f"[Anspire] 成功解析 {len(results)} 条结果")

            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
            )

        except requests.exceptions.Timeout:
            error_msg = "请求超时"
            logger.error(f"[Anspire] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求失败：{str(e)}"
            logger.error(f"[Anspire] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"未知错误：{str(e)}"
            logger.error(f"[Anspire] {error_msg}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=error_msg
            )

    @staticmethod
    def _extract_domain(url: str) -> str:
        """从 URL 提取域名作为来源"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            return domain or '未知来源'
        except Exception:
            return '未知来源'


class XAIXSearchProvider(BaseSearchProvider):
    """xAI X Search provider for social-signal enrichment on US stocks."""

    API_ENDPOINT = "https://api.x.ai/v1/responses"

    def __init__(self, api_keys: List[str], model: str = "grok-4-1-fast-reasoning"):
        super().__init__(api_keys, "xAI X Search")
        self._model = (model or "grok-4-1-fast-reasoning").strip()

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        lookback_days = max(1, min(int(days or 7), 30))
        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=lookback_days - 1)

        payload = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": self._build_prompt(query, max_results=max_results, days=lookback_days),
                }
            ],
            "tools": [
                {
                    "type": "x_search",
                    "from_date": start_date.isoformat(),
                    "to_date": end_date.isoformat(),
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = _post_with_retry(self.API_ENDPOINT, headers=headers, json=payload, timeout=20)
        except requests.exceptions.Timeout:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="xAI X Search 请求超时",
            )
        except requests.exceptions.RequestException as e:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=f"xAI X Search 网络请求失败: {e}",
            )

        if response.status_code != 200:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=self._parse_http_error(response),
            )

        try:
            data = response.json()
        except ValueError as e:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=f"xAI X Search 响应JSON解析失败: {e}",
            )

        results, summary_text = self._extract_results(data, max_results=max_results)
        if not results:
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message="xAI X Search 未返回可解析的社交信号",
                metadata={"summary": summary_text},
            )

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            success=True,
            metadata={
                "summary": summary_text,
                "model": self._model,
                "citations": data.get("citations", []),
            },
        )

    @staticmethod
    def _build_prompt(query: str, *, max_results: int, days: int) -> str:
        return (
            f"Search X for the last {days} days about {query}. "
            f"Return up to {max_results} high-signal items as a numbered list. "
            "Each item must be one single line in this format: "
            "Short title — concise summary. "
            "Focus on earnings, guidance, official company posts, executive comments, "
            "product issues, lawsuits, short reports, analyst or market-moving posts. "
            "Avoid generic hype, jokes, memes, and duplicated points."
        )

    @staticmethod
    def _parse_http_error(response) -> str:
        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                error_data = response.json()
                if isinstance(error_data, dict):
                    error = error_data.get("error")
                    if isinstance(error, dict):
                        message = error.get("message") or error.get("code")
                        if message:
                            return f"HTTP {response.status_code}: {message}"
                    message = error_data.get("message")
                    if message:
                        return f"HTTP {response.status_code}: {message}"
                return f"HTTP {response.status_code}: {error_data}"
            return f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception:
            return f"HTTP {response.status_code}: {response.text[:200]}"

    def _extract_results(self, data: Dict[str, Any], *, max_results: int) -> Tuple[List[SearchResult], str]:
        citations = data.get("citations", [])
        output = data.get("output", [])
        seen_urls = set()
        results: List[SearchResult] = []
        collected_text: List[str] = []

        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict) or content.get("type") != "output_text":
                    continue
                text = str(content.get("text") or "").strip()
                if text:
                    collected_text.append(text)
                annotations = content.get("annotations") or []
                if text and annotations:
                    block_results = self._extract_results_from_output_block(
                        text,
                        annotations,
                        seen_urls=seen_urls,
                    )
                    results.extend(block_results)
                    if len(results) >= max_results:
                        return results[:max_results], "\n".join(collected_text).strip()

        summary_text = "\n".join(collected_text).strip()
        if len(results) < max_results and citations:
            results.extend(
                self._fallback_results_from_citations(
                    citations,
                    summary_text,
                    max_results=max_results,
                    seen_urls=seen_urls,
                )
            )

        return results[:max_results], summary_text

    def _extract_results_from_output_block(
        self,
        text: str,
        annotations: List[Dict[str, Any]],
        *,
        seen_urls: set,
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        sorted_annotations = sorted(
            [ann for ann in annotations if isinstance(ann, dict) and ann.get("url")],
            key=lambda ann: int(ann.get("start_index", 0)),
        )

        for ann in sorted_annotations:
            url = str(ann.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            line = self._extract_line(text, int(ann.get("start_index", 0)), int(ann.get("end_index", 0)))
            clean_line = self._strip_inline_citations(line)
            title, snippet = self._split_title_and_snippet(clean_line, url)
            seen_urls.add(url)
            results.append(
                SearchResult(
                    title=title,
                    snippet=snippet,
                    url=url,
                    source=self._extract_domain(url),
                )
            )

        return results

    def _fallback_results_from_citations(
        self,
        citations: List[Any],
        summary_text: str,
        *,
        max_results: int,
        seen_urls: set,
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        clean_lines = [
            self._strip_inline_citations(line).strip()
            for line in summary_text.splitlines()
            if self._strip_inline_citations(line).strip()
        ]
        candidate_lines = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in clean_lines
            if line.strip()
        ]

        for idx, raw_url in enumerate(citations):
            url = str(raw_url or "").strip()
            if not url or url in seen_urls:
                continue
            line = candidate_lines[min(idx, len(candidate_lines) - 1)] if candidate_lines else "X 社交信号摘要"
            title, snippet = self._split_title_and_snippet(line, url)
            seen_urls.add(url)
            results.append(
                SearchResult(
                    title=title,
                    snippet=snippet,
                    url=url,
                    source=self._extract_domain(url),
                )
            )
            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def _extract_line(text: str, start_index: int, end_index: int) -> str:
        line_start = text.rfind("\n", 0, max(start_index, 0))
        if line_start < 0:
            line_start = 0
        else:
            line_start += 1
        line_end = text.find("\n", max(end_index, 0))
        if line_end < 0:
            line_end = len(text)
        return text[line_start:line_end].strip()

    @staticmethod
    def _strip_inline_citations(text: str) -> str:
        return re.sub(r"\[\[\d+\]\]\([^)]+\)", "", text or "").strip()

    def _split_title_and_snippet(self, line: str, url: str) -> Tuple[str, str]:
        normalized = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line or "").strip(" \t-–—:")
        for separator in (" — ", " – ", " - ", ": "):
            if separator in normalized:
                left, right = normalized.split(separator, 1)
                title = (left or "").strip()[:120]
                snippet = (right or "").strip()[:500]
                if title and snippet:
                    return title, snippet

        fallback_title = normalized[:120] if normalized else self._extract_domain(url)
        fallback_snippet = normalized[:500] if normalized else "X 社交信号摘要"
        return fallback_title, fallback_snippet

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain or "x.com"
        except Exception:
            return "x.com"


class SearchService:
    """
    搜索服务
    
    功能：
    1. 管理多个搜索引擎
    2. 自动故障转移
    3. 结果聚合和格式化
    4. 数据源失败时的增强搜索（股价、走势等）
    5. 港股/美股自动使用英文搜索关键词
    """
    
    # 增强搜索关键词模板（A股 中文）
    ENHANCED_SEARCH_KEYWORDS = [
        "{name} 股票 今日 股价",
        "{name} {code} 最新 行情 走势",
        "{name} 股票 分析 走势图",
        "{name} K线 技术分析",
        "{name} {code} 涨跌 成交量",
    ]

    # 增强搜索关键词模板（港股/美股 英文）
    ENHANCED_SEARCH_KEYWORDS_EN = [
        "{name} stock price today",
        "{name} {code} latest quote trend",
        "{name} stock analysis chart",
        "{name} technical analysis",
        "{name} {code} performance volume",
    ]
    NEWS_OVERSAMPLE_FACTOR = 2
    NEWS_OVERSAMPLE_MAX = 10
    FUTURE_TOLERANCE_DAYS = 1
    
    def __init__(
        self,
        bocha_keys: Optional[List[str]] = None,
        tavily_keys: Optional[List[str]] = None,
        anspire_keys: Optional[List[str]] = None,
        brave_keys: Optional[List[str]] = None,
        serpapi_keys: Optional[List[str]] = None,
        minimax_keys: Optional[List[str]] = None,
        xai_keys: Optional[List[str]] = None,
        xai_search_model: str = "grok-4-1-fast-reasoning",
        searxng_base_urls: Optional[List[str]] = None,
        searxng_public_instances_enabled: bool = True,
        news_max_age_days: int = 3,
        news_strategy_profile: str = "short",
    ):
        """
        初始化搜索服务

        Args:
            bocha_keys: 博查搜索 API Key 列表
            tavily_keys: Tavily API Key 列表
            anspire_keys: Anspire Search API Key 列表（中文搜索优化）
            brave_keys: Brave Search API Key 列表
            serpapi_keys: SerpAPI Key 列表
            minimax_keys: MiniMax API Key 列表
            xai_keys: xAI API Key 列表（用于 X 社交信号）
            xai_search_model: xAI X Search 使用的模型
            searxng_base_urls: SearXNG 实例地址列表（自建无配额兜底）
            searxng_public_instances_enabled: 未配置自建实例时，是否自动使用公共 SearXNG 实例
            news_max_age_days: 新闻最大时效（天）
            news_strategy_profile: 新闻窗口策略档位（ultra_short/short/medium/long）
        """
        self._providers: List[BaseSearchProvider] = []
        self._x_signal_provider: Optional[XAIXSearchProvider] = None
        self.news_max_age_days = max(1, news_max_age_days)
        raw_profile = (news_strategy_profile or "short").strip().lower()
        self.news_strategy_profile = normalize_news_strategy_profile(news_strategy_profile)
        if raw_profile != self.news_strategy_profile:
            logger.warning(
                "NEWS_STRATEGY_PROFILE '%s' 无效，已回退为 'short'",
                news_strategy_profile,
            )
        self.news_window_days = resolve_news_window_days(
            news_max_age_days=self.news_max_age_days,
            news_strategy_profile=self.news_strategy_profile,
        )
        self.news_profile_days = NEWS_STRATEGY_WINDOWS.get(
            self.news_strategy_profile,
            NEWS_STRATEGY_WINDOWS["short"],
        )

        # 初始化搜索引擎（按优先级排序）
        # 1. Bocha 优先（中文搜索优化，AI摘要）
        if bocha_keys:
            self._providers.append(BochaSearchProvider(bocha_keys))
            logger.info(f"已配置 Bocha 搜索，共 {len(bocha_keys)} 个 API Key")

        # 2. Tavily（免费额度更多，每月 1000 次）
        if tavily_keys:
            self._providers.append(TavilySearchProvider(tavily_keys))
            logger.info(f"已配置 Tavily 搜索，共 {len(tavily_keys)} 个 API Key")

        # 3. Brave Search（隐私优先，全球覆盖）
        if brave_keys:
            self._providers.append(BraveSearchProvider(brave_keys))
            logger.info(f"已配置 Brave 搜索，共 {len(brave_keys)} 个 API Key")

        # 4. SerpAPI 作为备选（每月 100 次）
        if serpapi_keys:
            self._providers.append(SerpAPISearchProvider(serpapi_keys))
            logger.info(f"已配置 SerpAPI 搜索，共 {len(serpapi_keys)} 个 API Key")

        # 5. MiniMax（Coding Plan Web Search，结构化结果）
        if minimax_keys:
            self._providers.append(MiniMaxSearchProvider(minimax_keys))
            logger.info(f"已配置 MiniMax 搜索，共 {len(minimax_keys)} 个 API Key")

        # 6. SearXNG（自建实例优先；未配置其他引擎时可自动发现公共实例兜底）
        use_public = bool(
            searxng_public_instances_enabled
            and not searxng_base_urls
            and not self._providers
        )
        searxng_provider = SearXNGSearchProvider(
            searxng_base_urls,
            use_public_instances=use_public,
        )
        if searxng_provider.is_available:
            self._providers.append(searxng_provider)
            if searxng_base_urls:
                logger.info("已配置 SearXNG 搜索，共 %s 个自建实例", len(searxng_base_urls))
            else:
                logger.info("已启用 SearXNG 公共实例自动发现模式")

        # 7. Anspire Search（实时智能搜索优化）
        if anspire_keys:
            self._providers.insert(0, AnspireSearchProvider(anspire_keys))
            logger.info(f"已配置 Anspire Search 搜索，共 {len(anspire_keys)} 个 API Key")

        if not self._providers:
            logger.warning("未配置任何搜索能力，新闻搜索功能将不可用")

        if xai_keys:
            self._x_signal_provider = XAIXSearchProvider(xai_keys, model=xai_search_model)
            logger.info(f"已配置 xAI X Search，共 {len(xai_keys)} 个 API Key")

        # In-memory search result cache: {cache_key: (timestamp, SearchResponse)}
        self._cache: Dict[str, Tuple[float, 'SearchResponse']] = {}
        # Default cache TTL in seconds (10 minutes)
        self._cache_ttl: int = 600
        self._us_sec_adapter = UsSecFundamentalAdapter()
        logger.info(
            "新闻时效策略已启用: profile=%s, profile_days=%s, NEWS_MAX_AGE_DAYS=%s, effective_window=%s",
            self.news_strategy_profile,
            self.news_profile_days,
            self.news_max_age_days,
            self.news_window_days,
        )
    
    @staticmethod
    def _is_foreign_stock(stock_code: str) -> bool:
        """判断是否为港股或美股"""
        import re
        code = stock_code.strip()
        # 美股：1-5个大写字母，可能包含点（如 BRK.B）
        if re.match(r'^[A-Za-z]{1,5}(\.[A-Za-z])?$', code):
            return True
        # 港股：带 hk 前缀或 5位纯数字
        lower = code.lower()
        if lower.startswith('hk'):
            return True
        if code.isdigit() and len(code) == 5:
            return True
        return False

    @staticmethod
    def _market_tag(stock_code: str) -> str:
        """返回市场标签: cn/us/hk。"""
        market = get_market_for_stock(stock_code)
        return market or "cn"

    # A-share ETF code prefixes (Shanghai 51/52/56/58, Shenzhen 15/16/18)
    _A_ETF_PREFIXES = ('51', '52', '56', '58', '15', '16', '18')
    _ETF_NAME_KEYWORDS = ('ETF', 'FUND', 'TRUST', 'INDEX', 'TRACKER', 'UNIT')  # US/HK ETF name hints

    @staticmethod
    def is_index_or_etf(stock_code: str, stock_name: str) -> bool:
        """
        Judge if symbol is index-tracking ETF or market index.
        For such symbols, analysis focuses on index movement only, not issuer company risks.
        """
        code = (stock_code or '').strip().split('.')[0]
        if not code:
            return False
        # A-share ETF
        if code.isdigit() and len(code) == 6 and code.startswith(SearchService._A_ETF_PREFIXES):
            return True
        # US index (SPX, DJI, IXIC etc.)
        if is_us_index_code(code):
            return True
        # US/HK ETF: foreign symbol + name contains fund-like keywords
        if SearchService._is_foreign_stock(code):
            name_upper = (stock_name or '').upper()
            return any(kw in name_upper for kw in SearchService._ETF_NAME_KEYWORDS)
        return False

    _US_CHINA_EXPOSURE_KEYWORDS = {
        "revenue": (
            "greater china",
            "china revenue",
            "china sales",
            "china demand",
            "mainland china",
            "prc",
            "中国市场",
            "中国收入",
            "大中华",
        ),
        "supply_chain": (
            "supply chain",
            "supplier",
            "manufactur",
            "assembly",
            "factory",
            "made in china",
            "中国供应链",
            "中国制造",
            "组装",
        ),
        "policy": (
            "tariff",
            "export control",
            "sanction",
            "geopolitical",
            "rare earth",
            "china policy",
            "中美",
            "关税",
            "出口管制",
            "稀土",
            "中国政策",
        ),
    }
    _US_CHINA_SIGNAL_LABELS = {
        "revenue": "中国收入/需求",
        "supply_chain": "中国供应链/制造",
        "policy": "中国政策/关税/出口管制",
    }

    @staticmethod
    def _market_label(market: str) -> str:
        return {
            "cn": "A股",
            "hk": "港股",
            "us": "美股",
        }.get(market, market or "未知市场")

    def _summarize_us_china_exposure(
        self,
        stock_name: str,
        intel_results: Optional[Dict[str, SearchResponse]],
        is_index_etf: bool,
    ) -> Dict[str, Any]:
        """Heuristically assess whether China policy should materially affect a US stock."""
        if is_index_etf:
            return {
                "level": "holdings_based",
                "level_label": "看持仓结构",
                "policy_weight": "holdings_based",
                "policy_weight_label": "按持仓结构决定",
                "signals": [],
                "reasoning": (
                    f"{stock_name} 属于指数/ETF，需先看成分股和行业权重；只有当重仓行业或核心持仓对中国"
                    "收入、供应链或政策高度敏感时，才提高中国政策权重。"
                ),
            }

        results = intel_results or {}
        direct_response = results.get("china_exposure")
        direct_summary = (
            direct_response.metadata.get("china_exposure", {})
            if isinstance(direct_response, SearchResponse)
            else {}
        )
        if isinstance(direct_summary, dict) and direct_summary:
            level = str(direct_summary.get("level") or "unknown")
            level_label = {
                "high": "高",
                "medium": "中",
                "low": "低",
                "unknown": "未知",
            }.get(level, "未知")
            policy_weight = {
                "high": "high",
                "medium": "medium",
                "low": "low",
            }.get(level, "guarded")
            policy_weight_label = {
                "high": "高权重",
                "medium": "中等权重",
                "low": "低权重",
                "guarded": "谨慎使用",
            }[policy_weight if policy_weight in {"high", "medium", "low"} else "guarded"]
            signals = list(direct_summary.get("signals", []))
            if signals:
                reasoning = (
                    f"基于 SEC {direct_summary.get('filing_form') or 'filing'} 原文检索到 "
                    f"{'、'.join(signals)} 证据，中国政策影响权重判定为{level_label}。"
                )
            else:
                reasoning = (
                    f"已检查 SEC {direct_summary.get('filing_form') or 'filing'} 原文，未检索到明确的中国收入、"
                    "供应链或政策传导证据；默认不应把中国政策当成核心驱动。"
                )
            return {
                "level": level,
                "level_label": level_label,
                "policy_weight": policy_weight,
                "policy_weight_label": policy_weight_label,
                "signals": signals,
                "evidence": list(direct_summary.get("evidence", []))[:4],
                "reasoning": reasoning,
                "filing_url": direct_summary.get("filing_url", ""),
            }

        matched_groups = set()
        evidence: List[str] = []

        for dim_name in ("china_exposure", "official_filings", "risk_check", "industry", "latest_news"):
            response = results.get(dim_name)
            if not response or not response.success:
                continue
            for item in response.results[:3]:
                text = f"{item.title} {item.snippet}".lower()
                local_hits = []
                for group, keywords in self._US_CHINA_EXPOSURE_KEYWORDS.items():
                    if any(keyword in text for keyword in keywords):
                        matched_groups.add(group)
                        local_hits.append(group)
                if local_hits:
                    signals = "、".join(self._US_CHINA_SIGNAL_LABELS[group] for group in sorted(set(local_hits)))
                    evidence.append(f"{item.source}:{signals}")

        signal_labels = [self._US_CHINA_SIGNAL_LABELS[group] for group in sorted(matched_groups)]
        signals_text = "、".join(signal_labels)

        if "revenue" in matched_groups and ("supply_chain" in matched_groups or "policy" in matched_groups):
            level = "high"
            level_label = "高"
            policy_weight = "high"
            policy_weight_label = "高权重"
            reasoning = (
                f"检索结果同时出现 {signals_text} 证据，说明该美股与中国业务/供应链/政策传导关系较强，"
                "分析时应把中国政策视作高权重外部变量之一，但仍需与 SEC 披露和财报指引交叉验证。"
            )
        elif len(matched_groups) >= 2:
            level = "medium"
            level_label = "中"
            policy_weight = "medium"
            policy_weight_label = "中等权重"
            reasoning = (
                f"检索结果出现 {signals_text} 线索，说明该美股对中国存在一定经营或政策暴露，"
                "中国政策可作为中等权重因子，不应脱离财报与估值单独下结论。"
            )
        elif len(matched_groups) == 1:
            level = "low"
            level_label = "低"
            policy_weight = "low"
            policy_weight_label = "低权重"
            reasoning = (
                f"当前只看到 {signals_text} 的零散线索，中国政策只应作为低权重辅助因子，"
                "除非后续公告或财报进一步确认，否则不要让其主导结论。"
            )
        else:
            level = "unknown"
            level_label = "未知"
            policy_weight = "guarded"
            policy_weight_label = "谨慎使用"
            reasoning = (
                "当前检索结果没有足够证据证明该美股对中国收入、供应链或政策高度敏感，"
                "默认不应把中国政策当成核心驱动，只能作为待验证背景变量。"
            )

        return {
            "level": level,
            "level_label": level_label,
            "policy_weight": policy_weight,
            "policy_weight_label": policy_weight_label,
            "signals": signal_labels,
            "evidence": evidence[:4],
            "reasoning": reasoning,
        }

    def build_market_intel_summary(
        self,
        stock_code: str,
        stock_name: str,
        intel_results: Optional[Dict[str, SearchResponse]] = None,
    ) -> Dict[str, Any]:
        """Return market-specific guidance so downstream prompts can apply the right logic."""
        market = self._market_tag(stock_code)
        is_index_etf = self.is_index_or_etf(stock_code, stock_name)
        summary: Dict[str, Any] = {
            "market": market,
            "market_label": self._market_label(market),
            "is_index_etf": is_index_etf,
        }

        if market == "us":
            summary.update({
                "official_source_priority": "SEC 直连披露、财报电话会、公司指引、美国市场定位数据",
                "analysis_focus": (
                    "先看 SEC/财报/指引/诉讼与估值，再决定政策变量权重；不要默认把中国政策当成主线。"
                ),
                "policy_scope": (
                    "中国政策只有在存在明确的中国收入、供应链、制造、关税或出口管制暴露时，"
                    "才应上调为重要因子。"
                ),
                "china_exposure": self._summarize_us_china_exposure(stock_name, intel_results, is_index_etf),
            })
            return summary

        if market == "hk":
            summary.update({
                "official_source_priority": "HKEX 官方事件页/公告、业绩公告、配股/回购/分红安排、港股卖方报告",
                "analysis_focus": "先看 HKEX 披露和盈利/融资安排，再看行业景气、南向资金与内地业务暴露。",
                "policy_scope": (
                    "内地政策通常是港股的重要二级因子；若公司主营、客户或资产明显依赖内地，"
                    "可进一步上调权重，否则不要单靠内地政策归因。"
                ),
            })
            return summary

        summary.update({
            "official_source_priority": "巨潮直连公告、沪深交易所公告、业绩预告、监管函、资金流与板块数据",
            "analysis_focus": "政策、监管、业绩预告与资金流本身就是 A 股核心变量，应直接纳入主判断。",
            "policy_scope": "中国政策与产业监管属于核心主因，不需要额外做 China exposure 闸门判断。",
        })
        return summary

    @staticmethod
    def _build_failed_response(
        query: str,
        provider: str,
        error_message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        return SearchResponse(
            query=query,
            results=[],
            provider=provider,
            success=False,
            error_message=error_message,
            metadata=metadata or {},
        )

    @staticmethod
    def _strip_markup(text: str) -> str:
        clean = re.sub(r"(?s)<[^>]+>", " ", text or "")
        return re.sub(r"\s+", " ", clean).strip()

    @staticmethod
    def _format_timestamp_ms(timestamp_ms: Any) -> Optional[str]:
        try:
            return datetime.fromtimestamp(float(timestamp_ms) / 1000.0).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    @staticmethod
    def _hk_row_matches_code(row_text: str, target_code: str) -> bool:
        target_digits = re.sub(r"\D", "", target_code or "")
        if not target_digits:
            return False
        try:
            normalized_target = str(int(target_digits))
        except ValueError:
            normalized_target = target_digits.lstrip("0") or "0"

        candidates = set()
        for token in re.findall(r"\b\d{1,5}\b", row_text or ""):
            try:
                candidates.add(str(int(token)))
            except ValueError:
                continue
        return normalized_target in candidates

    def can_run_comprehensive_intel(self, stock_code: str = "") -> bool:
        """Whether comprehensive intel can run via direct official feeds or search engines."""
        market = self._market_tag(stock_code) if stock_code else "cn"
        if market in {"cn", "hk", "us"}:
            return True
        return self.is_available

    def _run_provider_search(
        self,
        query: str,
        provider_index: int,
        *,
        max_results: int = 3,
    ) -> Tuple[SearchResponse, int]:
        available_providers = [p for p in self._providers if p.is_available]
        if not available_providers:
            return self._build_failed_response(query, "None", "未配置搜索引擎 API Key"), provider_index

        provider = available_providers[provider_index % len(available_providers)]
        provider_index += 1
        logger.info(f"[情报搜索] 使用 {provider.name}: {query}")
        response = provider.search(query, max_results=max_results, days=self.news_max_age_days)
        return response, provider_index

    def _recent_day_range(self, lookback_days: int = 180) -> Tuple[str, str]:
        end = datetime.now()
        start = end - timedelta(days=max(lookback_days, self.news_max_age_days))
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _direct_cninfo_announcements(
        self,
        stock_code: str,
        stock_name: str,
        *,
        query: str,
        limit: int = 4,
    ) -> SearchResponse:
        normalized_code = re.sub(r"\D", "", stock_code or "")
        searchkey = stock_name or normalized_code
        if not searchkey:
            return self._build_failed_response(query, "CNINFO", "missing stock identifier")

        start_date, end_date = self._recent_day_range(lookback_days=180)
        payload = {
            "pageNum": 1,
            "pageSize": max(10, limit * 3),
            "tabName": "fulltext",
            "plate": "",
            "searchkey": searchkey,
            "secid": "",
            "stock": "",
            "category": "",
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
        }
        try:
            response = requests.post(
                "https://www.cninfo.com.cn/new/hisAnnouncement/query",
                data=payload,
                headers=headers,
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return self._build_failed_response(query, "CNINFO", f"cninfo:{type(exc).__name__}")

        announcements = data.get("announcements") or []
        if normalized_code:
            matched = [item for item in announcements if str(item.get("secCode") or "").zfill(6) == normalized_code.zfill(6)]
            if matched:
                announcements = matched

        results: List[SearchResult] = []
        for item in announcements[:limit]:
            title = self._strip_markup(item.get("announcementTitle") or item.get("shortTitle") or "")
            sec_name = self._strip_markup(item.get("secName") or item.get("tileSecName") or stock_name)
            file_path = str(item.get("adjunctUrl") or "").lstrip("/")
            url = f"https://static.cninfo.com.cn/{file_path}" if file_path else "https://www.cninfo.com.cn/"
            snippet = f"{sec_name} 公告，类型={item.get('announcementType') or '未分类'}。"
            results.append(
                SearchResult(
                    title=title or f"{sec_name} 公告",
                    snippet=snippet,
                    url=url,
                    source="cninfo.com.cn",
                    published_date=self._format_timestamp_ms(item.get("announcementTime")),
                )
            )

        if not results:
            return self._build_failed_response(query, "CNINFO", "未找到匹配公告")

        return SearchResponse(
            query=query,
            results=results,
            provider="CNINFO",
            success=True,
            metadata={"official_source": "cninfo"},
        )

    def _build_event_response_from_results(
        self,
        query: str,
        provider: str,
        source_results: List[SearchResult],
        keywords: Tuple[str, ...],
        *,
        fallback_message: str,
    ) -> SearchResponse:
        lower_keywords = tuple(keyword.lower() for keyword in keywords)
        matched = [
            item for item in source_results
            if any(keyword in f"{item.title} {item.snippet}".lower() for keyword in lower_keywords)
        ]
        if not matched:
            return self._build_failed_response(query, provider, fallback_message)
        return SearchResponse(
            query=query,
            results=matched[:4],
            provider=provider,
            success=True,
            metadata={"official_source": provider.lower()},
        )

    def _direct_hk_event_calendar(
        self,
        stock_code: str,
        stock_name: str,
        *,
        query: str,
        limit: int = 4,
    ) -> SearchResponse:
        code_digits = re.sub(r"\D", "", stock_code or "").zfill(5)
        if not code_digits:
            return self._build_failed_response(query, "HKEX", "missing hk stock code")

        endpoints = [
            (
                "https://www3.hkexnews.hk/reports/bmn/ebmn.htm",
                ("board meeting", "results", "profit warning", "annual", "interim"),
                "board meeting notification",
            ),
            (
                "https://www3.hkexnews.hk/reports/doe/eent.htm",
                ("dividend", "entitlement", "ex-date", "book close"),
                "dividend / entitlement schedule",
            ),
        ]
        collected: List[SearchResult] = []
        for url, keywords, label in endpoints:
            try:
                tables = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                tables.raise_for_status()
                dfs = pd.read_html(StringIO(tables.text))
            except Exception as exc:
                logger.debug(f"[HKEX] 官方事件页面读取失败 {url}: {exc}")
                continue
            if len(dfs) < 2:
                continue
            df = dfs[1].copy()
            df.columns = [str(col).strip() for col in df.columns]
            joined = df.apply(lambda row: " ".join(str(v) for v in row.tolist()), axis=1)
            matched = df[joined.map(lambda value: self._hk_row_matches_code(value, code_digits))]
            for _, row in matched.head(limit).iterrows():
                values = [str(v).strip() for v in row.tolist() if str(v).strip() and str(v).strip() != "nan"]
                if not values:
                    continue
                title = f"{stock_name or stock_code} {label}"
                snippet = " | ".join(values[:5])
                collected.append(
                    SearchResult(
                        title=title,
                        snippet=snippet,
                        url=url,
                        source="hkex.com.hk",
                    )
                )

        if not collected:
            return self._build_failed_response(query, "HKEX", "未找到港股官方事件日历")

        return SearchResponse(
            query=query,
            results=collected[:limit],
            provider="HKEX",
            success=True,
            metadata={"official_source": "hkex_event_pages"},
        )

    def _direct_sec_filings(
        self,
        stock_code: str,
        stock_name: str,
        *,
        query: str,
        limit: int = 4,
    ) -> SearchResponse:
        filings = self._us_sec_adapter.get_recent_filings(
            stock_code,
            forms=["10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "20-F", "20-F/A", "6-K", "4", "13D", "13D/A", "13G", "13G/A"],
            limit=limit,
        )
        if filings.get("status") != "ok":
            error = ";".join(filings.get("errors", [])) or "未找到 SEC 披露"
            return self._build_failed_response(query, "SEC", error)

        results = [
            SearchResult(
                title=f"{stock_name or stock_code} {item.get('form')} filed",
                snippet=f"SEC filing form {item.get('form')} on {item.get('filed')}.",
                url=item.get("url") or "https://www.sec.gov/",
                source="sec.gov",
                published_date=item.get("filed"),
            )
            for item in filings.get("items", [])[:limit]
        ]
        return SearchResponse(
            query=query,
            results=results,
            provider="SEC",
            success=True,
            metadata={"official_source": "sec", "filings": filings.get("items", [])[:limit]},
        )

    def _direct_us_china_exposure(
        self,
        stock_code: str,
        stock_name: str,
        *,
        query: str,
    ) -> SearchResponse:
        summary = self._us_sec_adapter.get_china_exposure_summary(stock_code)
        if summary.get("status") not in {"partial", "ok"}:
            error = ";".join(summary.get("errors", [])) or "no sec exposure evidence"
            return self._build_failed_response(query, "SEC", error, metadata={"china_exposure": summary})

        signals = summary.get("signals", [])
        title = (
            f"{stock_name or stock_code} China exposure ({summary.get('level', 'unknown')})"
        )
        snippet_parts = []
        if signals:
            snippet_parts.append("、".join(signals))
        if summary.get("evidence"):
            snippet_parts.append(summary["evidence"][0])
        snippet = " | ".join(part for part in snippet_parts if part) or "SEC filing text did not show strong China exposure signals."
        results = [
            SearchResult(
                title=title,
                snippet=snippet,
                url=summary.get("filing_url") or "https://www.sec.gov/",
                source="sec.gov",
                published_date=summary.get("filing_date"),
            )
        ]
        return SearchResponse(
            query=query,
            results=results,
            provider="SEC",
            success=True,
            metadata={"china_exposure": summary},
        )

    def _direct_x_social_signal(
        self,
        stock_code: str,
        stock_name: str,
        *,
        query: str,
        limit: int = 4,
    ) -> SearchResponse:
        if not self._x_signal_provider or not self._x_signal_provider.is_available:
            return self._build_failed_response(query, "xAI X Search", "未配置 XAI_API_KEY")

        return self._x_signal_provider.search(
            query=query or f"{stock_name} {stock_code} X social signal",
            max_results=limit,
            days=self.news_max_age_days,
        )

    @property
    def is_available(self) -> bool:
        """检查是否有可用的搜索引擎"""
        return any(p.is_available for p in self._providers)

    def _cache_key(self, query: str, max_results: int, days: int) -> str:
        """Build a cache key from query parameters."""
        return f"{query}|{max_results}|{days}"

    def _get_cached(self, key: str) -> Optional['SearchResponse']:
        """Return cached SearchResponse if still valid, else None."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, response = entry
        if time.time() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        logger.debug(f"Search cache hit: {key[:60]}...")
        return response

    def _put_cache(self, key: str, response: 'SearchResponse') -> None:
        """Store a successful SearchResponse in cache."""
        # Hard cap: evict oldest entries when cache exceeds limit
        _MAX_CACHE_SIZE = 500
        if len(self._cache) >= _MAX_CACHE_SIZE:
            now = time.time()
            # First pass: remove expired entries
            expired = [k for k, (ts, _) in self._cache.items() if now - ts > self._cache_ttl]
            for k in expired:
                del self._cache[k]
            # Second pass: if still over limit, evict oldest entries (FIFO)
            if len(self._cache) >= _MAX_CACHE_SIZE:
                excess = len(self._cache) - _MAX_CACHE_SIZE + 1
                oldest = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])[:excess]
                for k in oldest:
                    del self._cache[k]
        self._cache[key] = (time.time(), response)

    def _effective_news_window_days(self) -> int:
        """Resolve effective news window from strategy profile and global max-age."""
        return resolve_news_window_days(
            news_max_age_days=self.news_max_age_days,
            news_strategy_profile=self.news_strategy_profile,
        )

    @classmethod
    def _provider_request_size(cls, max_results: int) -> int:
        """Apply light overfetch before time filtering to avoid sparse outputs."""
        target = max(1, int(max_results))
        return max(target, min(target * cls.NEWS_OVERSAMPLE_FACTOR, cls.NEWS_OVERSAMPLE_MAX))

    @staticmethod
    def _parse_relative_news_date(text: str, now: datetime) -> Optional[date]:
        """Parse common Chinese/English relative-time strings."""
        raw = (text or "").strip()
        if not raw:
            return None

        lower = raw.lower()
        if raw in {"今天", "今日", "刚刚"} or lower in {"today", "just now", "now"}:
            return now.date()
        if raw == "昨天" or lower == "yesterday":
            return (now - timedelta(days=1)).date()
        if raw == "前天":
            return (now - timedelta(days=2)).date()

        zh = re.match(r"^\s*(\d+)\s*(分钟|小时|天|周|个月|月|年)\s*前\s*$", raw)
        if zh:
            amount = int(zh.group(1))
            unit = zh.group(2)
            if unit == "分钟":
                return (now - timedelta(minutes=amount)).date()
            if unit == "小时":
                return (now - timedelta(hours=amount)).date()
            if unit == "天":
                return (now - timedelta(days=amount)).date()
            if unit == "周":
                return (now - timedelta(weeks=amount)).date()
            if unit in {"个月", "月"}:
                return (now - timedelta(days=amount * 30)).date()
            if unit == "年":
                return (now - timedelta(days=amount * 365)).date()

        en = re.match(
            r"^\s*(\d+)\s*(minute|minutes|min|mins|hour|hours|day|days|week|weeks|month|months|year|years)\s*ago\s*$",
            lower,
        )
        if en:
            amount = int(en.group(1))
            unit = en.group(2)
            if unit in {"minute", "minutes", "min", "mins"}:
                return (now - timedelta(minutes=amount)).date()
            if unit in {"hour", "hours"}:
                return (now - timedelta(hours=amount)).date()
            if unit in {"day", "days"}:
                return (now - timedelta(days=amount)).date()
            if unit in {"week", "weeks"}:
                return (now - timedelta(weeks=amount)).date()
            if unit in {"month", "months"}:
                return (now - timedelta(days=amount * 30)).date()
            if unit in {"year", "years"}:
                return (now - timedelta(days=amount * 365)).date()

        return None

    @classmethod
    def _normalize_news_publish_date(cls, value: Any) -> Optional[date]:
        """Normalize provider date value into a date object."""
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                local_tz = datetime.now().astimezone().tzinfo or timezone.utc
                return value.astimezone(local_tz).date()
            return value.date()
        if isinstance(value, date):
            return value

        text = str(value).strip()
        if not text:
            return None
        now = datetime.now()
        local_tz = now.astimezone().tzinfo or timezone.utc

        relative_date = cls._parse_relative_news_date(text, now)
        if relative_date:
            return relative_date

        # Unix timestamp fallback
        if text.isdigit() and len(text) in (10, 13):
            try:
                ts = int(text[:10]) if len(text) == 13 else int(text)
                # Provider timestamps are typically UTC epoch seconds.
                # Normalize to local date to keep window checks aligned with local "today".
                return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(local_tz).date()
            except (OSError, OverflowError, ValueError):
                pass

        iso_candidate = text.replace("Z", "+00:00")
        try:
            parsed_iso = datetime.fromisoformat(iso_candidate)
            if parsed_iso.tzinfo is not None:
                return parsed_iso.astimezone(local_tz).date()
            return parsed_iso.date()
        except ValueError:
            pass

        normalized = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)

        try:
            parsed_rfc = parsedate_to_datetime(normalized)
            if parsed_rfc:
                if parsed_rfc.tzinfo is not None:
                    return parsed_rfc.astimezone(local_tz).date()
                return parsed_rfc.date()
        except (TypeError, ValueError):
            pass

        zh_match = re.search(r"(\d{4})\s*[年/\-.]\s*(\d{1,2})\s*[月/\-.]\s*(\d{1,2})\s*日?", text)
        if zh_match:
            try:
                return date(int(zh_match.group(1)), int(zh_match.group(2)), int(zh_match.group(3)))
            except ValueError:
                pass

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
            "%Y.%m.%d %H:%M:%S",
            "%Y.%m.%d %H:%M",
            "%Y.%m.%d",
            "%Y%m%d",
            "%b %d, %Y",
            "%B %d, %Y",
            "%d %b %Y",
            "%d %B %Y",
            "%a, %d %b %Y %H:%M:%S %z",
        ):
            try:
                parsed_dt = datetime.strptime(normalized, fmt)
                if parsed_dt.tzinfo is not None:
                    return parsed_dt.astimezone(local_tz).date()
                return parsed_dt.date()
            except ValueError:
                continue

        return None

    def _filter_news_response(
        self,
        response: SearchResponse,
        *,
        search_days: int,
        max_results: int,
        log_scope: str,
    ) -> SearchResponse:
        """Hard-filter results by published_date recency and normalize date strings."""
        if not response.success or not response.results:
            return response

        today = datetime.now().date()
        earliest = today - timedelta(days=max(0, int(search_days) - 1))
        latest = today + timedelta(days=self.FUTURE_TOLERANCE_DAYS)

        filtered: List[SearchResult] = []
        dropped_unknown = 0
        dropped_old = 0
        dropped_future = 0

        for item in response.results:
            published = self._normalize_news_publish_date(item.published_date)
            if published is None:
                dropped_unknown += 1
                continue
            if published < earliest:
                dropped_old += 1
                continue
            if published > latest:
                dropped_future += 1
                continue

            filtered.append(
                SearchResult(
                    title=item.title,
                    snippet=item.snippet,
                    url=item.url,
                    source=item.source,
                    published_date=published.isoformat(),
                )
            )
            if len(filtered) >= max_results:
                break

        if dropped_unknown or dropped_old or dropped_future:
            logger.info(
                "[新闻过滤] %s: provider=%s, total=%s, kept=%s, drop_unknown=%s, drop_old=%s, drop_future=%s, window=[%s,%s]",
                log_scope,
                response.provider,
                len(response.results),
                len(filtered),
                dropped_unknown,
                dropped_old,
                dropped_future,
                earliest.isoformat(),
                latest.isoformat(),
            )

        return SearchResponse(
            query=response.query,
            results=filtered,
            provider=response.provider,
            success=response.success,
            error_message=response.error_message,
            search_time=response.search_time,
        )

    def _normalize_and_limit_response(
        self,
        response: SearchResponse,
        *,
        max_results: int,
    ) -> SearchResponse:
        """Normalize parseable dates without enforcing freshness filtering."""
        if not response.success or not response.results:
            return response

        normalized_results: List[SearchResult] = []
        for item in response.results[:max_results]:
            normalized_date = self._normalize_news_publish_date(item.published_date)
            normalized_results.append(
                SearchResult(
                    title=item.title,
                    snippet=item.snippet,
                    url=item.url,
                    source=item.source,
                    published_date=(
                        normalized_date.isoformat() if normalized_date is not None else item.published_date
                    ),
                )
            )

        return SearchResponse(
            query=response.query,
            results=normalized_results,
            provider=response.provider,
            success=response.success,
            error_message=response.error_message,
            search_time=response.search_time,
        )
    
    def search_stock_news(
        self,
        stock_code: str,
        stock_name: str,
        max_results: int = 5,
        focus_keywords: Optional[List[str]] = None
    ) -> SearchResponse:
        """
        搜索股票相关新闻
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            max_results: 最大返回结果数
            focus_keywords: 重点关注的关键词列表
            
        Returns:
            SearchResponse 对象
        """
        # 策略窗口优先：ultra_short/short/medium/long = 1/3/7/30 天，
        # 并统一受 NEWS_MAX_AGE_DAYS 上限约束。
        search_days = self._effective_news_window_days()
        provider_max_results = self._provider_request_size(max_results)

        # 构建搜索查询（优化搜索效果）
        is_foreign = self._is_foreign_stock(stock_code)
        if focus_keywords:
            # 如果提供了关键词，直接使用关键词作为查询
            query = " ".join(focus_keywords)
        elif is_foreign:
            # 港股/美股使用英文搜索关键词
            query = f"{stock_name} {stock_code} stock latest news"
        else:
            # 默认主查询：股票名称 + 核心关键词
            query = f"{stock_name} {stock_code} 股票 最新消息"

        logger.info(
            (
                "搜索股票新闻: %s(%s), query='%s', 时间范围: 近%s天 "
                "(profile=%s, NEWS_MAX_AGE_DAYS=%s), 目标条数=%s, provider请求条数=%s"
            ),
            stock_name,
            stock_code,
            query,
            search_days,
            self.news_strategy_profile,
            self.news_max_age_days,
            max_results,
            provider_max_results,
        )

        # Check cache first
        cache_key = self._cache_key(query, max_results, search_days)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info(f"使用缓存搜索结果: {stock_name}({stock_code})")
            return cached

        # 依次尝试各个搜索引擎（若过滤后为空，继续尝试下一引擎）
        had_provider_success = False
        for provider in self._providers:
            if not provider.is_available:
                continue

            search_kwargs: Dict[str, Any] = {}
            if isinstance(provider, TavilySearchProvider):
                search_kwargs["topic"] = "news"

            response = provider.search(query, provider_max_results, days=search_days, **search_kwargs)
            filtered_response = self._filter_news_response(
                response,
                search_days=search_days,
                max_results=max_results,
                log_scope=f"{stock_code}:{provider.name}:stock_news",
            )
            had_provider_success = had_provider_success or bool(response.success)

            if filtered_response.success and filtered_response.results:
                logger.info(f"使用 {provider.name} 搜索成功")
                self._put_cache(cache_key, filtered_response)
                return filtered_response
            else:
                if response.success and not filtered_response.results:
                    logger.info(
                        "%s 搜索成功但过滤后无有效新闻，继续尝试下一引擎",
                        provider.name,
                    )
                else:
                    logger.warning(
                        "%s 搜索失败: %s，尝试下一个引擎",
                        provider.name,
                        response.error_message,
                    )

        if had_provider_success:
            return SearchResponse(
                query=query,
                results=[],
                provider="Filtered",
                success=True,
                error_message=None,
            )
        
        # 所有引擎都失败
        return SearchResponse(
            query=query,
            results=[],
            provider="None",
            success=False,
            error_message="所有搜索引擎都不可用或搜索失败"
        )
    
    def search_stock_events(
        self,
        stock_code: str,
        stock_name: str,
        event_types: Optional[List[str]] = None
    ) -> SearchResponse:
        """
        搜索股票特定事件（年报预告、减持等）
        
        专门针对交易决策相关的重要事件进行搜索
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            event_types: 事件类型列表
            
        Returns:
            SearchResponse 对象
        """
        if event_types is None:
            if self._is_foreign_stock(stock_code):
                event_types = ["earnings report", "insider selling", "quarterly results"]
            else:
                event_types = ["年报预告", "减持公告", "业绩快报"]
        
        # 构建针对性查询
        event_query = " OR ".join(event_types)
        query = f"{stock_name} ({event_query})"
        
        logger.info(f"搜索股票事件: {stock_name}({stock_code}) - {event_types}")
        
        # 依次尝试各个搜索引擎
        for provider in self._providers:
            if not provider.is_available:
                continue
            
            response = provider.search(query, max_results=5)
            
            if response.success:
                return response
        
        return SearchResponse(
            query=query,
            results=[],
            provider="None",
            success=False,
            error_message="事件搜索失败"
        )

    def search_x_signals(self, query: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        """Search X social signals via xAI X Search provider."""
        if not self._x_signal_provider or not self._x_signal_provider.is_available:
            return self._build_failed_response(query, "xAI X Search", "未配置 XAI_API_KEY")
        return self._x_signal_provider.search(query, max_results=max_results, days=days)
    
    def search_comprehensive_intel(
        self,
        stock_code: str,
        stock_name: str,
        max_searches: int = 9
    ) -> Dict[str, SearchResponse]:
        """
        多维度情报搜索（同时使用多个引擎、多个维度）
        
        搜索维度：
        1. 最新消息 - 近期新闻动态
        2. 风险排查 - 减持、处罚、利空
        3. 业绩预期 - 年报预告、业绩快报
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            max_searches: 最大搜索次数
            
        Returns:
            {维度名称: SearchResponse} 字典
        """
        results: Dict[str, SearchResponse] = {}
        search_count = 0

        market = self._market_tag(stock_code)
        is_foreign = self._is_foreign_stock(stock_code)
        is_index_etf = self.is_index_or_etf(stock_code, stock_name)

        # --- Phase 1: direct-only intelligence dimensions (market-specific) ---
        direct_dimensions: List[Dict[str, Any]] = []
        if market == "us" and not is_index_etf:
            direct_dimensions = [
                {
                    'name': 'official_filings',
                    'query': (
                        f"site:sec.gov {stock_name} {stock_code} "
                        f"(10-Q OR 10-K OR 8-K OR \"Form 4\" OR \"13D\" OR \"13G\")"
                    ),
                    'desc': '官方披露',
                    'direct_kind': 'sec_filings',
                },
                {
                    'name': 'china_exposure',
                    'query': (
                        f"{stock_name} {stock_code} "
                        f"(\"Greater China\" OR China revenue OR China sales OR China supply chain "
                        f"OR tariff OR export control OR PRC)"
                    ),
                    'desc': '中国暴露',
                    'direct_kind': 'sec_china_exposure',
                },
            ]
            if self._x_signal_provider and self._x_signal_provider.is_available:
                direct_dimensions.append({
                    'name': 'x_signal',
                    'query': (
                        f"{stock_name} {stock_code} earnings guidance product launch lawsuit "
                        f"short report analyst downgrade CEO comments"
                    ),
                    'desc': 'X社交信号',
                    'direct_kind': 'x_social_signal',
                    'engine_fallback': False,
                })
        elif market == "hk":
            direct_dimensions = [
                {
                    'name': 'official_announcements',
                    'query': (
                        f"(site:hkexnews.hk OR site:hkex.com.hk) "
                        f"{stock_name} {stock_code} announcement results profit warning buyback"
                    ),
                    'desc': '官方公告',
                    'direct_kind': 'hk_events',
                },
            ]
        elif market == "cn":
            direct_dimensions = [
                {
                    'name': 'official_announcements',
                    'query': (
                        f"(site:cninfo.com.cn OR site:sse.com.cn OR site:szse.cn) "
                        f"{stock_name} {stock_code} "
                        f"公告 OR 减持 OR 业绩预告 OR 监管函 OR 问询函"
                    ),
                    'desc': '官方公告',
                    'direct_kind': 'cn_official',
                    'engine_fallback': False,
                },
                {
                    'name': 'risk_alerts',
                    'query': (
                        f"{stock_name} {stock_code} "
                        f"减持 OR 立案调查 OR 退市风险 OR 监管函 OR 问询函 OR ST OR 行政处罚"
                    ),
                    'desc': '风险预警',
                    'engine_fallback': True,
                },
            ]

        provider_index = 0
        deferred_fallbacks: List[Dict[str, Any]] = []
        for dim in direct_dimensions:
            query = dim.get('query', '')
            response: Optional[SearchResponse] = None
            direct_kind = dim.get('direct_kind')

            if direct_kind == 'cn_official':
                response = self._direct_cninfo_announcements(stock_code, stock_name, query=query)
            elif direct_kind == 'hk_events':
                response = self._direct_hk_event_calendar(stock_code, stock_name, query=query)
            elif direct_kind == 'sec_filings':
                response = self._direct_sec_filings(stock_code, stock_name, query=query)
            elif direct_kind == 'sec_china_exposure':
                response = self._direct_us_china_exposure(stock_code, stock_name, query=query)
            elif direct_kind == 'x_social_signal':
                response = self._direct_x_social_signal(stock_code, stock_name, query=query)

            if response is not None and response.success:
                results[dim['name']] = response
                logger.info(f"[情报搜索] {dim['desc']}: 获取 {len(response.results)} 条结果 (来源: {response.provider})")
            else:
                failed_response = response if response is not None else self._build_failed_response(query, "None", "无可用情报源")
                results[dim['name']] = failed_response
                logger.warning(
                    "[情报搜索] %s: 直连失败 - %s",
                    dim['desc'],
                    failed_response.error_message or "未知错误",
                )
                if dim.get('engine_fallback', True) and query:
                    deferred_fallbacks.append(dim)

        # --- Phase 2: search-engine dimensions (with strict_freshness / tavily_topic) ---
        if is_foreign:
            search_dimensions = [
                {
                    'name': 'latest_news',
                    'query': f"{stock_name} {stock_code} latest news events",
                    'desc': '最新消息',
                    'tavily_topic': 'news',
                    'strict_freshness': True,
                },
                {
                    'name': 'market_analysis',
                    'query': f"{stock_name} analyst rating target price report",
                    'desc': '机构分析',
                    'tavily_topic': None,
                    'strict_freshness': False,
                },
                {
                    'name': 'risk_check',
                    'query': (
                        f"{stock_name} {stock_code} index performance outlook tracking error"
                        if is_index_etf else f"{stock_name} risk insider selling lawsuit litigation"
                    ),
                    'desc': '风险排查',
                    'tavily_topic': None if is_index_etf else 'news',
                    'strict_freshness': not is_index_etf,
                },
                {
                    'name': 'earnings',
                    'query': (
                        f"{stock_name} {stock_code} index performance composition outlook"
                        if is_index_etf else f"{stock_name} earnings revenue profit growth forecast"
                    ),
                    'desc': '业绩预期',
                    'tavily_topic': None,
                    'strict_freshness': False,
                },
                {
                    'name': 'industry',
                    'query': (
                        f"{stock_name} {stock_code} index sector allocation holdings"
                        if is_index_etf else (
                            f"{stock_name} industry competitors market share outlook "
                            f"{_BACKGROUND_SOURCE_EXCLUSIONS}"
                        )
                    ),
                    'desc': '行业分析',
                    'tavily_topic': None,
                    'strict_freshness': False,
                },
            ]
        else:
            search_dimensions = [
                {
                    'name': 'latest_news',
                    'query': f"{stock_name} {stock_code} 最新 新闻 重大 事件",
                    'desc': '最新消息',
                    'tavily_topic': 'news',
                    'strict_freshness': True,
                },
                {
                    'name': 'market_analysis',
                    'query': f"{stock_name} 研报 目标价 评级 深度分析",
                    'desc': '机构分析',
                    'tavily_topic': None,
                    'strict_freshness': False,
                },
                {
                    'name': 'risk_check',
                    'query': (
                        f"{stock_name} 指数走势 跟踪误差 净值 表现"
                        if is_index_etf else (
                            f"{stock_name} {stock_code} "
                            f"减持 OR 监管函 OR 问询函 OR 立案调查 OR 退市风险 OR "
                            f"行政处罚 OR 违规 OR 诉讼 OR 大宗交易折价"
                        )
                    ),
                    'desc': '风险排查',
                    'tavily_topic': None if is_index_etf else 'news',
                    'strict_freshness': not is_index_etf,
                },
                {
                    'name': 'earnings',
                    'query': (
                        f"{stock_name} 指数成分 净值 跟踪表现"
                        if is_index_etf else f"{stock_name} 业绩预告 财报 营收 净利润 同比增长"
                    ),
                    'desc': '业绩预期',
                    'tavily_topic': None,
                    'strict_freshness': False,
                },
                {
                    'name': 'industry',
                    'query': (
                        f"{stock_name} 指数成分股 行业配置 权重"
                        if is_index_etf else (
                            f"{stock_name} 所在行业 竞争对手 市场份额 行业前景 "
                            f"{_BACKGROUND_SOURCE_EXCLUSIONS}"
                        )
                    ),
                    'desc': '行业分析',
                    'tavily_topic': None,
                    'strict_freshness': False,
                },
            ]

            # A 股社交舆情维度：定向检索国内投资社区
            if not is_index_etf:
                search_dimensions.append({
                    'name': 'social_sentiment',
                    'query': (
                        f"(site:xueqiu.com OR site:guba.eastmoney.com) "
                        f"{stock_name} {stock_code}"
                    ),
                    'desc': '社交舆情',
                    'tavily_topic': None,
                    'strict_freshness': True,
                })
        
        search_days = self._effective_news_window_days()
        target_per_dimension = 3
        provider_max_results = self._provider_request_size(target_per_dimension)

        logger.info(
            (
                "开始多维度情报搜索: %s(%s), 时间范围: 近%s天 "
                "(profile=%s, NEWS_MAX_AGE_DAYS=%s), 目标条数=%s, provider请求条数=%s"
            ),
            stock_name,
            stock_code,
            search_days,
            self.news_strategy_profile,
            self.news_max_age_days,
            target_per_dimension,
            provider_max_results,
        )
        
        for dim in search_dimensions:
            if search_count >= max_searches:
                break
            
            # 选择搜索引擎（轮流使用）
            available_providers = [p for p in self._providers if p.is_available]
            if not available_providers:
                break
            
            provider = available_providers[provider_index % len(available_providers)]
            provider_index += 1
            
            logger.info(f"[情报搜索] {dim['desc']}: 使用 {provider.name}")

            if isinstance(provider, TavilySearchProvider) and dim.get('tavily_topic'):
                response = provider.search(
                    dim['query'],
                    max_results=provider_max_results,
                    days=search_days,
                    topic=dim['tavily_topic'],
                )
            else:
                response = provider.search(
                    dim['query'],
                    max_results=provider_max_results,
                    days=search_days,
                )
            if dim.get('strict_freshness'):
                filtered_response = self._filter_news_response(
                    response,
                    search_days=search_days,
                    max_results=target_per_dimension,
                    log_scope=f"{stock_code}:{provider.name}:{dim['name']}",
                )
            else:
                filtered_response = self._normalize_and_limit_response(
                    response,
                    max_results=target_per_dimension,
                )
            results[dim['name']] = filtered_response
            search_count += 1
            
            if response.success:
                logger.info(
                    "[情报搜索] %s: 原始=%s条, 过滤后=%s条",
                    dim['desc'],
                    len(response.results),
                    len(filtered_response.results),
                )
            else:
                logger.warning(f"[情报搜索] {dim['desc']}: 搜索失败 - {response.error_message}")
            
            # 短暂延迟避免请求过快
            time.sleep(0.5)

        # --- Phase 3: deferred engine fallback for failed direct dimensions ---
        for dim in deferred_fallbacks:
            if not self.is_available:
                break
            query = dim.get('query', '')
            logger.info(f"[情报搜索] {dim['desc']}: 搜索引擎回退")
            response, provider_index = self._run_provider_search(query, provider_index, max_results=3)
            if response is not None:
                results[dim['name']] = response
            time.sleep(0.5)
        
        return results
    
    @staticmethod
    def _format_search_error(error_message: Optional[str], max_length: int = 160) -> str:
        """Condense provider errors so degraded intel remains readable in prompts/logs."""
        if not error_message:
            return "未知错误"
        compact = " ".join(str(error_message).split())
        if len(compact) <= max_length:
            return compact
        return compact[: max_length - 3] + "..."

    def format_intel_report(self, intel_results: Dict[str, SearchResponse], stock_name: str) -> str:
        """
        格式化情报搜索结果为报告
        
        Args:
            intel_results: 多维度搜索结果
            stock_name: 股票名称
            
        Returns:
            格式化的情报报告文本
        """
        lines = [f"【{stock_name} 情报搜索结果】"]
        if (
            intel_results.get("industry")
            and intel_results["industry"].success
            and intel_results["industry"].results
        ):
            lines.append(
                "注：`行业分析` 维度可能包含百科、公司介绍或历史财务等背景资料，只能作背景参考，"
                "不能直接当作近7日新闻、最新催化或当前业绩展望。"
            )
        
        # 维度展示顺序
        display_order = [
            'latest_news',
            'official_announcements',
            'official_filings',
            'china_exposure',
            'x_signal',
            'event_calendar',
            'risk_check',
            'risk_alerts',
            'earnings',
            'market_analysis',
            'social_sentiment',
            'macro_flows',
            'industry',
        ]

        _DIM_LABELS = {
            'latest_news': '📰 最新消息',
            'official_announcements': '📣 官方公告',
            'official_filings': '📄 官方披露',
            'china_exposure': '🇨🇳 中国暴露',
            'x_signal': 'X 社交信号',
            'event_calendar': '🗓️ 事件日历',
            'market_analysis': '📈 机构分析',
            'risk_check': '⚠️ 风险排查',
            'risk_alerts': '🚨 风险预警',
            'earnings': '📊 业绩预期',
            'social_sentiment': '💬 社交舆情',
            'macro_flows': '🌐 宏观资金',
            'industry': '🏭 行业分析（背景资料）',
        }

        # 只输出有实际结果的维度，跳过搜索失败/无结果的维度，
        # 避免将错误信息（如 "余额不足"、"HTTP 403"）浪费 LLM token。
        skipped_dims: List[str] = []

        for dim_name in display_order:
            if dim_name not in intel_results:
                continue

            resp = intel_results[dim_name]
            dim_desc = _DIM_LABELS.get(dim_name, dim_name)

            if not resp.success or not resp.results:
                # 记录跳过原因用于日志，但不传给 LLM
                reason = self._format_search_error(resp.error_message) if resp.error_message else "无结果"
                skipped_dims.append(f"{dim_desc}({reason})")
                continue

            lines.append(f"\n{dim_desc} (来源: {resp.provider}):")
            for i, r in enumerate(resp.results[:4], 1):
                date_str = f" [{r.published_date}]" if r.published_date else ""
                lines.append(f"  {i}. {r.title}{date_str}")
                snippet = r.snippet[:150] if len(r.snippet) > 20 else r.snippet
                lines.append(f"     {snippet}...")

        if skipped_dims:
            logger.info(
                "[情报格式化] %s: 跳过无效维度 (%d/%d): %s",
                stock_name,
                len(skipped_dims),
                sum(1 for d in display_order if d in intel_results),
                "; ".join(skipped_dims),
            )

        # 如果所有维度都失败/无结果，返回空字符串，
        # 让调用方走 "未搜索到该股票近期的相关新闻" 分支。
        has_any_content = any(
            intel_results.get(d) and intel_results[d].success and intel_results[d].results
            for d in display_order
            if d in intel_results
        )
        if not has_any_content:
            return ""

        return "\n".join(lines)
    
    def batch_search(
        self,
        stocks: List[Dict[str, str]],
        max_results_per_stock: int = 3,
        delay_between: float = 1.0
    ) -> Dict[str, SearchResponse]:
        """
        Batch search news for multiple stocks.
        
        Args:
            stocks: List of stocks
            max_results_per_stock: Max results per stock
            delay_between: Delay between searches (seconds)
            
        Returns:
            Dict of results
        """
        results = {}
        
        for i, stock in enumerate(stocks):
            if i > 0:
                time.sleep(delay_between)
            
            code = stock.get('code', '')
            name = stock.get('name', '')
            
            response = self.search_stock_news(code, name, max_results_per_stock)
            results[code] = response
        
        return results

    def search_stock_price_fallback(
        self,
        stock_code: str,
        stock_name: str,
        max_attempts: int = 3,
        max_results: int = 5
    ) -> SearchResponse:
        """
        Enhance search when data sources fail.
        
        When all data sources (efinance, akshare, tushare, baostock, etc.) fail to get
        stock data, use search engines to find stock trends and price info as supplemental data for AI analysis.
        
        Strategy:
        1. Search using multiple keyword templates
        2. Try all available search engines for each keyword
        3. Aggregate and deduplicate results
        
        Args:
            stock_code: Stock Code
            stock_name: Stock Name
            max_attempts: Max search attempts (using different keywords)
            max_results: Max results to return
            
        Returns:
            SearchResponse object with aggregated results
        """

        if not self.is_available:
            return SearchResponse(
                query=f"{stock_name} 股价走势",
                results=[],
                provider="None",
                success=False,
                error_message="未配置搜索能力"
            )
        
        logger.info(f"[增强搜索] 数据源失败，启动增强搜索: {stock_name}({stock_code})")
        
        all_results = []
        seen_urls = set()
        successful_providers = []
        
        # 使用多个关键词模板搜索
        is_foreign = self._is_foreign_stock(stock_code)
        keywords = self.ENHANCED_SEARCH_KEYWORDS_EN if is_foreign else self.ENHANCED_SEARCH_KEYWORDS
        for i, keyword_template in enumerate(keywords[:max_attempts]):
            query = keyword_template.format(name=stock_name, code=stock_code)
            
            logger.info(f"[增强搜索] 第 {i+1}/{max_attempts} 次搜索: {query}")
            
            # 依次尝试各个搜索引擎
            for provider in self._providers:
                if not provider.is_available:
                    continue
                
                try:
                    response = provider.search(query, max_results=3)
                    
                    if response.success and response.results:
                        # 去重并添加结果
                        for result in response.results:
                            if result.url not in seen_urls:
                                seen_urls.add(result.url)
                                all_results.append(result)
                                
                        if provider.name not in successful_providers:
                            successful_providers.append(provider.name)
                        
                        logger.info(f"[增强搜索] {provider.name} 返回 {len(response.results)} 条结果")
                        break  # 成功后跳到下一个关键词
                    else:
                        logger.debug(f"[增强搜索] {provider.name} 无结果或失败")
                        
                except Exception as e:
                    logger.warning(f"[增强搜索] {provider.name} 搜索异常: {e}")
                    continue
            
            # 短暂延迟避免请求过快
            if i < max_attempts - 1:
                time.sleep(0.5)
        
        # 汇总结果
        if all_results:
            # 截取前 max_results 条
            final_results = all_results[:max_results]
            provider_str = ", ".join(successful_providers) if successful_providers else "None"
            
            logger.info(f"[增强搜索] 完成，共获取 {len(final_results)} 条结果（来源: {provider_str}）")
            
            return SearchResponse(
                query=f"{stock_name}({stock_code}) 股价走势",
                results=final_results,
                provider=provider_str,
                success=True,
            )
        else:
            logger.warning(f"[增强搜索] 所有搜索均未返回结果")
            return SearchResponse(
                query=f"{stock_name}({stock_code}) 股价走势",
                results=[],
                provider="None",
                success=False,
                error_message="增强搜索未找到相关信息"
            )

    def search_stock_with_enhanced_fallback(
        self,
        stock_code: str,
        stock_name: str,
        include_news: bool = True,
        include_price: bool = False,
        max_results: int = 5
    ) -> Dict[str, SearchResponse]:
        """
        综合搜索接口（支持新闻和股价信息）
        
        当 include_price=True 时，会同时搜索新闻和股价信息。
        主要用于数据源完全失败时的兜底方案。
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            include_news: 是否搜索新闻
            include_price: 是否搜索股价/走势信息
            max_results: 每类搜索的最大结果数
            
        Returns:
            {'news': SearchResponse, 'price': SearchResponse} 字典
        """
        results = {}
        
        if include_news:
            results['news'] = self.search_stock_news(
                stock_code, 
                stock_name, 
                max_results=max_results
            )
        
        if include_price:
            results['price'] = self.search_stock_price_fallback(
                stock_code,
                stock_name,
                max_attempts=3,
                max_results=max_results
            )
        
        return results

    def format_price_search_context(self, response: SearchResponse) -> str:
        """
        将股价搜索结果格式化为 AI 分析上下文
        
        Args:
            response: 搜索响应对象
            
        Returns:
            格式化的文本，可直接用于 AI 分析
        """
        if not response.success or not response.results:
            return "【股价走势搜索】未找到相关信息，请以其他渠道数据为准。"
        
        lines = [
            f"【股价走势搜索结果】（来源: {response.provider}）",
            "⚠️ 注意：以下信息来自网络搜索，仅供参考，可能存在延迟或不准确。",
            ""
        ]
        
        for i, result in enumerate(response.results, 1):
            date_str = f" [{result.published_date}]" if result.published_date else ""
            lines.append(f"{i}. 【{result.source}】{result.title}{date_str}")
            lines.append(f"   {result.snippet[:200]}...")
            lines.append("")
        
        return "\n".join(lines)


# === 便捷函数 ===
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """获取搜索服务单例"""
    global _search_service
    
    if _search_service is None:
        from src.config import get_config
        config = get_config()
        
        _search_service = SearchService(
            bocha_keys=config.bocha_api_keys,
            tavily_keys=config.tavily_api_keys,
            anspire_keys=config.anspire_api_keys,
            brave_keys=config.brave_api_keys,
            serpapi_keys=config.serpapi_keys,
            minimax_keys=config.minimax_api_keys,
            xai_keys=config.xai_api_keys,
            xai_search_model=config.xai_search_model,
            searxng_base_urls=config.searxng_base_urls,
            searxng_public_instances_enabled=config.searxng_public_instances_enabled,
            news_max_age_days=config.news_max_age_days,
            news_strategy_profile=getattr(config, "news_strategy_profile", "short"),
        )
    
    return _search_service


def reset_search_service() -> None:
    """重置搜索服务（用于测试）"""
    global _search_service
    _search_service = None


if __name__ == "__main__":
    # 测试搜索服务
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
    )
    
    # 手动测试（需要配置 API Key）
    service = get_search_service()
    
    if service.is_available:
        print("=== 测试股票新闻搜索 ===")
        response = service.search_stock_news("300389", "艾比森")
        print(f"搜索状态: {'成功' if response.success else '失败'}")
        print(f"搜索引擎: {response.provider}")
        print(f"结果数量: {len(response.results)}")
        print(f"耗时: {response.search_time:.2f}s")
        print("\n" + response.to_context())
    else:
        print("未配置搜索能力，跳过测试")
