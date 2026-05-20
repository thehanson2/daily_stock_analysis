# -*- coding: utf-8 -*-
"""
===================================
TushareFetcher - 备用数据源 1 (Priority 2)
===================================

数据来源：Tushare Pro API（挖地兔）
特点：需要 Token、有请求配额限制
优点：数据质量高、接口稳定

流控策略：
1. 实现"每分钟调用计数器"
2. 超过免费配额（80次/分）时，强制休眠到下一分钟
3. 使用 tenacity 实现指数退避重试
"""

import json as _json
import logging
import re
import time
import traceback
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, RateLimitError, STANDARD_COLUMNS, is_bse_code, is_st_stock, is_kc_cy_stock, normalize_stock_code, _is_hk_market
from .realtime_types import UnifiedRealtimeQuote, ChipDistribution
from src.config import get_config
import os
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# ETF code prefixes by exchange
# Shanghai: 51xxxx, 52xxxx, 56xxxx, 58xxxx
# Shenzhen: 15xxxx, 16xxxx, 18xxxx
_ETF_SH_PREFIXES = ('51', '52', '56', '58')
_ETF_SZ_PREFIXES = ('15', '16', '18')
_ETF_ALL_PREFIXES = _ETF_SH_PREFIXES + _ETF_SZ_PREFIXES


def _is_etf_code(stock_code: str) -> bool:
    """
    Check if the code is an ETF fund code.

    ETF code ranges:
    - Shanghai ETF: 51xxxx, 52xxxx, 56xxxx, 58xxxx
    - Shenzhen ETF: 15xxxx, 16xxxx, 18xxxx
    """
    code = stock_code.strip().split('.')[0]
    return code.startswith(_ETF_ALL_PREFIXES) and len(code) == 6


def _is_us_code(stock_code: str) -> bool:
    """
    判断代码是否为美股

    美股代码规则：
    - 1-5个大写字母，如 'AAPL', 'TSLA'
    - 可能包含 '.'，如 'BRK.B'
    """
    code = stock_code.strip().upper()
    return bool(re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', code))


class _TushareHttpClient:
    """Lightweight Tushare Pro client that does not require the tushare SDK."""

    def __init__(self, token: str, timeout: int = 30, api_url: str = "http://124.222.60.121:8020") -> None:
        self._token = token
        self._timeout = timeout
        self._api_url = api_url

    def query(self, api_name: str, fields: str = "", **kwargs) -> pd.DataFrame:
        req_params = {
            "api_name": api_name,
            "token": self._token,
            "params": kwargs,
            "fields": fields,
        }
        res = requests.post(self._api_url, json=req_params, timeout=self._timeout)
        if res.status_code != 200:
            raise Exception(f"Tushare API HTTP {res.status_code}")

        result = _json.loads(res.text)
        if result.get("code") != 0:
            raise Exception(result.get("msg") or f"Tushare API error code {result.get('code')}")

        data = result.get("data") or {}
        columns = data.get("fields") or []
        items = data.get("items") or []
        return pd.DataFrame(items, columns=columns)

    def __getattr__(self, api_name: str):
        if api_name.startswith("_"):
            raise AttributeError(api_name)

        def caller(**kwargs) -> pd.DataFrame:
            return self.query(api_name, **kwargs)

        return caller


class TushareFetcher(BaseFetcher):
    """
    Tushare Pro 数据源实现

    优先级：2
    数据来源：Tushare Pro API

    关键策略：
    - 每分钟调用计数器，防止超出配额
    - 超过 80 次/分钟时强制等待
    - 失败后指数退避重试

    配额说明（Tushare 免费用户）：
    - 每分钟最多 80 次请求
    - 每天最多 500 次请求
    """

    name = "TushareFetcher"
    priority = int(os.getenv("TUSHARE_PRIORITY", "2"))

    def __init__(self, rate_limit_per_minute: int = 80):
        self.rate_limit_per_minute = rate_limit_per_minute
        self._call_count = 0
        self._minute_start: Optional[float] = None
        self._api: Optional[object] = None
        self.date_list: Optional[List[str]] = None
        self._date_list_end: Optional[str] = None

        self._init_api()
        self.priority = self._determine_priority()

    def _init_api(self) -> None:
        config = get_config()
        if not config.tushare_token:
            logger.warning("Tushare Token 未配置，此数据源不可用")
            return
        try:
            self._api = self._build_api_client(config.tushare_token)
            logger.info("Tushare API 初始化成功")
        except Exception as e:
            logger.error(f"Tushare API 初始化失败: {e}")
            self._api = None

    def _build_api_client(self, token: str) -> _TushareHttpClient:
        client = _TushareHttpClient(token=token)
        logger.debug("Tushare API client configured for direct HTTP calls")
        return client

    def _determine_priority(self) -> int:
        config = get_config()
        if config.tushare_token and self._api is not None:
            logger.info("✅ 检测到 TUSHARE_TOKEN 且 API 初始化成功，Tushare 数据源优先级提升为最高 (Priority -1)")
            return -1
        return 2

    def is_available(self) -> bool:
        return self._api is not None

    def _check_rate_limit(self) -> None:
        current_time = time.time()
        if self._minute_start is None:
            self._minute_start = current_time
            self._call_count = 0
        elif current_time - self._minute_start >= 60:
            self._minute_start = current_time
            self._call_count = 0
            logger.debug("速率限制计数器已重置")

        if self._call_count >= self.rate_limit_per_minute:
            elapsed = current_time - self._minute_start
            sleep_time = max(0, 60 - elapsed) + 1
            logger.warning(
                f"Tushare 达到速率限制 ({self._call_count}/{self.rate_limit_per_minute} 次/分钟)，"
                f"等待 {sleep_time:.1f} 秒..."
            )
            time.sleep(sleep_time)
            self._minute_start = time.time()
            self._call_count = 0

        self._call_count += 1
        logger.debug(f"Tushare 当前分钟调用次数: {self._call_count}/{self.rate_limit_per_minute}")

    def _call_api_with_rate_limit(self, method_name: str, **kwargs) -> pd.DataFrame:
        if self._api is None:
            raise DataFetchError("Tushare API 未初始化，请检查 Token 配置")
        self._check_rate_limit()
        method = getattr(self._api, method_name)
        return method(**kwargs)

    def _get_china_now(self) -> datetime:
        return datetime.now(ZoneInfo("Asia/Shanghai"))

    def _get_trade_dates(self, end_date: Optional[str] = None) -> List[str]:
        if self._api is None:
            return []
        china_now = self._get_china_now()
        requested_end_date = end_date or china_now.strftime("%Y%m%d")
        if self.date_list is not None and self._date_list_end == requested_end_date:
            return self.date_list
        start_date = (china_now - timedelta(days=20)).strftime("%Y%m%d")
        try:
            df_cal = self._call_api_with_rate_limit(
                "trade_cal",
                exchange="SSE",
                start_date=start_date,
                end_date=requested_end_date,
            )
        except Exception as e:
            logger.warning(f"[Tushare] trade_cal 调用失败: {e}")
            return []

        if df_cal is None or df_cal.empty or "cal_date" not in df_cal.columns:
            logger.warning("[Tushare] trade_cal 返回为空，无法更新交易日历缓存")
            self.date_list = []
            self._date_list_end = requested_end_date
            return self.date_list

        trade_dates = sorted(
            df_cal[df_cal["is_open"] == 1]["cal_date"].astype(str).tolist(),
            reverse=True,
        )
        self.date_list = trade_dates
        self._date_list_end = requested_end_date
        return trade_dates

    @staticmethod
    def _pick_trade_date(trade_dates: List[str], use_today: bool) -> Optional[str]:
        if not trade_dates:
            return None
        if use_today or len(trade_dates) == 1:
            return trade_dates[0]
        return trade_dates[1]

    @staticmethod
    def _detect_exchange_hint(stock_code: str) -> Optional[str]:
        upper = (stock_code or "").strip().upper()
        if upper.startswith(("SH", "SS")) or upper.endswith((".SH", ".SS")):
            return "SH"
        if upper.startswith("SZ") or upper.endswith(".SZ"):
            return "SZ"
        if upper.startswith("BJ") or upper.endswith(".BJ"):
            return "BJ"
        return None

    @classmethod
    def _get_legacy_realtime_symbol(cls, stock_code: str) -> str:
        code = normalize_stock_code(stock_code)
        exchange_hint = cls._detect_exchange_hint(stock_code)
        if code == '000001' and exchange_hint == 'SH':
            return 'sh000001'
        if code == '399001':
            return 'sz399001'
        if code == '399006':
            return 'sz399006'
        if code == '000300':
            return 'sh000300'
        if is_bse_code(code):
            return f"bj{code}"
        return code

    def _convert_stock_code(self, stock_code: str) -> str:
        raw_code = stock_code.strip()
        if '.' in raw_code:
            ts_code = raw_code.upper()
            if ts_code.endswith('.SS'):
                return f"{ts_code[:-3]}.SH"
            return ts_code
        if _is_us_code(raw_code):
            raise DataFetchError(f"TushareFetcher 不支持美股 {raw_code}，请使用 AkshareFetcher 或 YfinanceFetcher")
        if _is_hk_market(raw_code):
            return normalize_stock_code(raw_code)
        code = normalize_stock_code(raw_code)
        exchange_hint = self._detect_exchange_hint(raw_code)
        if exchange_hint == "SH":
            return f"{code}.SH"
        if exchange_hint == "SZ":
            return f"{code}.SZ"
        if exchange_hint == "BJ":
            return f"{code}.BJ"
        if code.startswith(_ETF_SH_PREFIXES) and len(code) == 6:
            return f"{code}.SH"
        if code.startswith(_ETF_SZ_PREFIXES) and len(code) == 6:
            return f"{code}.SZ"
        if is_bse_code(code):
            return f"{code}.BJ"
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SH"
        elif code.startswith(('000', '002', '300')):
            return f"{code}.SZ"
        else:
            logger.warning(f"无法确定股票 {code} 的市场，默认使用深市")
            return f"{code}.SZ"

    def _convert_hk_stock_code_for_tushare(self, stock_code: str) -> str:
        raw_code = stock_code.strip()
        if _is_hk_market(raw_code):
            if "." in raw_code:
                ts_code = raw_code.upper()
                if ts_code.endswith(".SS"):
                    return f"{ts_code[:-3]}.SH"
                if ts_code.endswith(".HK"):
                    return ts_code
            digits = re.sub(r"\D", "", raw_code)
            if not digits:
                raise DataFetchError(f"无法识别港股代码 {raw_code}")
            code = digits[-5:].rjust(5, "0")
            return f"{code}.HK"
        return self._convert_stock_code(stock_code)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Tushare 获取原始数据

        根据代码类型选择不同接口：
        - 普通股票：daily()
        - ETF 基金：fund_daily()
        - 港股：hk_daily()
        """
        if self._api is None:
            raise DataFetchError("Tushare API 未初始化，请检查 Token 配置")

        if _is_us_code(stock_code):
            raise DataFetchError(f"TushareFetcher 不支持美股 {stock_code}，请使用 AkshareFetcher 或 YfinanceFetcher")

        self._check_rate_limit()

        is_hk = _is_hk_market(stock_code)
        is_etf = _is_etf_code(stock_code)

        if is_hk:
            ts_code = self._convert_hk_stock_code_for_tushare(stock_code)
            api_name = "hk_daily"
        else:
            ts_code = self._convert_stock_code(stock_code)
            api_name = "fund_daily" if is_etf else "daily"

        ts_start = start_date.replace('-', '')
        ts_end = end_date.replace('-', '')

        logger.debug(f"调用 Tushare {api_name}({ts_code}, {ts_start}, {ts_end})")

        try:
            if is_hk:
                df = self._api.hk_daily(
                    ts_code=ts_code,
                    start_date=ts_start,
                    end_date=ts_end,
                )
            elif is_etf:
                df = self._api.fund_daily(
                    ts_code=ts_code,
                    start_date=ts_start,
                    end_date=ts_end,
                )
            else:
                df = self._api.daily(
                    ts_code=ts_code,
                    start_date=ts_start,
                    end_date=ts_end,
                )
            return df
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['quota', '配额', 'limit', '权限']):
                logger.warning(f"Tushare 配额可能超限: {e}")
                raise RateLimitError(f"Tushare 配额超限: {e}") from e
            raise DataFetchError(f"Tushare 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()
        is_hk = _is_hk_market(stock_code)

        column_mapping = {
            'trade_date': 'date',
            'vol': 'volume',
        }
        df = df.rename(columns=column_mapping)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        if 'volume' in df.columns and not is_hk:
            df['volume'] = df['volume'] * 100

        if 'amount' in df.columns and not is_hk:
            df['amount'] = df['amount'] * 1000

        df['code'] = stock_code
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        return df

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        if self._api is None:
            logger.warning("Tushare API 未初始化，无法获取股票名称")
            return None
        if hasattr(self, '_stock_name_cache') and stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}
        try:
            self._check_rate_limit()
            if _is_hk_market(stock_code):
                ts_code = self._convert_hk_stock_code_for_tushare(stock_code)
                df = self._api.hk_basic(
                    ts_code=ts_code,
                    fields='ts_code,name'
                )
            elif _is_etf_code(stock_code):
                ts_code = self._convert_stock_code(stock_code)
                df = self._api.fund_basic(
                    ts_code=ts_code,
                    fields='ts_code,name'
                )
            else:
                ts_code = self._convert_stock_code(stock_code)
                df = self._api.stock_basic(
                    ts_code=ts_code,
                    fields='ts_code,name'
                )
            if df is not None and not df.empty:
                name = df.iloc[0]['name']
                self._stock_name_cache[stock_code] = name
                logger.debug(f"Tushare 获取股票名称成功: {stock_code} -> {name}")
                return name
        except Exception as e:
            logger.warning(f"Tushare 获取股票名称失败 {stock_code}: {e}")
        return None

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        if self._api is None:
            logger.warning("Tushare API 未初始化，无法获取股票列表")
            return None
        try:
            self._check_rate_limit()
            df = self._api.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,name,industry,area,market'
            )
            if df is None or df.empty:
                return None
            df = df.copy()
            df['code'] = df['ts_code'].astype(str).str.split('.').str[0]
            if not hasattr(self, '_stock_name_cache'):
                self._stock_name_cache = {}
            for _, row in df.iterrows():
                self._stock_name_cache[row['code']] = row['name']
            logger.info(f"Tushare 获取股票列表成功: {len(df)} 条")
            return df[['code', 'name', 'industry', 'area', 'market']]
        except Exception as e:
            logger.warning(f"Tushare 获取股票列表失败: {e}")
        return None

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        if self._api is None:
            return None
        if _is_hk_market(stock_code):
            logger.debug(f"TushareFetcher 跳过港股实时行情 {stock_code}")
            return None
        normalized_code = normalize_stock_code(stock_code)
        from .realtime_types import RealtimeSource, safe_float, safe_int
        self._check_rate_limit()
        try:
            ts_code = self._convert_stock_code(stock_code)
            df = self._api.quotation(ts_code=ts_code)
            if df is not None and not df.empty:
                row = df.iloc[0]
                logger.debug(f"Tushare Pro 实时行情获取成功: {stock_code}")
                return UnifiedRealtimeQuote(
                    code=normalized_code,
                    name=str(row.get('name', '')),
                    source=RealtimeSource.TUSHARE,
                    price=safe_float(row.get('price')),
                    change_pct=safe_float(row.get('pct_chg')),
                    change_amount=safe_float(row.get('change')),
                    volume=safe_int(row.get('vol')),
                    amount=safe_float(row.get('amount')),
                    high=safe_float(row.get('high')),
                    low=safe_float(row.get('low')),
                    open_price=safe_float(row.get('open')),
                    pre_close=safe_float(row.get('pre_close')),
                    turnover_rate=safe_float(row.get('turnover_ratio')),
                    pe_ratio=safe_float(row.get('pe')),
                    pb_ratio=safe_float(row.get('pb')),
                    total_mv=safe_float(row.get('total_mv')),
                )
        except Exception as e:
            logger.debug(f"Tushare Pro 实时行情不可用 (可能是积分不足): {e}")
        try:
            import tushare as ts
            symbol = self._get_legacy_realtime_symbol(stock_code)
            df = ts.get_realtime_quotes(symbol)
            if df is None or df.empty:
                return None
            row = df.iloc[0]
            price = safe_float(row['price'])
            pre_close = safe_float(row['pre_close'])
            change_pct = 0.0
            change_amount = 0.0
            if price and pre_close and pre_close > 0:
                change_amount = price - pre_close
                change_pct = (change_amount / pre_close) * 100
            return UnifiedRealtimeQuote(
                code=normalized_code,
                name=str(row['name']),
                source=RealtimeSource.TUSHARE,
                price=price,
                change_pct=round(change_pct, 2),
                change_amount=round(change_amount, 2),
                volume=safe_int(row['volume']) // 100,
                amount=safe_float(row['amount']),
                high=safe_float(row['high']),
                low=safe_float(row['low']),
                open_price=safe_float(row['open']),
                pre_close=pre_close,
            )
        except Exception as e:
            logger.warning(f"Tushare (旧版) 获取实时行情失败 {stock_code}: {e}")
            return None

    def get_main_indices(self, region: str = "cn") -> Optional[List[dict]]:
        if region != "cn":
            return None
        if self._api is None:
            return None
        from .realtime_types import safe_float
        indices_map = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指',
            '000688.SH': '科创50',
            '000016.SH': '上证50',
            '000300.SH': '沪深300',
        }
        try:
            self._check_rate_limit()
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - pd.Timedelta(days=5)).strftime('%Y%m%d')
            results = []
            for ts_code, name in indices_map.items():
                try:
                    df = self._api.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    if df is not None and not df.empty:
                        row = df.iloc[0]
                        current = safe_float(row['close'])
                        prev_close = safe_float(row['pre_close'])
                        results.append({
                            'code': ts_code.split('.')[0],
                            'name': name,
                            'current': current,
                            'change': safe_float(row['change']),
                            'change_pct': safe_float(row['pct_chg']),
                            'open': safe_float(row['open']),
                            'high': safe_float(row['high']),
                            'low': safe_float(row['low']),
                            'prev_close': prev_close,
                            'volume': safe_float(row['vol']),
                            'amount': safe_float(row['amount']) * 1000,
                            'amplitude': 0.0
                        })
                except Exception as e:
                    logger.debug(f"Tushare 获取指数 {name} 失败: {e}")
                    continue
            if results:
                return results
            else:
                logger.warning("[Tushare] 未获取到指数行情数据")
        except Exception as e:
            logger.error(f"[Tushare] 获取指数行情失败: {e}")
        return None

    def get_market_stats(self) -> Optional[dict]:
        if self._api is None:
            return None
        try:
            logger.info("[Tushare] ts.pro_api() 获取市场统计...")
            china_now = self._get_china_now()
            current_clock = china_now.strftime("%H:%M")
            current_date = china_now.strftime("%Y%m%d")
            trade_dates = self._get_trade_dates(current_date)
            if not trade_dates:
                return None
            if current_date in trade_dates:
                if current_clock < '09:30' or current_clock > '16:30':
                    use_realtime = False
                else:
                    use_realtime = True
            else:
                use_realtime = False
            if use_realtime:
                try:
                    df = self._call_api_with_rate_limit("rt_k", ts_code='3*.SZ,6*.SH,0*.SZ,92*.BJ')
                    if df is not None and not df.empty:
                        return self._calc_market_stats(df)
                except Exception as e:
                    logger.error(f"[Tushare] ts.pro_api().rt_k 尝试获取实时数据失败: {e}")
                    return None
            else:
                if current_date not in trade_dates:
                    last_date = self._pick_trade_date(trade_dates, use_today=True)
                else:
                    if current_clock < '09:30':
                        last_date = self._pick_trade_date(trade_dates, use_today=False)
                    else:
                        last_date = self._pick_trade_date(trade_dates, use_today=True)
                if last_date is None:
                    return None
                try:
                    df = self._call_api_with_rate_limit(
                        "daily",
                        TS_CODE='3*.SZ,6*.SH,0*.SZ,92*.BJ',
                        start_date=last_date,
                        end_date=last_date,
                    )
                    df.columns = [col.lower() for col in df.columns]
                    df_basic = self._call_api_with_rate_limit("stock_basic", fields='ts_code,name')
                    df = pd.merge(df, df_basic, on='ts_code', how='left')
                    if 'amount' in df.columns:
                        df['amount'] = df['amount'] * 1000
                    if df is not None and not df.empty:
                        return self._calc_market_stats(df)
                except Exception as e:
                    logger.error(f"[Tushare] ts.pro_api().daily 获取数据失败: {e}")
        except Exception as e:
            logger.error(f"[Tushare] 获取市场统计失败: {e}")
        return None

    def _calc_market_stats(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        import numpy as np
        df = df.copy()
        code_col = next((c for c in ['代码', '股票代码', 'ts_code','stock_code'] if c in df.columns), None)
        name_col = next((c for c in ['名称', '股票名称','name','name'] if c in df.columns), None)
        close_col = next((c for c in ['最新价', '最新价', 'close','lastPrice'] if c in df.columns), None)
        pre_close_col = next((c for c in ['昨收', '昨日收盘', 'pre_close','lastClose'] if c in df.columns), None)
        amount_col = next((c for c in ['成交额', '成交额', 'amount','amount'] if c in df.columns), None)
        limit_up_count = 0
        limit_down_count = 0
        up_count = 0
        down_count = 0
        flat_count = 0
        for code, name, current_price, pre_close, amount in zip(
            df[code_col], df[name_col], df[close_col], df[pre_close_col], df[amount_col]
        ):
            if pd.isna(current_price) or pd.isna(pre_close) or current_price in ['-'] or pre_close in ['-'] or amount == 0:
                continue
            current_price = float(current_price)
            pre_close = float(pre_close)
            pure_code = normalize_stock_code(str(code))
            if is_bse_code(pure_code):
                ratio = 0.30
            elif is_kc_cy_stock(pure_code):
                ratio = 0.20
            elif is_st_stock(name):
                ratio = 0.05
            else:
                ratio = 0.10
            limit_up_price = np.floor(pre_close * (1 + ratio) * 100 + 0.5) / 100.0
            limit_down_price = np.floor(pre_close * (1 - ratio) * 100 + 0.5) / 100.0
            limit_up_price_Tolerance = round(abs(pre_close * (1 + ratio) - limit_up_price), 10)
            limit_down_price_Tolerance = round(abs(pre_close * (1 - ratio) - limit_down_price), 10)
            if current_price > 0:
                is_limit_up = (current_price > 0) and (abs(current_price - limit_up_price) <= limit_up_price_Tolerance)
                is_limit_down = (current_price > 0) and (abs(current_price - limit_down_price) <= limit_down_price_Tolerance)
                if is_limit_up:
                    limit_up_count += 1
                if is_limit_down:
                    limit_down_count += 1
                if current_price > pre_close:
                    up_count += 1
                elif current_price < pre_close:
                    down_count += 1
                else:
                    flat_count += 1
        stats = {
            'up_count': up_count,
            'down_count': down_count,
            'flat_count': flat_count,
            'limit_up_count': limit_up_count,
            'limit_down_count': limit_down_count,
            'total_amount': 0.0,
        }
        if amount_col and amount_col in df.columns:
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
            stats['total_amount'] = (df[amount_col].sum() / 1e8)
        return stats

    def get_trade_time(self, early_time='09:30', late_time='16:30') -> Optional[str]:
        china_now = self._get_china_now()
        china_date = china_now.strftime("%Y%m%d")
        china_clock = china_now.strftime("%H:%M")
        trade_dates = self._get_trade_dates(china_date)
        if not trade_dates:
            return None
        if china_date in trade_dates:
            if early_time < china_clock < late_time:
                use_today = False
            else:
                use_today = True
        else:
            use_today = True
        start_date = self._pick_trade_date(trade_dates, use_today=use_today)
        if start_date is None:
            return None
        if not use_today:
            logger.info(f"[Tushare] 当前时间 {china_clock} 可能无法获取当天筹码分布，尝试获取前一个交易日的数据 {start_date}")
        return start_date

    def get_sector_rankings(self, n: int = 5) -> Optional[Tuple[list, list]]:
        def _get_rank_top_n(df: pd.DataFrame, change_col: str, industry_name: str, n: int) -> Tuple[list, list]:
            df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
            df = df.dropna(subset=[change_col])
            top = df.nlargest(n, change_col)
            top_sectors = [
                {'name': row[industry_name], 'change_pct': row[change_col]}
                for _, row in top.iterrows()
            ]
            bottom = df.nsmallest(n, change_col)
            bottom_sectors = [
                {'name': row[industry_name], 'change_pct': row[change_col]}
                for _, row in bottom.iterrows()
            ]
            return top_sectors, bottom_sectors

        start_date = self.get_trade_time(early_time='00:00', late_time='15:30')
        if not start_date:
            return None
        logger.info("[Tushare] ts.pro_api().moneyflow_ind_ths 获取板块排行(同花顺)...")
        try:
            df = self._call_api_with_rate_limit("moneyflow_ind_ths", trade_date=start_date)
            if df is not None and not df.empty:
                change_col = 'pct_change'
                name = 'industry'
                if change_col in df.columns:
                    return _get_rank_top_n(df, change_col, name, n)
        except Exception as e:
            logger.warning(f"[Tushare] 获取同花顺行业板块涨跌榜失败: {e} 尝试东财接口")
        logger.info("[Tushare] ts.pro_api().moneyflow_ind_dc 获取板块排行(东财)...")
        try:
            df = self._call_api_with_rate_limit("moneyflow_ind_dc", trade_date=start_date)
            if df is not None and not df.empty:
                df = df[df['content_type'] == '行业']
                change_col = 'pct_change'
                name = 'name'
                if change_col in df.columns:
                    return _get_rank_top_n(df, change_col, name, n)
        except Exception as e:
            logger.warning(f"[Tushare] 获取东财行业板块涨跌榜失败: {e}")
            return None
        return None

    # ==================== 筹码分布相关方法 ====================

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """
        获取筹码分布数据（优化版）

        数据来源：
        - 平均成本、获利比例：优先从 ts.pro_api().cyq_perf() 获取
        - 集中度（90/70）：从 ts.pro_api().cyq_chips() 计算

        注意：ETF/指数/港股不支持，直接返回 None。
        """
        if self._api is None:
            logger.warning("[Tushare] Tushare API 未初始化，无法获取筹码分布")
            return None

        if _is_us_code(stock_code):
            logger.warning(f"[Tushare] TushareFetcher 不支持美股 {stock_code} 的筹码分布")
            return None
        if _is_etf_code(stock_code):
            logger.warning(f"[Tushare] TushareFetcher 不支持 ETF {stock_code} 的筹码分布")
            return None
        if _is_hk_market(stock_code):
            logger.warning(f"[Tushare] TushareFetcher 不支持港股 {stock_code} 的筹码分布")
            return None

        try:
            # 1. 获取交易日（19点后才有当天数据）
            start_date = self.get_trade_time(early_time='00:00', late_time='19:00')
            if not start_date:
                logger.warning(f"[Tushare] 无法获取交易日，筹码分布获取失败")
                return None
            logger.debug(f"[筹码分布] 使用交易日期: {start_date}")

            ts_code = self._convert_stock_code(stock_code)
            logger.debug(f"[筹码分布] ts_code: {ts_code}")

            # 2. 获取 cyq_chips 数据（用于集中度）
            df_chips = self._call_api_with_rate_limit(
                "cyq_chips",
                ts_code=ts_code,
                start_date=start_date,
                end_date=start_date,
            )
            if df_chips is None or df_chips.empty:
                logger.warning(f"[Tushare] cyq_chips 返回空数据，stock={stock_code}, date={start_date}")
                return None
            logger.debug(f"[筹码分布] cyq_chips 数据行数: {len(df_chips)}")

            # 3. 获取 cyq_perf 数据（平均成本、获利比例）
            df_perf = None
            try:
                df_perf = self._call_api_with_rate_limit(
                    "cyq_perf",
                    ts_code=ts_code,
                    trade_date=start_date,
                    fields='ts_code,trade_date,avg_cost,profit_ratio'
                )
            except Exception as e:
                logger.warning(f"[Tushare] cyq_perf 调用失败，将降级使用 cyq_chips 计算: {e}")

            # 4. 获取当日收盘价（用于集中度计算）
            daily_df = self._call_api_with_rate_limit(
                "daily",
                ts_code=ts_code,
                start_date=start_date,
                end_date=start_date,
            )
            if daily_df is None or daily_df.empty:
                logger.warning(f"[Tushare] daily 接口未返回数据，stock={stock_code}, date={start_date}")
                return None
            if 'close' not in daily_df.columns:
                logger.warning(f"[Tushare] daily 返回缺少 'close' 列，实际列: {daily_df.columns.tolist()}")
                return None
            current_price = daily_df.iloc[0]['close']
            logger.debug(f"[筹码分布] 当前价格: {current_price}")

            # 5. 计算集中度（始终从 cyq_chips 计算）
            metrics = self.compute_cyq_metrics(df_chips, current_price)
            logger.debug(f"[筹码分布] 集中度: 90%={metrics['90集中度']:.4f}, 70%={metrics['70集中度']:.4f}")

            # 6. 处理平均成本和获利比例
            use_perf = False
            avg_cost = None
            profit_ratio = None

            if df_perf is not None and not df_perf.empty:
                # 检查是否包含所需列
                if 'avg_cost' in df_perf.columns and 'profit_ratio' in df_perf.columns:
                    # 检查数据是否有有效值（非空、非NaN）
                    if not pd.isna(df_perf.iloc[0]['avg_cost']) and not pd.isna(df_perf.iloc[0]['profit_ratio']):
                        avg_cost = float(df_perf.iloc[0]['avg_cost'])
                        profit_ratio_raw = float(df_perf.iloc[0]['profit_ratio'])
                        # 自适应转换：如果获利比例 > 1，认为是百分比，除以100；否则保持原值
                        if profit_ratio_raw > 1:
                            profit_ratio = profit_ratio_raw / 100.0
                        else:
                            profit_ratio = profit_ratio_raw
                        use_perf = True
                        logger.info(f"[筹码分布] 使用 cyq_perf: avg_cost={avg_cost}, profit_ratio={profit_ratio:.4f} (原始={profit_ratio_raw})")
                    else:
                        logger.warning(f"[Tushare] cyq_perf 返回数据包含空值，将降级使用 cyq_chips 计算")
                else:
                    logger.warning(f"[Tushare] cyq_perf 缺少必要列，实际列: {df_perf.columns.tolist()}，将降级使用 cyq_chips 计算")

            if not use_perf:
                avg_cost = metrics['平均成本']
                profit_ratio = metrics['获利比例']
                logger.info(f"[筹码分布] 降级使用 cyq_chips 计算: avg_cost={avg_cost}, profit_ratio={profit_ratio:.4f}")

            # 7. 构建 ChipDistribution 对象
            chip = ChipDistribution(
                code=stock_code,
                date=datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d'),
                profit_ratio=profit_ratio,
                avg_cost=avg_cost,
                cost_90_low=metrics['90成本-低'],
                cost_90_high=metrics['90成本-高'],
                concentration_90=metrics['90集中度'],
                cost_70_low=metrics['70成本-低'],
                cost_70_high=metrics['70成本-高'],
                concentration_70=metrics['70集中度'],
            )

            logger.info(f"[筹码分布] {stock_code} 日期={chip.date}: 获利比例={chip.profit_ratio:.1%}, "
                        f"平均成本={chip.avg_cost}, 90%集中度={chip.concentration_90:.2%}, "
                        f"70%集中度={chip.concentration_70:.2%}")
            return chip

        except Exception as e:
            logger.warning(f"[Tushare] 获取筹码分布失败 {stock_code}: {e}\n{traceback.format_exc()}")
            return None

    def compute_cyq_metrics(self, df: pd.DataFrame, current_price: float) -> dict:
        """
        基于 Tushare 的筹码分布明细表 (cyq_chips) 计算常用筹码指标（优化版）

        改进点：
        1. 输入校验，避免静默失败
        2. 归一化到 1.0，避免多余的乘除运算
        3. 获利比例采用严格小于当前价（不含当前价）
        4. 成本区间使用线性插值，提高分位价格精度
        5. 集中度直接返回小数（0-1区间）

        :param df: 包含 'price' 和 'percent' 列的 DataFrame
        :param current_price: 股票当天的当前价/收盘价
        :return: 包含各项筹码指标的字典（小数形式）
        """
        import numpy as np

        # ---------- 1. 输入校验 ----------
        if df is None or df.empty:
            raise ValueError("筹码分布数据为空，无法计算指标")
        if 'price' not in df.columns or 'percent' not in df.columns:
            raise ValueError("DataFrame 缺少必需的 'price' 或 'percent' 列")
        total_percent = df['percent'].sum()
        if total_percent == 0:
            raise ValueError("筹码占比总和为0，数据无效")

        # ---------- 2. 排序与归一化（到 1.0）----------
        df_sorted = df.sort_values(by='price', ascending=True).reset_index(drop=True)
        df_sorted['norm'] = df_sorted['percent'] / total_percent   # 归一化，总和 = 1.0
        df_sorted['cumsum'] = df_sorted['norm'].cumsum()

        # ---------- 3. 获利比例（严格小于当前价）----------
        winner = df_sorted[df_sorted['price'] < current_price]['norm'].sum()

        # ---------- 4. 平均成本 ----------
        avg_cost = np.average(df_sorted['price'], weights=df_sorted['norm'])

        # ---------- 5. 辅助函数：线性插值获取分位价格 ----------
        def percentile_price(target_pct: float) -> float:
            if target_pct <= 0:
                return df_sorted.iloc[0]['price']
            if target_pct >= 1:
                return df_sorted.iloc[-1]['price']
            idx = df_sorted['cumsum'].searchsorted(target_pct)
            # 精确匹配
            if idx < len(df_sorted) and df_sorted.iloc[idx]['cumsum'] == target_pct:
                return df_sorted.iloc[idx]['price']
            # 边界处理
            if idx == 0:
                return df_sorted.iloc[0]['price']
            if idx >= len(df_sorted):
                return df_sorted.iloc[-1]['price']
            # 线性插值
            prev = df_sorted.iloc[idx-1]
            curr = df_sorted.iloc[idx]
            if curr['cumsum'] - prev['cumsum'] == 0:
                return prev['price']
            ratio = (target_pct - prev['cumsum']) / (curr['cumsum'] - prev['cumsum'])
            return prev['price'] + ratio * (curr['price'] - prev['price'])

        # ---------- 6. 90% 成本区与集中度 ----------
        low90 = percentile_price(0.05)
        high90 = percentile_price(0.95)
        concentration_90 = (high90 - low90) / (high90 + low90) if (high90 + low90) != 0 else 0.0

        # ---------- 7. 70% 成本区与集中度 ----------
        low70 = percentile_price(0.15)
        high70 = percentile_price(0.85)
        concentration_70 = (high70 - low70) / (high70 + low70) if (high70 + low70) != 0 else 0.0

        # ---------- 8. 返回结果（所有数值为小数，保留4位）----------
        return {
            "获利比例": round(winner, 4),
            "平均成本": round(avg_cost, 4),
            "90成本-低": round(low90, 4),
            "90成本-高": round(high90, 4),
            "90集中度": round(concentration_90, 4),
            "70成本-低": round(low70, 4),
            "70成本-高": round(high70, 4),
            "70集中度": round(concentration_70, 4),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fetcher = TushareFetcher()
    try:
        df = fetcher.get_daily_data('600519')
        if df is not None:
            print(f"获取成功，共 {len(df)} 条数据")
            print(df.tail())
        else:
            print("获取日线数据失败")
        name = fetcher.get_stock_name('600519')
        print(f"股票名称: {name}")
    except Exception as e:
        print(f"获取失败: {e}")

    print("\n" + "=" * 50)
    print("Testing get_market_stats (tushare)")
    print("=" * 50)
    try:
        stats = fetcher.get_market_stats()
        if stats:
            print(f"Market Stats successfully computed:")
            print(f"Up: {stats['up_count']} (Limit Up: {stats['limit_up_count']})")
            print(f"Down: {stats['down_count']} (Limit Down: {stats['limit_down_count']})")
            print(f"Flat: {stats['flat_count']}")
            print(f"Total Amount: {stats['total_amount']:.2f} 亿 (Yi)")
        else:
            print("Failed to compute market stats.")
    except Exception as e:
        print(f"Failed to compute market stats: {e}")

    print("\n" + "=" * 50)
    print("测试筹码分布数据获取")
    print("=" * 50)
    try:
        chip = fetcher.get_chip_distribution('600519')
        if chip:
            print(f"获利比例: {chip.profit_ratio:.2%}, 平均成本: {chip.avg_cost}, 90集中度: {chip.concentration_90:.2%}")
        else:
            print("未获取到筹码分布数据")
    except Exception as e:
        print(f"[筹码分布] 获取失败: {e}")

    print("\n" + "=" * 50)
    print("测试行业板块排名获取")
    print("=" * 50)
    try:
        rankings = fetcher.get_sector_rankings(n=5)
        if rankings:
            top, bottom = rankings
            print("涨幅榜 Top 5:")
            for sector in top:
                print(f"{sector['name']}: {sector['change_pct']}%")
            print("\n跌幅榜 Top 5:")
            for sector in bottom:
                print(f"{sector['name']}: {sector['change_pct']}%")
        else:
            print("未获取到行业板块排名数据")
    except Exception as e:
        print(f"[行业板块排名] 获取失败: {e}")
