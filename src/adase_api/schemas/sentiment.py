import os
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
from enum import Enum


class FreqGranularity(str, Enum):
    """Down sample data frequency (from the last), usually to reduce latency and/or data size"""
    NONE = ''
    NONE2 = '-'
    HIGH = "-1h"
    MEDIUM = "-3h"
    LOW = "-1d"
    DEFAULT = os.environ.get('DEFAULT_FreqGranularity', '-1d')


class RollPeriod(str, Enum):
    NONE = ''
    ONE_WEEK = "7d"
    FOUR_WEEKS = "28d"
    QUARTER = "92d"
    YEAR = "365d"
    DEFAULT = os.environ.get('DEFAULT_RollPeriod', '28d')


class BBandPeriod(str, Enum):
    """Supported Bollinger band periods"""
    NONE = ''
    WEEK = "7d"
    TWO_WEEK = '14d'
    FOUR_WEEK = '28d'
    QUARTER = "92d"
    YEAR = "365d"
    DEFAULT = os.environ.get('DEFAULT_BBandPeriod', '28d')


class TAIndicator(str, Enum):
    """Supported indicators, used for technical analysis on sentiment"""
    NONE = ''
    COVERAGE = "coverage"  # query hits / total collection
    SCORE = "score"  # sentiment polarity score
    SCORE_COVERAGE = "score_coverage"  # `coverage` x `score`
    HITS = "hits"  # absolute no. of query hits
    DEFAULT = os.environ.get('DEFAULT_TAIndicator', 'coverage')


class BBandConfig(BaseModel):
    """Back-end side Bollinger bands settings (optional)"""
    indicator: Optional[TAIndicator] = 'coverage'
    period: Optional[BBandPeriod] = '7d'
    std: Optional[int] = 2


class ProcessConfig(BaseModel):
    """Back-end side data processing (optional)"""
    roll_period: Optional[RollPeriod] = ''  # rolling average
    freq: Optional[FreqGranularity] = '-1d'  # down sample (from the end)
    z_score: Optional[bool] = False  # z-score normalisation


class SentimentEngine(str, Enum):
    topic = 'topic'  # deep-learning, shorter history
    keyword = 'keyword'  # keywords


class Credentials(BaseModel):
    username: Optional[str] = os.environ.get('ADA_API_USERNAME', '')
    password: Optional[str] = os.environ.get('ADA_API_PASSWORD', '')


class QuerySentimentAPI(BaseModel):
    token: Optional[str] = None
    many_query: List  # comma separated, syntax depends on engine
    engine: Optional[SentimentEngine] = 'topic'
    start_date: Optional[datetime] = datetime.utcnow() - timedelta(days=92)
    end_date: Optional[datetime] = None
    time_started: Optional[datetime] = datetime.utcnow()
    process_cfg: Optional[ProcessConfig] = ProcessConfig()
    bband: Optional[BBandConfig] = None
    normalize_to_global: Optional[bool] = True
    z_score: Optional[bool] = True
    run_async: Optional[bool] = True
    credentials: Optional[Credentials] = Credentials()


class QuerySentimentTopic(BaseModel):
    token: Optional[str] = None

    text: List
    top_n: Optional[int] = 3
    normalize_to_global: Optional[bool] = True
    z_score: Optional[bool] = True
    min_global_row_count: Optional[int] = 100  # min no. of global rows to estimate a chart
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    freq: Optional[str] = '-1h'
    languages: Optional[list] = []
    check_geoh3: Optional[bool] = False

    live: Optional[bool] = True
    max_rows: Optional[int] = 10000
    run_async: Optional[bool] = True
    credentials: Optional[Credentials] = Credentials()
