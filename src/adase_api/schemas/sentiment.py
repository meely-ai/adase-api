import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, validator, ValidationError
from enum import Enum


class Credentials(BaseModel):
    username: Optional[str] = os.environ.get('ADA_API_USERNAME', '')
    password: Optional[str] = os.environ.get('ADA_API_PASSWORD', '')


class FilterSampleDailySize(BaseModel):
    daily_threshold: int = 10  # Minimum daily hits
    total_records: float = 1e6  # Estimated total daily size
    window: Optional[str] = "35d"  # Time window, e.g., "35d", "3d", "1w"

    @validator("daily_threshold")
    def validate_daily_threshold(cls, value):
        if not (1 <= value <= 1e3):
            raise ValueError(f"daily_threshold must be between 1 and 1000, got {value}")
        return value

    @validator("total_records")
    def validate_total_records(cls, value):
        if not (1e4 <= value <= 1e8):
            raise ValueError(f"total_records must be between 10,000 and 100,000,000, got {value}")
        return value

    @validator("window")
    def validate_window(cls, value):
        """Validate window format and convert to timedelta."""
        try:
            timedelta_obj = pd.to_timedelta(value)
            if timedelta_obj <= timedelta(0):
                raise ValueError("Window must be a positive time interval.")
            return value  # Keep original string format
        except ValueError:
            raise ValueError(f"Invalid window format: '{value}'. Use formats like '35d', '2h', '1w'.")

    def get_timedelta(self) -> timedelta:
        """Returns the window as a timedelta object."""
        return pd.to_timedelta(self.window)


class QuerySentimentTopic(BaseModel):
    token: Optional[str] = None

    text: List[str]  # List of strings, each containing multiple comma-separated queries
    top_n: Optional[int] = 3
    normalize_to_global: Optional[bool] = True
    z_score: Optional[bool] = True
    min_global_row_count: Optional[int] = 100  # min no. of global rows to estimate a chart
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    freq: Optional[str] = '-1h'
    languages: Optional[list] = []
    check_geoh3: Optional[bool] = False
    filter_sample_daily_size: Optional[FilterSampleDailySize] = FilterSampleDailySize()
    adjust_gap: Optional[list] = None  # dates known to contain gaps in data
    
    live: Optional[bool] = True
    max_rows: Optional[int] = 10000
    run_async: Optional[bool] = True
    credentials: Optional[Credentials] = Credentials()

    @validator("text", each_item=True)
    def validate_text(cls, value):
        sub_queries = [q.strip() for q in value.split(",")]
        if len(sub_queries) > 5:
            raise ValueError(
                f"Each query string can contain at most 5 sub-queries, but got {len(sub_queries)}: {value}")
        if len(sub_queries) == 0:
            raise ValueError(f"Each query string must contain at least 1 sub-query, but got 0: {value}")
        return value
