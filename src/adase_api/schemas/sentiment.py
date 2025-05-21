import os
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Union, Dict
from pydantic import BaseModel, Field, validator, ValidationError, root_validator


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


class ZScoreWindow(BaseModel):
    window: Optional[Union[int, pd.Timedelta, str]] = 14  # window can be int, Timedelta, or a string


class OnNotFoundQuery(str, Enum):
    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"


class Subquery(BaseModel):
    query: str  # The subquery string
    weight: Optional[float] = 1.0  # Weight for the subquery, defaults to 1 if not specified


class QuerySentimentTopic(BaseModel):
    """
    Parameters for querying sentiment topics through the ADASE API.

    Attributes:
    ----------
    token : Optional[str]
        Authorization token for the request. Either `token` or `credentials` must be provided.

    text : List[str]
        List of query strings, where each string can contain multiple comma-separated sub-queries.
        Each query string must contain between 1 and 5 sub-queries.

    normalize_to_global : Optional[bool]
        Whether to normalize results to global data (default is True).

    normalize_score_to_global : Optional[bool]
        Whether to normalize sentiment scores to global data (default is True).

    z_score : Optional[Union[bool, ZScoreWindow]]
        Whether to apply Z-score normalization. Can be a boolean or a `ZScoreWindow` object (default is False).

    min_global_row_count : Optional[int]
        Minimum number of global rows required to estimate a chart (default is 100).

    start_date : Optional[datetime]
        The start date for the query period.

    end_date : Optional[datetime]
        The end date for the query period.

    freq : Optional[str]
        The frequency for time-series data aggregation (default is '-1h'). Must be a negative value and at least -30 minutes.

    languages : Optional[List[str]]
        List of languages to filter the query by (default is an empty list). TODO: Add coverage regions.

    check_geoh3 : Optional[bool]
        Whether to check GeoH3 indexing (default is False).

    filter_sample_daily_size : Optional[FilterSampleDailySize]
        Filter settings for daily sample size (default is `FilterSampleDailySize()`).

    adjust_gap : Optional[List[str]]
        List of dates known to contain gaps in the data. Automatically set to [['2023-11-01']] when `live` is False.

    live : Optional[bool]
        Whether to retrieve live data (default is True).

    max_rows : Optional[int]
        Maximum number of rows to return, randomly down-sampled if needed (default is 10,000). Must be between `min_global_row_count` and 20,000.

    run_async : Optional[bool]
        Whether to execute queries asynchronously (default is True).

    on_not_found_query: Defines the behavior when a query does not return any results.
            - `"raise"`: Raises an exception if no results are found.
            - `"warn"`: Logs a warning if no results are found.
            - `"ignore"`: Silently ignores the case of no results.

    query_aliases : Optional[List[str]]
        Optional list of aliases corresponding to `text`. Must have the same length as `text`.

    credentials : Optional[Credentials]
        Credentials for accessing the API (default is `Credentials()`).

    Validators:
    -----------
    - `text` validator ensures each query contains between 1 and 5 sub-queries.
    - `freq` validator ensures the value is greater than 30 minutes.
    - `max_rows` validator ensures it falls within the allowed range.
    - `query_aliases` validator ensures it matches the length of `text`.
    - `check_token_or_credentials` validator ensures either `token` or `credentials` is provided.
    """
    text: List[str]  # List of strings, each containing multiple comma-separated queries
    query_aliases: Optional[List] = None
    weights: Optional[List[List[float]]] = None  # List of lists for weights
    live: Optional[bool] = True
    z_score: Optional[Union[bool, ZScoreWindow]] = False
    start_date: Optional[pd.Timestamp] = Field(default_factory=lambda: datetime.utcnow() - timedelta(days=35))
    end_date: Optional[pd.Timestamp]
    freq: Optional[str] = '3h'
    filter_sample_daily_size: Optional[FilterSampleDailySize] = FilterSampleDailySize()
    normalize_to_global: Optional[bool] = True
    normalize_score_to_global: Optional[bool] = True
    min_global_row_count: Optional[int] = 100  # min no. of global rows to estimate a chart, and what about emerging?
    max_rows: Optional[int] = 10000  # random down-sample
    keep_no_hits_rows: Optional[bool] = False  # keep or remove rows with no hits
    adjust_gap: Optional[Union[List[str], None]] = None  # dates (as string) known to contain gaps in data
    languages: Optional[list] = []  # TODO: add coverage regions
    extra_wait_time_pct: Optional[float] = 0.2
    wait_for_all: Optional[bool] = False

    run_async: Optional[bool] = True  # each query in parallel
    on_not_found_query: Optional[OnNotFoundQuery] = OnNotFoundQuery.RAISE
    credentials: Optional[Credentials] = Credentials()
    token: Optional[str] = None  # authorise request

    @root_validator
    def check_token_or_credentials(cls, values):
        token = values.get("token")
        credentials = values.get("credentials")

        if not token and not credentials:
            raise ValueError("Either 'token' or 'credentials' must be provided.")

        return values

    @validator("text", each_item=True)
    def validate_text(cls, value):
        sub_queries = [q.strip() for q in value.split(",")]
        if len(sub_queries) > 5:
            raise ValueError(
                f"Each query string can contain at most 5 sub-queries, but got {len(sub_queries)}: {value}")
        if len(sub_queries) == 0:
            raise ValueError(f"Each query string must contain at least 1 sub-query, but got 0: {value}")
        return value

    @validator('start_date', 'end_date', pre=True, always=True)
    def parse_dates(cls, v):
        if isinstance(v, str):
            try:
                return pd.to_datetime(v)  # Parse pandas datetime string
            except ValueError:
                raise ValueError("Invalid date format. Please use a valid pandas datetime string.")
        elif isinstance(v, datetime):
            return pd.Timestamp(v)
        return v  # Return as is if it's already pd.Timestamp or None.

    @validator("freq")
    def validate_freq(cls, value):
        """
        Ensure the frequency is a negative value and less than -30 minutes.
        """
        try:
            # Convert freq string to Timedelta object
            delta = pd.to_timedelta(value)

            # Check if it's less than 15 minutes
            if delta < timedelta(minutes=15):
                raise ValueError(f"Frequency must be a value greater than 15 minutes, got '{value}'.")

        except ValueError:
            raise ValueError(f"Invalid frequency format: '{value}'. Must be a valid timedelta format.")

        return value

    @validator('freq')
    def check_freq(cls, v, values):
        if not values.get('live'):
            # Convert freq to a pandas Timedelta to compare
            try:
                freq_timedelta = pd.to_timedelta(v)
            except ValueError:
                raise ValueError("Invalid frequency format. Use a valid pandas timedelta string.")

            # Check if freq is less than 12 hours
            if freq_timedelta < pd.Timedelta(hours=12):
                raise ValueError("When live=False, freq must be at least 12 hours.")
        return v

    @root_validator(pre=True)
    def set_adjust_gap_based_on_live(cls, values):
        # Check if 'live' is set in values
        live = values.get('live', True)

        # If live is False, set adjust_gap to [['2023-11-01']], else set it to None
        if not live:
            values['adjust_gap'] = ['2023-11-01']
        else:
            values['adjust_gap'] = None

        return values

    @root_validator
    def validate_max_rows(cls, values):
        min_global_row_count = values.get('min_global_row_count', 100)
        max_rows = values.get('max_rows', 10000)

        # Ensure max_rows is between min_global_row_count and 20000
        if not (min_global_row_count <= max_rows <= 20000):
            raise ValueError(f"max_rows must be between {min_global_row_count} and 20,000, got {max_rows}")

        return values

    @root_validator
    def validate_query_aliases(cls, values):
        """
        Validate that length of `query_aliases` is the same as `text` (queries)
        """
        text = values.get("text")
        query_aliases = values.get("query_aliases")

        if query_aliases and len(text) != len(query_aliases):
            raise ValueError(f"`query_aliases` {len(query_aliases)} different from `text` {len(text)}")
        return values

    @root_validator(pre=True)
    def pad_weights_if_needed(cls, values):
        # Check if weights are provided
        text = values.get('text')
        weights = values.get('weights')

        # If weights are not provided, pad with 1's based on the number of subqueries in each `text`
        if weights is None:
            # Pad weights with 1's for each query string in `text`
            weights = [[1] * len(query.split(',')) for query in text]
            values['weights'] = weights

        # If weights are provided, ensure the number of weights matches the number of subqueries
        elif len(weights) != len(text):
            raise ValueError(f"Number of weights does not match the number of queries in `text`.")

        else:
            # Ensure each list of weights matches the number of subqueries in the corresponding text entry
            for idx, query in enumerate(text):
                subqueries_count = len(query.split(','))
                if len(weights[idx]) != subqueries_count:
                    raise ValueError(f"Number of weights for query '{text[idx]}' does not match number of subqueries.")

        return values

    @root_validator(pre=True)
    def pad_aliases_if_needed(cls, values):
        # Check if query_aliases are provided
        text = values.get('text')
        query_aliases = values.get('query_aliases')

        if query_aliases is None:
            values['query_aliases'] = text
        return values

    @root_validator(pre=True)
    def validate_wait_for_all_and_extra_wait_time(cls, values):
        """
        Validates the relationship between `wait_for_all` and `extra_wait_time_pct`:
        1. If `wait_for_all` is True, `extra_wait_time_pct` must not be provided.
        2. If `extra_wait_time_pct` is provided, `wait_for_all` must be False.
        """
        wait_for_all = values.get('wait_for_all', False)
        extra_wait_time_pct = values.get('extra_wait_time_pct', None)

        if wait_for_all and extra_wait_time_pct is not None:
            raise ValueError("If `wait_for_all` is True, `extra_wait_time_pct` cannot be provided.")

        return values

    @classmethod
    def from_assets(cls, assets: Dict[str, Dict[str, Union[str, List[float]]]], **kwargs):
        text = []
        query_aliases = []
        weights = []

        for alias, asset_data in assets.items():
            queries = asset_data.get("queries", "").split(",")  # Split the query string by commas
            query_weights = asset_data.get("weights", [1] * len(queries))  # Default weight 1 if no weights provided

            # If weights are provided, pair each query with a corresponding weight
            if len(query_weights) != len(queries):
                raise ValueError(f"Number of weights does not match number of queries for alias: {alias}")

            # Add the alias to the query_aliases
            query_aliases.append(alias)

            # Add the query string to the text list
            text.append(",".join(queries))  # Keep the original structure with comma-separated queries

            # Add the list of weights to the weights list
            weights.append(query_weights)

        return cls(text=text, query_aliases=query_aliases, weights=weights, **kwargs)