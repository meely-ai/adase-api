import requests, aiohttp, json, warnings, time
from requests.exceptions import HTTPError
from datetime import datetime
from io import StringIO
import pandas as pd
from typing import Optional, List
from scipy.stats import zscore
from adase_api.docs.config import AdaApiConfig
from adase_api.helpers import auth, filter_by_sample_size, adjust_gap, get_rolling_z_score
from adase_api.schemas.sentiment import QuerySentimentTopic, ZScoreWindow


def _load_sentiment_topic_one(q: QuerySentimentTopic):
    if not q.token:
        auth_token = auth(q.credentials.username, q.credentials.password)
        q.token = auth_token
    url = f'{AdaApiConfig.HOST_TOPIC}/topic-stats/{q.token}'
    json_payload = {k: v for k, v in json.loads(q.json()).items() if k != 'z_score'}
    response = requests.post(url, json=json_payload)
    if response.status_code != 200:
        msg = response.text
        if q.on_not_found_query.value == 'raise':
            raise HTTPError(msg)
        elif q.on_not_found_query.value == 'warn':
            warnings.warn(msg)
        time.sleep(30)  # server might be temporally unavailable
        return

    if 'Internal Server Error' in response.text:
        msg = f"Server Error {q.text}. Try repeat query later"
        if q.on_not_found_query.value == 'raise':
            raise ValueError(msg)
        elif q.on_not_found_query.value == 'warn':
            warnings.warn(msg)
        time.sleep(30)  # server might be temporally unavailable
        return

    json_data = StringIO(response.json())
    df = pd.read_json(json_data)

    df.set_index(pd.DatetimeIndex(pd.to_datetime(df['date_time'], unit='ms'), name='date_time'), inplace=True)
    df = df.set_index(['query'], append=True).drop('date_time', axis=1
                                                   ).groupby(['date_time', 'query']).first().unstack('query')
    return df


def average_text_bool_results(ada, search_topic, weights: Optional[List[float]] = None):
    """
    Compute the weighted average of score and coverage, if weights are provided.

    Args:
    - ada: The dataset (assumed to have 'score' and 'coverage' as attributes/columns)
    - search_topic: A string representing the search topic for naming columns
    - weights: A list of weights corresponding to each row (query/entry) in ada. If None, no weighting is applied.

    Returns:
    - A DataFrame with the weighted average results.
    """
    if len(weights) == 1:
        ada_tmp_ = ada.xs(search_topic, level=1, axis=1)
        ada_tmp_.columns = pd.MultiIndex.from_tuples([[c, search_topic] for c in ada_tmp_.columns])
        return ada_tmp_

    # If no weights are provided, use equal weighting (1 for each row)
    if weights is None:
        weights = [1] * len(ada.score)  # Equal weights if no specific weights provided
    # Normalize weights to ensure they sum to 1 (optional but commonly done in weighted averaging)
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]

    # Calculate weighted averages for score and coverage
    weighted_score = (ada.score * normalized_weights).sum(axis=1)
    weighted_coverage = (ada.coverage * normalized_weights).sum(axis=1)

    # Combine weighted averages into a DataFrame
    ada_tmp_ = pd.DataFrame({
        'score': weighted_score,
        'coverage': weighted_coverage
    })

    # Join with search_topic for column naming
    ada_tmp_.columns = pd.MultiIndex.from_tuples([[c, search_topic] for c in ada_tmp_.columns])
    return ada_tmp_


def check_which_query_found(one_q: QuerySentimentTopic, ada, one_ada_query, weights):
    if len(ada.coverage.columns) != len(one_q.text):
        missing = set(one_q.text) - set(ada.coverage.columns)
        msg = f"Queries not found={missing}. Adjust subquery, remove it or `set on_not_found_query`='ignore'"
        if one_q.on_not_found_query.value == 'raise':
            raise ValueError(msg)
        else:  # filter weights that are found
            if one_q.on_not_found_query.value == 'warn':
                warnings.warn(msg)
            one_q.text = [q for q in one_q.text if q in ada.score.columns]
            one_ada_query = one_ada_query.replace(','.join(missing), '').strip(', ').strip()
            weights = [w for q, w in zip(one_q.text, weights) if q in ada.score.columns]
    return one_ada_query, weights


def load_sentiment_topic(q: QuerySentimentTopic):
    """
    Queries the ADASE API for sentiment topic data and returns it as a DataFrame.

    Parameters:
    -----------
    q : QuerySentimentTopic
        An instance containing the parameters for the API query.
            from adase_api.schemas.sentiment import QuerySentimentTopic
    Process:
    --------
    - Splits the `text` query strings into individual sub-queries.
    - Applies filters, adjustments, and normalization (e.g., Z-scores) based on the provided parameters.
    - Optionally adjusts for gaps in data and applies daily sample size filtering.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the sentiment topic data, with applied filters and normalization.

    Example:
    --------
    - Input:
        q = QuerySentimentTopic(
            text=["(+Bitcoin -Luna) OR (+ETH)", "(+crypto)"],
            credentials=credentials,
            start_date="2025-01-01"
            z_score='35d'
        )
    - Output:
        A DataFrame with sentiment data, adjusted for gaps and normalized with Z-scores.
    """
    lada = []
    many_ada_queries = q.text
    for en, (one_ada_query, weights, alias) in enumerate(zip(many_ada_queries, q.weights, q.query_aliases)):
        one_q = q.copy()
        one_q.text = [sub_query.strip() for sub_query in one_ada_query.split(",")]

        ada = _load_sentiment_topic_one(one_q)
        if ada is None:
            continue
        ada = filter_by_sample_size(ada, **q.filter_sample_daily_size.dict())
        one_ada_query, weights = check_which_query_found(one_q, ada, one_ada_query, weights)

        if q.adjust_gap:
            ada = adjust_gap(ada)
        if q.z_score:
            if isinstance(q.z_score, ZScoreWindow):
                ada = get_rolling_z_score(ada, q.z_score)
            else:
                ada = zscore(ada)

        dt = datetime.utcnow().strftime('%H:%M:%S')
        print(f"[{dt}] | {en}/{len(q.text)} | {one_ada_query} | rows={len(ada)}")
        ada = average_text_bool_results(ada, one_ada_query, weights)
        if q.query_aliases:
            ada.columns = pd.MultiIndex.from_tuples(
                [(c1, ada.get(c2, alias)) for c1, c2 in ada.columns]
            )

        lada += [ada]
    ada = pd.concat(lada, axis=1).ffill()
    ada.columns.names = ['indicator', 'query']

    return ada

