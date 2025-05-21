import requests
import aiohttp
import json
import warnings
import time
from datetime import datetime
from io import StringIO
import pandas as pd
from typing import Optional, List
from scipy.stats import zscore
from adase_api.docs.config import AdaApiConfig
from adase_api.helpers import auth, filter_by_sample_size, adjust_gap, get_rolling_z_score
from adase_api.schemas.sentiment import QuerySentimentTopic, ZScoreWindow
from loguru import logger


def _load_sentiment_topic_one(q: QuerySentimentTopic):
    if not q.token:
        auth_token = auth(q.credentials.username, q.credentials.password)
        q.token = auth_token
    url = f'{AdaApiConfig.HOST_TOPIC}/topic-stats/{q.token}'
    json_payload = {k: v for k, v in json.loads(q.json()).items() if k != 'z_score'}
    try:
        response = requests.post(url, json=json_payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        if q.on_not_found_query.value == 'raise':
            raise e
        elif q.on_not_found_query.value == 'warn':
            warnings.warn(str(e))
        time.sleep(30)
        return

    if 'Internal Server Error' in response.text:
        msg = f"Server Error: {q.text}. Try repeating the query later"
        logger.error(msg)
        if q.on_not_found_query.value == 'raise':
            raise ValueError(msg)
        elif q.on_not_found_query.value == 'warn':
            warnings.warn(msg)
        time.sleep(30)
        return

    json_data = StringIO(response.json())
    df = pd.read_json(json_data)

    df.set_index(pd.DatetimeIndex(pd.to_datetime(df['date_time'], unit='ms'), name='date_time'), inplace=True)
    if q.keep_no_hits_rows:
        df['query'].ffill(inplace=True)
    df = df.set_index(['query'], append=True).drop('date_time', axis=1).groupby(['date_time', 'query']).first().unstack(
        'query')

    logger.info(f"Loaded sentiment data for query: {q.text}")
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
        try:
            ada_tmp_ = ada.xs(search_topic, level=1, axis=1)
        except KeyError:
            logger.warning(f"Search topic '{search_topic}' not found in data.")
            return
        ada_tmp_.columns = pd.MultiIndex.from_tuples([[c, search_topic] for c in ada_tmp_.columns])
        return ada_tmp_

    if weights is None:
        weights = [1] * len(ada.score)

    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]

    weighted_score = (ada.score * normalized_weights).sum(axis=1)
    weighted_coverage = (ada.coverage * normalized_weights).sum(axis=1)

    ada_tmp_ = pd.DataFrame({
        'score': weighted_score,
        'coverage': weighted_coverage
    })

    ada_tmp_.columns = pd.MultiIndex.from_tuples([[c, search_topic] for c in ada_tmp_.columns])
    return ada_tmp_


def check_which_query_found(one_q: QuerySentimentTopic, ada, one_ada_query, weights):
    if len(ada.coverage.columns) != len(one_q.text):
        missing = set(one_q.text) - set(ada.coverage.columns)
        msg = f"Queries not found={missing}. Adjust subquery, remove it or `set on_not_found_query`='ignore'"
        logger.warning(msg)
        if one_q.on_not_found_query.value == 'raise':
            raise ValueError(msg)
        else:
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

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the sentiment topic data, with applied filters and normalization.
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
        logger.info(f"[{dt}] | Query {en + 1}/{len(q.text)} | {one_ada_query} | Rows={len(ada)}")
        ada = average_text_bool_results(ada, one_ada_query, weights)
        if ada is None:
            logger.warning(f'Query not found: {one_ada_query}')
        elif q.query_aliases:
            ada.columns = pd.MultiIndex.from_tuples(
                [(c1, ada.get(c2, alias)) for c1, c2 in ada.columns]
            )

        lada += [ada]

    ada = pd.concat(lada, axis=1).ffill()
    ada.columns.names = ['indicator', 'query']

    logger.info(f"Successfully loaded sentiment data for {len(lada)} queries.")
    return ada
