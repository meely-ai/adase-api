import requests, asyncio, aiohttp, logging, json
from asgiref import sync
from datetime import datetime, timedelta
from functools import reduce
from urllib.parse import quote as quote_url
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
from aiohttp_retry import RetryClient
from scipy.stats import zscore
from adase_api.docs.config import AdaApiConfig
from adase_api.helpers import auth
from adase_api.schemas.sentiment import QuerySentimentAPI, QuerySentimentTopic


def async_aiohttp_get_all(urls, retry_attempts=3):
    """
    Performs asynchronous get requests
    """
    async def get_all(_urls):
        async with aiohttp.ClientSession() as session:
            retry_session = RetryClient(session, retry_attempts=retry_attempts)

            async def fetch(url):
                async with retry_session.get(url) as response:
                    try:
                        return await response.json()
                    except Exception as exc:
                        return
            return await asyncio.gather(*[
                fetch(url) for url in _urls
            ])
    # call get_all as a sync function to be used in a sync context
    return sync.async_to_sync(get_all)(urls)


def http_get_all(urls, retry_attempts=3, backoff_factor=30, **kwargs):
    """
    Sequential get requests
    """

    def get_http_with_retry(url, method='GET', **kwargs):
        """
        Retry unreliable service with delay between retries
        """
        s = requests.Session()
        retries = Retry(total=retry_attempts,
                        backoff_factor=backoff_factor,
                        status_forcelist=[500, 502, 503, 504, 429, 400])
        s.mount('http://', HTTPAdapter(max_retries=retries))
        resp = s.get(url) if method == 'GET' else s.post(url, **kwargs) if method == 'POST' else {}
        return resp

    return [get_http_with_retry(url, **kwargs) for url in urls]


def adjust_data_change(df, change_date=pd.to_datetime('2021-08-15'), overlap=timedelta(days=7)):
    before, after = df.loc[(df.index < change_date - overlap)], df.loc[(df.index > change_date + overlap)]
    if len(before) == 0:
        return df
    return zscore(before).append(zscore(after)).clip(lower=-3, upper=3)


def get_query_urls(search_text, q: QuerySentimentAPI):

    query = quote_url(search_text)
    if q.engine == 'keyword':
        host = AdaApiConfig.HOST_KEYWORD
        api_path = q.engine
    elif q.engine == 'topic':
        host = AdaApiConfig.HOST_TOPIC
        api_path = 'topic'
    else:
        raise NotImplemented(f"engine={q.engine} not supported")

    url_request = f"{host}:{AdaApiConfig.PORT}/{api_path}/{query}&token={q.token}" \
                  f"?freq={q.process_cfg.freq}&roll_period={q.process_cfg.roll_period}"

    if q.start_date is not None:
        start_date = quote_url(pd.to_datetime(q.start_date).isoformat())
        url_request += f'&start_date={start_date}'
        if q.end_date is not None:
            end_date = quote_url(pd.to_datetime(q.end_date).isoformat())
            url_request += f'&end_date={end_date}'

    if q.bband is not None:
        url_request += f'&bband_period={q.bband.period}&bband_std={q.bband.std}&ta_indicator={q.bband.indicator}'

    url_request += f'&z_score={q.process_cfg.z_score}'
    return url_request


def load_sentiment(q: QuerySentimentAPI, normalise_data_split=False, retry_attempts=3):
    """
    Query ADASE API to a frame
    :param q:
    :param normalise_data_split:
    :param z_score: bool, data normalisation
    :param ta_indicator: str, feature name to apply technical (chart) analysis
    :param bband_period: str, supported
        `7d`, `14d`, `28d`, `92d`, `365d`
    :param bband_std: float, standard deviation
    :param run_async: bool
    :param retry_attempts: int, number of HTTP retries sent to backend
    :param queries:  str, syntax varies by engine
        engine='keyword':
            `(+Bitcoin -Luna) OR (+ETH), (+crypto)`
        engine='topic':
            `inflation rates, OPEC cartel`
    :param engine: str,
        `keyword`: boolean operators, more https://solr.apache.org/guide/6_6/the-standard-query-parser.html
        `topic`: plain text, works best with 2-4 words
    :param freq: str, https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :param roll_period: str, supported
        `7d`, `28d`, `92d`, `365d`
    :param start_date: str
    :param end_date: str
    :return: pd.DataFrame
    """
    if not q.token:
        auth_token = auth(q.credentials.username, q.credentials.password)
        q.token = auth_token

    queries_split = q.many_query.split(',')
    frames = []
    urls = list(filter(None, [get_query_urls(search_text, q) for search_text in queries_split]))
    print(urls)
    if q.run_async:
        responses = async_aiohttp_get_all(urls, retry_attempts=retry_attempts)
    else:
        responses = http_get_all(urls)

    for query, response in zip(queries_split, responses):
        if response is None or 'data' not in response.json().keys():
            logging.warning(f"failed query: {query}")
            reason = json.loads(json.loads(response.content)['detail'])['detail']
            logging.warning(f"HTTP {response.status_code}, {reason}")
            continue
        frame = pd.DataFrame(response.json()['data'])
        frame.date_time = pd.DatetimeIndex(frame.date_time.apply(
            lambda dt: datetime.strptime(dt, "%Y%m%d" if len(dt) == 8 else "%Y%m%d%H")))
        if 'query' not in frame.columns:
            frames += [frame.assign(**{'query': query})]
        else:
            frame = frame.set_index(['date_time', 'query', 'source']).unstack(1)

            if normalise_data_split and q.engine == 'topic' and q.z_score:
                frame = adjust_data_change(frame.unstack(1)).stack()
            frames += [frame]

    if q.engine == 'news':
        return pd.concat(frames)  # assumed one topic (query) at a time

    if len(frames) == 0:
        logging.warning(f"No results")
        return
    resp = reduce(lambda l, r: l.join(r, how='outer'), frames).stack(0)

    return resp


def load_kei(query, thresh=0.2, top_n=30, retry_attempts=3):
    """
    Query economic indicators
    :query: str, one query, `Inflation in Germany`
    :thresh: float, minimal textual similarity
    :top_n: int, number of best hits
    :retry_attempts: int, number of HTTP retries
    :return: pd.DataFrame, economic indicator's ranked by relevance
    """
    auth_token = auth(AdaApiConfig.USERNAME, AdaApiConfig.PASSWORD)
    host = AdaApiConfig.HOST_TOPIC
    api_path = "rank-kei"

    query = quote_url(query)
    url = f"{host}:{AdaApiConfig.PORT}/{api_path}/{query}&token={auth_token}" \
          f"?one_query={query}&top_n={top_n}&thresh={thresh}"

    response = http_get_all([url], retry_attempts=retry_attempts)
    if len(response) == 0:
        logging.warning(f"No any results")
        return

    return pd.read_json(json.dumps(next(iter(response))['data']))


def load_sentiment_topic(q: QuerySentimentTopic):
    """
    Query ADASE API to a frame
    :param q:
    :param normalise_data_split:
    :param z_score: bool, data normalisation
    :param ta_indicator: str, feature name to apply technical (chart) analysis
    :param bband_period: str, supported
        `7d`, `14d`, `28d`, `92d`, `365d`
    :param bband_std: float, standard deviation
    :param run_async: bool
    :param retry_attempts: int, number of HTTP retries sent to backend
    :param queries:  str, syntax varies by engine
        engine='keyword':
            `(+Bitcoin -Luna) OR (+ETH), (+crypto)`
        engine='topic':
            `inflation rates, OPEC cartel`
    :param engine: str,
        `keyword`: boolean operators, more https://solr.apache.org/guide/6_6/the-standard-query-parser.html
        `topic`: plain text, works best with 2-4 words
    :param freq: str, https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :param roll_period: str, supported
        `7d`, `28d`, `92d`, `365d`
    :param start_date: str
    :param end_date: str
    :return: pd.DataFrame
    """
    if not q.token:
        auth_token = auth(q.credentials.username, q.credentials.password)
        q.token = auth_token
    url = f'{AdaApiConfig.HOST_TOPIC}/topic-stats/{q.token}'
    response = requests.post(url, json=json.loads(q.json()))
    df = pd.read_json(response.json())
    df.index = pd.DatetimeIndex(pd.to_datetime(df['index'], unit='ms'), name='date_time')
    df = df.set_index(['query'], append=True).drop("index", axis=1).unstack('query')
    return df.iloc[:-1, :]

