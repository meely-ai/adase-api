import requests, asyncio, aiohttp, logging, json
from asgiref import sync
from urllib.parse import quote as quote_url
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
from aiohttp_retry import RetryClient
from adase_api.docs.config import AdaApiConfig
from adase_api.helpers import auth
from adase_api.schemas.sentiment import QuerySentimentTopic


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
    if q.live:  # might be incomplete
        df = df.iloc[:-1, :]
    return df

