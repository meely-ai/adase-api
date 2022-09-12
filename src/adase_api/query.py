import requests, asyncio, aiohttp
from asgiref import sync
from datetime import datetime
from functools import reduce
from urllib.parse import quote as quote_url
import pandas as pd
from adase_api.docs.config import AdaApiConfig


def auth(username, password):
    return requests.post(AdaApiConfig.AUTH_HOST, data={'username': username, 'password': password}).json()


def async_aiohttp_get_all(urls):
    """
    Performs asynchronous get requests
    """
    async def get_all(_urls):
        async with aiohttp.ClientSession() as session:
            async def fetch(url):
                async with session.get(url) as response:
                    return await response.json()
            return await asyncio.gather(*[
                fetch(url) for url in _urls
            ])
    # call get_all as a sync function to be used in a sync context
    return sync.async_to_sync(get_all)(urls)


def http_get_all(urls):
    """
    Sequential get requests
    """
    return [requests.get(url).json() for url in urls]


def get_query_urls(token, query, engine='keyword', freq='-3h',
                   start_date=None, end_date=None,
                   roll_period='7d',
                   bband_period='21d',
                   bband_std=2,
                   ta_indicator='coverage',
                   z_score=False):

    if start_date is not None:
        start_date = quote_url(pd.to_datetime(start_date).isoformat())
    if end_date is not None:
        end_date = quote_url(pd.to_datetime(end_date).isoformat())

    query = quote_url(query)
    if engine == 'keyword':
        host = AdaApiConfig.HOST_KEYWORD
        api_path = engine
    elif engine == 'topic':
        host = AdaApiConfig.HOST_TOPIC
        api_path = f"{engine}/{engine}"
    else:
        raise NotImplemented(f"engine={engine} not supported")

    url_request = f"{host}:{AdaApiConfig.PORT}/{api_path}/{query}&token={token}" \
                  f"?freq={freq}&roll_period={roll_period}&"
    if start_date is not None:
        url_request += f'&start_date={start_date}'
        if end_date is not None:
            url_request += f'&end_date={end_date}'

    if bband_period is not None:
        url_request += f'&bband_period={bband_period}&bband_std={bband_std}&ta_indicator={ta_indicator}'

    url_request += f'&z_score={z_score}'
    return url_request


def load_frame(queries, engine='topic', freq='-1h', roll_period='7d',
               start_date=None, end_date=None, run_async=True,
               bband_period=None, bband_std=2, ta_indicator='coverage', z_score=False):
    """
    Query ADASE API to a frame
    :param z_score: bool, data normalisation
    :param ta_indicator: str, feature name to apply technical (chart) analysis
    :param bband_period: str, supported
        `7d`, `14d`, `28d`, `92d`, `365d`
    :param bband_std: float, standard deviation
    :param run_async: bool
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
    auth_resp = auth(AdaApiConfig.USERNAME, AdaApiConfig.PASSWORD)
    frames = []
    urls = filter(None, [get_query_urls(auth_resp['access_token'], query, engine=engine, freq=freq,
                                        start_date=start_date, end_date=end_date, roll_period=roll_period,
                                        bband_period=bband_period, bband_std=bband_std, z_score=z_score,
                                        ta_indicator=ta_indicator)
                         for query in queries.split(',')])

    if run_async:
        responses = async_aiohttp_get_all(urls)
    else:
        responses = http_get_all(urls)

    for response in responses:
        frame = pd.DataFrame(response['data'])
        frame.date_time = pd.DatetimeIndex(frame.date_time.apply(
            lambda dt: datetime.strptime(dt, "%Y%m%d%H")))
        frames += [frame.set_index(['date_time', 'query', 'source']).unstack(1)]
    return reduce(lambda l, r: l.join(r, how='outer'), frames).stack(0)
