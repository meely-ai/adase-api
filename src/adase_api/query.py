import requests, asyncio, aiohttp
from asgiref import sync
from datetime import datetime, timedelta
from functools import reduce
from urllib.parse import quote as quote_url
import pandas as pd
from scipy.stats import zscore
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
                    try:
                        return await response.json()
                    except Exception as exc:
                        print(exc)
                        return
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


def adjust_data_change(df, change_date=pd.to_datetime('2021-08-15'), overlap=timedelta(days=7)):
    before, after = df.loc[(df.index < change_date - overlap)], df.loc[(df.index > change_date + overlap)]

    return zscore(before).append(zscore(after)).clip(lower=-3, upper=3)


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
    elif engine == 'news':
        host = AdaApiConfig.HOST_TOPIC
        api_path = f"topic/rank-news"
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
               bband_period=None, bband_std=2, ta_indicator='coverage', z_score=False,
               normalise_data_split=True):
    """
    Query ADASE API to a frame
    :param normalise_data_split:
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
    queries_split = queries.split(',')
    frames = []
    urls = filter(None, [get_query_urls(auth_resp['access_token'], query, engine=engine, freq=freq,
                                        start_date=start_date, end_date=end_date, roll_period=roll_period,
                                        bband_period=bband_period, bband_std=bband_std, z_score=z_score,
                                        ta_indicator=ta_indicator)
                         for query in queries_split])

    if run_async:
        responses = async_aiohttp_get_all(urls)
    else:
        responses = http_get_all(urls)

    for query, response in zip(queries_split, responses):
        frame = pd.DataFrame(response['data'])
        frame.date_time = pd.DatetimeIndex(frame.date_time.apply(
            lambda dt: datetime.strptime(dt, "%Y%m%d%H")))
        if 'query' not in frame.columns:
            frames += [frame.assign(**{'query': query})]
        else:
            frames += [frame.set_index(['date_time', 'query', 'source']).unstack(1)]

    if engine == 'news':
        return pd.concat(frames)
    resp = reduce(lambda l, r: l.join(r, how='outer'), frames).stack(0)
    if normalise_data_split:
        return adjust_data_change(resp)
