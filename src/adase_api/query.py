import uuid, requests
from datetime import datetime
from functools import reduce
from urllib.parse import quote as quote_url
import pandas as pd
from functools import partial
from billiard import Pool
from adase_api.docs.config import AdaApiConfig


WORKER_ID = uuid.uuid4().hex[:6]


def log(stack_inspect, msg, status, **kwargs):
    dt = datetime.utcnow()
    _time = str(dt)[11:-5]
    try:
        scope = stack_inspect[0][1].split('/')[-1]
        method = stack_inspect[0][3]
    except IndexError:
        scope, method = '', ''
    print(f"[{_time}]-[{scope}]-[{method}]-[{msg}]-[{status}]-[{WORKER_ID}]")


def apply_multiprocess(f, jobs, workers=1, **kwargs):
    if workers == 1 or workers is None:
        return [f(ch, **kwargs) for ch in jobs]
    else:
        with Pool(workers) as pool:
            return list(pool.imap(partial(f, **kwargs), jobs))


def process_query(q_topic, auth=None, engine=None, freq=None,
                  roll_period=None, start_date=None, end_date=None, *args, **kwargs):
    topics_frame = Explorer.query_endpoint(auth['access_token'], q_topic,
                                           engine=engine, freq=freq, roll_period=roll_period,
                                           start_date=start_date, end_date=end_date)
    return topics_frame.unstack(1)


class Explorer:
    @staticmethod
    def auth(username, password):
        return requests.post(AdaApiConfig.AUTH_HOST, data={'username': username, 'password': password}).json()

    @staticmethod
    def query_endpoint(token, query, engine='keyword', freq='-3h',
                       start_date=None, end_date=None,
                       roll_period='7d', top_hits=100):
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

        url_request = f"{host}:{AdaApiConfig.PORT}/{api_path}/{query}&token={token}"\
                      f"?freq={freq}&roll_period={roll_period}&&return_top={top_hits}"
        if start_date is not None:
            url_request += f'&start_date={start_date}'
            if end_date is not None:
                url_request += f'&end_date={end_date}'

        response = requests.get(url_request)
        if response.status_code in range(400, 500):
            raise AttributeError(f"API server error {response.status_code} | {response.json()}")

        topics_frame = pd.DataFrame(response.json()['data'])
        topics_frame.date_time = pd.DatetimeIndex(topics_frame.date_time.apply(
            lambda dt: datetime.strptime(dt, "%Y%m%d%H")))
        return topics_frame.set_index(['date_time', 'query', 'source'])

    @staticmethod
    def get(query, engine='topic', process_count=2, freq='-1h', roll_period='7d',
            start_date=None, end_date=None):
        """
        Query ADASE API to a frame
        :param query: str, syntax varies by engine
            engine='keyword':
                `(+Bitcoin -Luna) OR (+ETH), (+crypto)`
            engine='topic':
                `inflation rates, OPEC cartel`
        :param engine: str,
            `keyword`: boolean operators, more https://solr.apache.org/guide/6_6/the-standard-query-parser.html
            `topic`: plain text, works best with 2-4 words
        :param process_count: int, parallel workers
        :param freq: str, https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        :param roll_period: str, supported
            `7d`, `28d`, `92d`, `365d`
        :param start_date: str
        :param end_date: str
        :return: pd.DataFrame
        """

        auth = Explorer.auth(AdaApiConfig.USERNAME, AdaApiConfig.PASSWORD)
        frames = apply_multiprocess(process_query, query.split(','), workers=process_count,
                                    auth=auth, engine=engine, freq=freq, roll_period=roll_period,
                                    start_date=start_date, end_date=end_date)

        return reduce(lambda l, r: l.join(r, how='outer'), frames).stack(0)
