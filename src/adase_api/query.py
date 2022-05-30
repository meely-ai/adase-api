import uuid, requests
from datetime import datetime, timedelta
from functools import reduce
from urllib.parse import quote as quote_url
import pandas as pd
from functools import partial
from multiprocessing import Pool
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


class Explorer:
    @staticmethod
    def auth(username, password):
        return requests.post(AdaApiConfig.AUTH_HOST, data={'username': username, 'password': password}).json()

    @staticmethod
    def query_endpoint(token, query, engine='keyword', freq='-3h',
                       start_date=None, end_date=None, rolling='true',
                       indicators=True, top_hits=100):
        if start_date is not None:
            start_date = quote_url(pd.to_datetime(start_date).isoformat())
        if end_date is not None:
            end_date = quote_url(pd.to_datetime(end_date).isoformat())

        query = quote_url(query)
        url_request = f"{AdaApiConfig.HOST}:{AdaApiConfig.PORT}/{engine}/{query}&token={token}"\
                      f"?freq={freq}&rolling={rolling}&indicators={indicators}&return_top={top_hits}"
        if start_date is not None:
            url_request += f'&start_date={start_date}'
            if end_date is not None:
                url_request += f'&end_date={end_date}'

        response = requests.get(url_request)
        topics_frame = pd.DataFrame(response.json()['data'])
        topics_frame.date_time = pd.DatetimeIndex(topics_frame.date_time.apply(
            lambda dt: datetime.strptime(dt, "%Y%m%d%H")))
        return topics_frame.set_index(['date_time', 'query', 'source'])

    @staticmethod
    def get(topics, engine='topic', process_count=2, indicators=True,
            start_date=None, end_date=None, freq='-3h'):
        def process_query(q_topic):
            topics_frame = Explorer.query_endpoint(auth['access_token'], q_topic,
                                                   engine=engine, freq=freq, indicators=indicators,
                                                   start_date=start_date, end_date=end_date)
            return topics_frame.unstack(1)

        auth = Explorer.auth(AdaApiConfig.USERNAME, AdaApiConfig.PASSWORD)
        frames = apply_multiprocess(process_query, topics.split(','), workers=process_count)

        return reduce(lambda l, r: l.join(r, how='outer'), frames).stack(0)
