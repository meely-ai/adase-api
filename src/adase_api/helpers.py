import json, requests
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime
from functools import reduce
import pandas as pd
from adase_api.schemas.geo import GeoH3Interface, QueryStationData, QueryTextMobility, \
    QueryTagGeo, Credentials, QueryMobility
from adase_api.docs.config import AdaApiConfig


def query_api(query, url, endpoint='encode'):
    s = requests.Session()
    retries = Retry(total=8, backoff_factor=0.5)
    s.mount('http://', HTTPAdapter(max_retries=retries))
    resp = s.post(f"{url}/{endpoint}", json=query)
    return json.loads(resp.content)


def process_mobility_api(resp_content):
    """Parse /get-mobility response content to pandas.DataFrame"""
    mobility_index = pd.DataFrame(
        json.loads(resp_content)).rename(columns={'0': 'km_min'})
    mobility_index.date_time = mobility_index.date_time.astype(str).map(
        lambda dt: datetime.utcfromtimestamp(int(dt[:-3])))

    return mobility_index.set_index(['date_time', 'geoh3']).unstack('geoh3')


def decode_geoh3(token, df, min_density=0.03, url=AdaApiConfig.HOST_GEO, port=None):
    """Decode geoh3 using GeoH3Interface to `gps_coord`, `polygon`, `airport_code` or `geonamid`"""
    if port is not None:
        url += f":{port}"
    ld = []
    for h3_res in range(*AdaApiConfig.GEO_H3_MOBILITY_RESOLUTION_RANGE):
        query_decode = GeoH3Interface(token=token, queries=list(df.columns),
                                      h3_res=h3_res, min_density=min_density, encoder='geonamid')
        ld += [query_api(query_decode.dict(), url, endpoint='decode')]
    decoded_geoh3 = reduce(lambda l, r: {**l, **r}, ld)
    df.columns = df.columns.map(decoded_geoh3).fillna('avg')
    return df


def auth(username, password):
    auth_resp = requests.post(AdaApiConfig.AUTH_HOST, data={'username': username, 'password': password}).json()
    try:
        return auth_resp['access_token']
    except KeyError:
        raise AssertionError(f"Anauthenticated, provided username: `{username}`")


def sentiment_by_geography_to_frame(ada_geoh3):
    frame = pd.DataFrame(ada_geoh3['data'])
    frame.date_time = pd.DatetimeIndex(frame.date_time.apply(
        lambda dt: datetime.strptime(dt, "%Y%m%d" if len(dt) == 8 else "%Y%m%d%H")))
    return frame.set_index("date_time")


def read_geoh3_station(q: QueryStationData, geoh3_airport_code, geoh3_dict):
    geoh3_airport_data = pd.read_json(query_api(json.loads(q.json()), AdaApiConfig.HOST_GEO, endpoint='get-station'))
    geoh3_airport_data['geoh3'] = geoh3_airport_data.index.map(
        {code: geoh3 for geoh3, code in geoh3_airport_code.items()})
    geoh3_airport_data['density'] = geoh3_airport_data['geoh3'].map(geoh3_dict)
    return geoh3_airport_data


def load_one_mobility(text: list, credentials: Credentials, aggregated=True):
    q = QueryTextMobility(
        credentials=credentials,
        tag_geo=QueryTagGeo(text=text),
        geo_h3_interface=GeoH3Interface(),
        mobility=QueryMobility(aggregated=aggregated)
    )
    if not q.token:
        auth_token = auth(q.credentials.username, q.credentials.password)
        q.token = auth_token
        q.tag_geo.token = auth_token
    # TODO: make it parse list of strings instead of a single str
    mobility = query_api(q.dict(), AdaApiConfig.HOST_GEO, endpoint='get-mobility-by-text')
    return process_mobility_api(mobility).km_min.interpolate(method='linear')
