import json
from datetime import datetime
from functools import reduce
from itertools import chain
import pandas as pd
from adase_api.helpers import query_api
from adase_api.schemas.geo import GeoH3Interface, QueryTextMobility
from adase_api.docs.config import AdaApiConfig, GeoH3Config
from adase_api.helpers import auth


def process_mobility_api(resp_content, geoh3_columns=None, aggregated=True):
    """Parse /get-mobility response content to pandas.DataFrame"""
    many = []
    for q, one in zip(geoh3_columns, resp_content):
        frame = pd.DataFrame(json.loads(one)).rename(columns={'0': 'km_min'})
        if aggregated:
            frame['geoh3'] = q
        many += [frame]
    mobility_index = pd.concat(many)
    mobility_index.date_time = mobility_index.date_time.astype(str).map(
        lambda dt: datetime.utcfromtimestamp(int(dt[:-3])))
    return mobility_index.groupby(['date_time', 'geoh3']).mean().unstack('geoh3')


def decode_geoh3(df, token, min_density=0.03):
    """Decode geoh3 using GeoH3Interface to `gps_coord`, `polygon`, `airport_code` or `geonamid`"""
    ld = []
    for h3_res in range(*GeoH3Config.MOBILITY_RES_RANGE):
        query_decode = GeoH3Interface(queries=list(df.columns),
                                      h3_res=h3_res, min_density=min_density, encoder='geonamid',
                                      token=token)
        ld += [query_api(query_decode.dict(), AdaApiConfig.HOST_GEO, endpoint='decode')]
    decoded_geoh3 = reduce(lambda l, r: {**l, **r}, chain(*ld))
    df.columns = df.columns.map(decoded_geoh3).fillna('avg')
    return df


def load_mobility_by_text(q: QueryTextMobility):
    if not q.token:
        auth_token = auth(q.credentials.username, q.credentials.password)
        q.token = auth_token
    mobility = query_api(q.dict(), AdaApiConfig.HOST_GEO, endpoint='get-mobility-by-text')
    mobility = process_mobility_api(mobility, geoh3_columns=q.tag_geo.text, aggregated=q.mobility.aggregated
                                    ).km_min.interpolate(method='linear')
    if q.mobility.aggregated:
        mobility.columns = q.tag_geo.text
    elif q.map_geoh3_to_names:
        mobility = decode_geoh3(mobility, q.token)
    return mobility

