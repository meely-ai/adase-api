import json
from datetime import datetime
from functools import reduce
import pandas as pd
from adase_api.helpers import query_api
from adase_api.schemas.geo import QueryTagGeo, GeoH3Interface, QueryTextMobility, QueryMobility
from adase_api.docs.config import AdaApiConfig, GeoH3Config


def process_mobility_api(resp_content):
    """Parse /get-mobility response content to pandas.DataFrame"""
    mobility_index = pd.DataFrame(
        json.loads(resp_content)).rename(columns={'0': 'km_min'})
    mobility_index.date_time = mobility_index.date_time.astype(str).map(
        lambda dt: datetime.utcfromtimestamp(int(dt[:-3])))

    return mobility_index.set_index(['date_time', 'geoh3']).unstack('geoh3')


def decode_geoh3(df, min_density=0.03):
    """Decode geoh3 using GeoH3Interface to `gps_coord`, `polygon`, `airport_code` or `geonamid`"""
    ld = []
    for h3_res in range(*GeoH3Config.MOBILITY_RES_RANGE):
        query_decode = GeoH3Interface(queries=list(df.columns),
                                      h3_res=h3_res, min_density=min_density, encoder='geonamid')
        ld += [query_api(query_decode.dict(), f"https://{AdaApiConfig.HOST_GEO}", endpoint='decode')]
    decoded_geoh3 = reduce(lambda l, r: {**l, **r}, ld)
    df.columns = df.columns.map(decoded_geoh3).fillna('avg')
    return df


def load_mobility_by_text(q: QueryTextMobility):
    mobility = query_api(q.dict(), f"https://{AdaApiConfig.HOST_GEO}", endpoint='get-mobility-by-text')
    mobility = process_mobility_api(mobility).km_min.interpolate(method='linear')
    return decode_geoh3(mobility)
