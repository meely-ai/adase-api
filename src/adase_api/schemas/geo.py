from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from enum import Enum
from adase_api.schemas.sentiment import Credentials


class QueryTagGeo(BaseModel):
    token: Optional[str] = None
    text: List
    keep_top_n: Optional[int] = 2  # maximal number of geographic matches
    min_avg_score: Optional[float] = 0.12  # value between [0, 1], too small value will lead to too many false positives


class Encoders(str, Enum):
    GPS_COORD = "gps_coord"  # single point
    POLYGON = "polygon"  # arbitrary region defined as GPS coord
    GEONAMID = "geonamid"  # city geonamid
    AIRPORT_CODE = "airport_code"  # IATA airport code
    GEOH3 = 'geoh3'  # h3 hashes


class GeoH3Interface(BaseModel):
    token: Optional[str] = ''
    queries: Optional[list] = []
    encoder: Optional[Encoders] = 'geonamid'
    min_density: Optional[float] = 0.03
    h3_res: Optional[int] = 3
    h3_res_range: Optional[tuple] = (2, 4)
    geonamid_to_name: Optional[bool] = True


class QueryMobility(BaseModel):
    geoh3_dict: Optional[dict] = {}
    aggregated: Optional[bool] = True  # if True, weighted density average across multiple geoh3
    days_back: Optional[int] = 92


class QueryTextDensity(BaseModel):
    token: Optional[str] = None
    tag_geo: QueryTagGeo
    geo_h3_interface: GeoH3Interface


class QueryTextMobility(BaseModel):
    token: Optional[str] = None
    tag_geo: Optional[QueryTagGeo]
    geo_h3_interface: GeoH3Interface
    mobility: QueryMobility
    credentials: Optional[Credentials] = Credentials()
    map_geoh3_to_names: Optional[bool] = True


class QueryStationData(BaseModel):
    token: Optional[str] = None
    queries: list
    encoder: Optional[Encoders] = 'geonamid'


class QueryKeywordGeoH3(BaseModel):
    token: Optional[str] = ''
    many_query: str
    geoh3_dict: dict
    start_date: Optional[datetime] = datetime(2000, 1, 1)
    end_date: Optional[datetime] = None
    freq: Optional[str] = None
    roll_period: Optional[str] = None


class QueryTextSentiment(BaseModel):
    token: Optional[str] = None
    tag_geo: Optional[QueryTagGeo]
    geo_h3_interface: Optional[GeoH3Interface]
    keyword: QueryKeywordGeoH3
    credentials: Optional[Credentials] = Credentials()
