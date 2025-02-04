import os


class AdaApiConfig:
    AUTH_HOST = os.environ.get('AUTH_API_HOST', "https://adalytica.io/user-identity/auth")
    HOST_TOPIC = os.environ.get('ADA_API_TOPIC_HOST', "https://adase-topic-zyz3xdv42q-lm.a.run.app")
    HOST_GEO = os.environ.get('ADA_API_TOPIC_HOST', "https://geo.adalytica.io")
    PORT = os.environ.get('ADA_API_PORT', None)
    USERNAME = os.environ.get('ADA_API_USERNAME', "")
    PASSWORD = os.environ.get('ADA_API_PASSWORD', "")
    DEFAULT_DAYS_BACK = int(os.environ.get('DEFAULT_DAYS_BACK', "183"))
    GEO_H3_MOBILITY_RESOLUTION_RANGE = (1, 5)  # range of supported H3 resolutions


class GeoH3Config:
    CATCHMENT_H3_RES = 2
    FIRST_N_GEOH3_LETTER = 7
    NEWS_RES_RANGE = (2, 4)
    MOBILITY_RES_RANGE = (1, 5)


