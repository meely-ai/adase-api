import os


class AdaApiConfig:
    AUTH_HOST = os.environ.get('AUTH_API_HOST', "https://adalytica.io/user-identity/auth")
    HOST = os.environ.get('ADA_API_HOST', "http://adalytica.io")
    PORT = os.environ.get('ADA_API_PORT', "80")
    USERNAME = os.environ.get('ADA_API_USERNAME', "")
    PASSWORD = os.environ.get('ADA_API_PASSWORD', "")
    DEFAULT_DAYS_BACK = int(os.environ.get('DEFAULT_DAYS_BACK', "243"))
