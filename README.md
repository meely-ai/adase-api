![logo](ADA_logo.png)
## ADA Sentiment Explorer API
### Introduction
Alpha Data Analytics ("ADA") is a data analytics company, core product is ADA Sentiment Explorer (“ADASE”), build on an opinion monitoring technology that intelligently reads news sources and social platforms into machine-readable indicators. It is designed to provide unbiased visibility of people's opinions as a driving force of capital markets, political processes, demand prediction or marketing

ADA's vision is to democratise advanced AI-system supporting decisions, that benefit data proficient people and small- or medium- quantitative institutions.<br><br>
ADASE supports `keyword` and `topic` engines, as explained below
### To install
```commandline
pip install adase-api
```
## Sentiment Open Query
To use API you need to provide API credentials as environment variables and search topic
```python
from adase_api.schemas.sentiment import Credentials, QuerySentimentTopic
from adase_api.sentiment import load_sentiment_topic

credentials = Credentials(username='youruser@gmail.com', password='yourpass')

search_topics = ["inflation rates", "OPEC cartel"]
ada_query = QuerySentimentTopic(
  text=search_topics,
  credentials=credentials
)
sentiment = load_sentiment_topic(ada_query)
sentiment.tail(10)
```
```text
                          score                    coverage                
query               OPEC cartel inflation rates OPEC cartel inflation rates
date_time                                                                  
2024-01-12 03:00:00    0.170492       -3.210051   -0.270801        1.600013
2024-01-12 04:00:00    0.184400       -0.621429   -0.270801        1.600013
2024-01-12 05:00:00    0.170492        0.952482   -0.270801        0.414950
2024-01-12 06:00:00    0.170492       -0.114074   -0.270801        0.414950
2024-01-12 07:00:00    0.170492        0.804350   -0.270801        0.414950
2024-01-12 08:00:00    0.170492        0.241445   -0.270801        1.600013
2024-01-12 09:00:00    0.170492        1.548717   -0.270801        3.970140
```
* Returns `coverage` and `score` (sentiment) to a pandas DataFrame.
* When `normalize_to_global`=True data comes more sparse, since query hits most likely won't be found every hour. 
* In this case missing records, both `coverage` and `score` are filled with 0's
* `coverage` field is usually seasonal, is adviced to apply a 7-day rolling average
* By default, is queried **live** data, that comes on an hourly basis and includes 6 months history

### Search topic syntax
1. **Plain text**
   - In contrast with keyword search, plain text relies on topics to query data on wider concept. It works the best when 2-5 words describe some concepts, examples:
     - `"stock market"`, it might also analyse terms as `"Dow Jones"`, `"FAANG"` etc.
     - `"Airline travel demand"`
     - `"Energy disruptions in Europe"`
     - `"President Joe Biden"`
   - analysed scope depends on how words normally co-occur together
   <br><br>
2. **Boolean search**
   - Search for exact keyword match 
   - Each condition is placed inside of round brackets `()`, where
     - `+` indicates a search term must be found
     - and `-` excludes it
   - For example `"(+Ford +Motor*)`, asterix `*` will include both `Motor` & `Motors`
```python
import pandas as pd

search_topics = ["(+inflation)"]
ada_query = QuerySentimentTopic(
    text=search_topics,
    credentials=credentials,
    languages=['de', 'ro', 'pt', 'pl'],
    live=False,
    start_date=pd.to_datetime('2010-01-01')
)
```
This query will do a boolean search on historical data starting from **Jan 1, 2010** and include only data in specified languages

## Mobility Index
**Monitor traffic (on the road) situation on the city-to-airport pairs**
<br><br>
Besides the news monitoring, the package also provides interface to query worldwide real-time traffic situation.
This can be useful in the combination with media or standalone.   
```python
from adase_api.schemas.geo import QueryTagGeo, GeoH3Interface, QueryTextMobility, QueryMobility
from adase_api.geo import load_mobility_by_text

q = QueryTextMobility(
    credentials=credentials,
    tag_geo=QueryTagGeo(text=['Gdansk']),
    geo_h3_interface=GeoH3Interface(),
    mobility=QueryMobility(aggregated=True)
)
mobility = load_mobility_by_text(q)
```

## API rate limit
All endpoints have set limit on API calls per minute, by default 10 calls  / min.

### In case you don't have yet the credentials, you can [sign up for free](https://adalytica.io/signup)
- Data available since January 1, 2001
- Easy way to explore or backtest
- In a trial version data lags 24-hours
- Probably something else? Hopefully the data can inspire you for other use cases

You can follow us on [LinkedIn](https://www.linkedin.com/company/alpha-data-analytics/)

### Questions?
For package questions, rate limit or feedback you can reach out to info@adalytica.io