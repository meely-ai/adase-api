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
## Keyword search engine
### Query syntax
- Each condition is placed inside of round brackets `()`, where
  - `+` indicates a search term must be found
  - and `-` excludes it
- Multiple conditions can be combined with logical operators
  - `OR`
  - `AND`
- Also you can separate by comma "," multiple requests for a parallel processing as below:
  - `"(+Bitcoin -Luna) OR (+ETH), (+crypto)"`
  - Will return matches to data that hit `Bitcoin` or `ETH` but not `Luna` for the first query, and  `crypto` for the second
  - Amount of sub-queries is not limited and is executed in parallel

#### To use API you need to provide API credentials as environment variables
`adase_api.query.load_sentiment` method has more configurations described in the docstring

```python
from adase_api.sentiment import load_sentiment
from adase_api.schemas.sentiment import Credentials
from adase_api.schemas.sentiment import QuerySentimentAPI, ProcessConfig

credentials = Credentials(username='youruser@gmail.com', password='yourpass')

search_keywords = "(+Bitcoin -Luna) OR (+ETH), (+crypto)"  # each query separated by ","
ada_query = QuerySentimentAPI(
  many_query=search_keywords,
  engine='keyword',
  process_cfg=ProcessConfig(roll_period='28d', freq='-1d', z_score=True),
  credentials=credentials,
  run_async=False
)
sentiment = load_sentiment(ada_query)
sentiment.unstack(2).tail()
```
Returns coverage, hits, score and score_coverage to a pandas dataframe
```text
query                      (+Bitcoin -Luna) OR (+ETH)                      (+crypto)                     
                                       coverage       hits     score  coverage       hits     score
date_time           source                                                                         
2022-05-27 11:00:00 all                0.026520  36.676056  0.218439  0.055207  76.487535  0.267412
2022-05-27 12:00:00 all                0.026497  36.668539  0.216516  0.055200  76.518006  0.267331
2022-05-27 13:00:00 all                0.026443  36.616246  0.215001  0.055238  76.554017  0.266730
2022-05-27 14:00:00 all                0.026442  36.605042  0.213506  0.055187  76.481994  0.266553
2022-05-27 15:00:00 all                0.026452  36.647059  0.212794  0.055199  76.512465  0.265416
```
Since data is weekly seasonal, a 7-day rolling average is applied by default

## Topic embedding search engine
### Topic syntax

- In contrast with keyword based search, topic syntax allows to query data in a fuzzy way. It works the best when 2-5 words describe some wider concepts, examples:
  - "NASDAQ technology index"
  - "Airline travel demand"
  - "Energy disruptions in Europe"
- Such queries will include related concept
  - for "NASDAQ technology index" it might also consider terms as "Dow Jones", "FAANG", "FTSE" etc.
  - exact structure depends mostly on how topics co-occur together
  - intuition behind is that NASDAQ is US tech stock index, but if data contains strong signals from FTSE, a British blue chip index, or Dow Jones, less tech heavy index, this will also have an impact on query of interest
  - to reflect changing world situation, underlying models are constantly re-trained making sure relations are up-to-date

```python
from adase_api.sentiment import load_sentiment_topic
from adase_api.schemas.sentiment import QuerySentimentTopic
search_topics = ["inflation rates", "OPEC cartel"]
ada_query = QuerySentimentTopic(
  many_query=search_topics,
  credentials=credentials,
  run_async=False
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
When `normalize_to_global`=True data comes more sparse, since query hits most likely won't be found every hour. 
In this case missing records, both `coverage` and `score` are filled with 0's

## Mobility Index
#### Monitor traffic (on the road) situation on the city-to-airport pairs

```python
from adase_api.schemas.geo import QueryTagGeo, GeoH3Interface, QueryTextMobility, QueryMobility
from adase_api.geo import load_mobility_by_text

q = QueryTextMobility(
    tag_geo=QueryTagGeo(text='Gdansk'),
    geo_h3_interface=GeoH3Interface(),
    mobility=QueryMobility(aggregated=False)
)
mobility = load_mobility_by_text(q)
```
### API rate limit
All endpoints have set limit on API calls per minute, by default 10 calls  / min.

### In case you don't have yet the credentials, you can [sign up for free](https://adalytica.io/signup)
- Data available since January 1, 2001
- Easy way to explore or backtest
- In a trial version data lags 24-hours
- Probably something else? Hopefully the data can inspire you for other use cases

You can follow us on [LinkedIn](https://www.linkedin.com/company/alpha-data-analytics/)

### Questions?
For package questions, rate limit or feedback you can reach out to info@adalytica.io