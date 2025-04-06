![logo](https://storage.googleapis.com/adalytica-webpage-assets/article-images/Color_black_horizontal.png)
## Adalytica Sentiment Explorer API
### Introduction
Alpha Data Analytics ("Adalytica") is a data analytics company specializing in sentiment analysis. Its core product, Sentiment Explorer, uses sentiment monitoring technology to analyze news and social media content, transforming it into machine-readable data. The platform provides insights into public sentiment that can be applied across various fields, including capital markets, demand forecasting, political analysis, marketing strategies, and more.
<br>
This package powers the [Fear and Greed Index](https://adalytica.io/stock-fear-and-greed-index), a key tool for assessing market sentiment.
<br>
For a detailed explanation of how it works, read the article: [How Fear and Greed Using Sentiment AI Works?](https://adalytica.io/news/how-fear-and-greed-sentiment-using-ai-works)

This packages is made for data proficient users, like quantitative analysts, data engineers, data scientist or analysts<br><br>
Package is build around two search engines `keyword` and `topic`, as explained below. 
Data contains two indicators: 
- `score` — sentiment polarity score in the range of [-1, +1]
- `coverage` — ratio (proportion) of query hits divided to total number of data
  - naturally, different queries will have varying levels of popularity. To make them comparable over time, z-score normalization is recommended and supported by the package
  - indicator shows for most of the queries a strong weekday seasonality
  
### To install
```commandline
pip install adase-api
```
### Credentials
In case you don't have yet the credentials, you can [Sign Up for free 14 day trial](https://adalytica.io/signup)
## Sentiment Open Query
To use API you need to provide API credentials and search terms
```python
from adase_api.schemas.sentiment import Credentials, QuerySentimentTopic
from adase_api.sentiment import load_sentiment_topic

credentials = Credentials(username='youruser@gmail.com', password='yourpass')

search_terms = ["inflation rates", "bitcoin"]
ada_query = QuerySentimentTopic(
  text=search_terms,
  credentials=credentials
)
sentiment = load_sentiment_topic(ada_query)
sentiment.tail(3)
```
```text
indicator                     score        coverage     score  coverage
query               inflation rates inflation rates   bitcoin   bitcoin
          date_time                                                              
2025-02-04 06:00:00        0.112417        0.002583  0.011417  0.001750
2025-02-04 09:00:00        0.103167        0.002750 -0.155000  0.001500
2025-02-04 12:00:00        0.047500        0.002750 -0.050375  0.003375
```
Returns two indicators, `coverage` and `score`, in a Pandas DataFrame indexed by timestamp objects, with columns organized as a multiindex
### More advanced queries
You can supply a dictionary of query keys, each containing multiple comma-separated sentiment queries along with the corresponding weight for each subquery: 
```python
ASSETS = {
    "NXPI": {
        "queries": "(+NXPI), (+semiconduct*)",
        "weights": [1.0, 0.5]
    },
    "PLTR": {
        "queries": "(+Palantir), (+PLTR)",
        "weights": [2.0, 1.0]
    }
}

ada_query = QuerySentimentTopic.from_assets(
    ASSETS,
    credentials=credentials
)
```
In this case, each search term is queried individually, and then a weighted average is calculated
```text
indicator               score  coverage     score  coverage
query                    NXPI      NXPI      PLTR      PLTR
          date_time                                                  
2025-02-04 06:00:00  0.056472  0.000333 -0.010778  0.000417
2025-02-04 09:00:00  0.051278  0.000194  0.039500  0.000417
2025-02-04 12:00:00  0.016750  0.000042  0.058500  0.000583
```
<br>

You can also apply z-score sentiment normalization. In this example, the z-score is calculated based on a 35-day rolling window.
```python
from adase_api.schemas.sentiment import ZScoreWindow

ada_query = QuerySentimentTopic.from_assets(
    ASSETS,
    credentials=credentials,
    z_score=ZScoreWindow(window='35d')
)
```
By default is queried last 35 days of live data collection, but you can switch to historical data
```python
ada_query = QuerySentimentTopic.from_assets(
    ASSETS,
    credentials=credentials,
    z_score=ZScoreWindow(window='365d'),
    live=False,
    start_date='2020-01-01',
    languages=['de', 'ro'],
    freq='1d',
    on_not_found_query='warn'
)
```
This query will retrieve data from 2020, in German and Romanian languages, apply a 1-year rolling z-score and return a daily data granularity
### More about search query syntax
1. **Plain text**
   - Unlike keyword search, plain text relies on topics to query data based on broader concepts. It works best when 2-5 words describe a particular concept. Examples include:
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

This query will do a boolean search on historical data starting from **Jan 1, 2010** and include only data in specified languages

## Key stats
### Data coverage, missing or insufficient data
- Curated from over 5,000 sources
- Approximately 1,000,000 unique stories daily
- Available in 72 languages; both original and translated versions for querying in base English
- To ensure sufficient data per query, you have the `filter_sample_daily_size` option (refer to the code docstring for more details). If not enough data is found, those rows will be set to NaN. The calculation uses a rolling window, meaning some periods may have sufficient data while others may not.
- If no results are found, you’ll receive a Server HTTP 404 error. For multiple sub-queries, any missing ones will be ignored, and the results will be based on the found sub-queries.
- For more details, refer to the code docstrings

### Data history
- Data available since January 1, 2019
### API rate limit
All endpoints have a set limit on API calls per minute, with a default of 10 calls per minute.

### Chat with data
You can also interact with data using LLM and integrate live news data feeds into your systems, although this is outside the scope of this particular package. 

### Questions?
- For package questions, rate limit or feedback you can reach out to [info@adalytica.io](mailto:info@adalytica.io)
- You can also follow us on [LinkedIn](https://www.linkedin.com/company/alpha-data-analytics/)
- Or check some of [our public research](https://adalytica.io/news) powered by this package data  
- If this feels too complex, there's also a lightweight web app solution that provides access to sentiment data.