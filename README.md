![logo](ADA_logo.png)
## ADA Sentiment Explorer API to pandas
### Introduction
Alpha Data Analytics ("ADA") is a data analytics company, flagman product is Alternative Data Sentiment Explorer (“ADASE”), build on a sentiment monitoring technology that turns mentions from 2000+ news sources and social platforms into real-time machine-readable indicators. It is designed to provide an unbiased understanding of people opinions as a major driving force of capital markets, political processes, demand prediction or marketing.  
ADA vision is to democratise advanced AI-system supporting business decisions, that benefit data proficient people and small- or medium- quantitative companies. 
<br><br>
ADASE supports `keyword` and `topic` engines, details below
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
```python
import os
os.environ['ADA_API_USERNAME'] = "myaccount@email.com"
os.environ['ADA_API_PASSWORD'] = "p@ssw0rd"
```
`adase_api.query.Explorer` class has more configurations described in the docstring
```python
from adase_api import query

q = "(+Bitcoin -Luna) OR (+ETH), (+crypto)"
df = query.Explorer.get(q, process_count=1, engine='keyword', start_date='2022-01-01', end_date='2022-05-29')
df.unstack(2).tail()
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
from adase_api import query

q = "inflation rates, OPEC cartel"
df = query.Explorer.get(q, process_count=1, engine='topic', start_date='2022-01-01')
df.unstack(2).tail(10)
```
```text
query                      inflation rates                      OPEC cartel                     
                                  coverage       hits     score    coverage       hits     score
date_time           source                                                                      
2022-05-26 07:00:00 media         0.002947   6.220238 -0.059335    0.001945   5.619048 -0.034639
                    social        0.008054  50.779762  0.023118    0.003774  29.595238  0.022136
2022-05-26 08:00:00 avg           0.004778  24.073413  0.002614    0.002553  15.003968  0.007849
                    corp          0.000297   0.565476  0.054003    0.000384   0.761905  0.050364
                    media         0.002935   6.172619 -0.060830    0.001940   5.595238 -0.034008
                    social        0.008023  50.416667  0.024123    0.003775  29.482143  0.020868
2022-05-26 09:00:00 avg           0.004770  23.942460  0.004983    0.002540  14.908730  0.009729
                    corp          0.000297   0.565476  0.054003    0.000384   0.761905  0.050364
                    media         0.002950   6.125000 -0.057586    0.001922   5.523810 -0.028692
                    social        0.007991  50.202381  0.025980    0.003767  29.363095  0.019497
```
it's visible data feed comes detailed per source type: 
- `media` indicates newspapers, TV, radio and other mass media
- `social` includes social platforms and blogs
- `corp` covers corporate communication as company newsrooms and regulatory filings
- `avg` is a weighted average of all
### In case you don't have yet the credentials, you can [sign up for free](https://adalytica.io/signup)
- Data available since January 1, 2006
- Easy way to explore or backtest
- In a trial version data lags 24-hours
- Probably something else? Hopefully this data could inspire for some innovative solutions to your problem

You can follow us on [LinkedIn](https://www.linkedin.com/company/alpha-data-analytics/) 