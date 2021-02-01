# stock_analysis [KOSPI/KOSDAQ]

This is Stock Analysis Package from Scraping to Prediction. Scraper uses Naver and Daum pages and, ML uses "A deep learning framework for financial time series using stacked autoencoders and long-short term memory" paper for modeling. More detail is in [another my repo](https://github.com/kbj2060/MachineLearningForStudy/tree/master/Papers/A%20deep%20learning%20framework%20for%20financial%20time). This project is planing to combine with [Pytrader](https://github.com/kbj2060/pytrader). So complete project could choose good stocks, buy those and finally sell in good price.

<br/>

## Environments


* OS : Windows
* Lang : Python3
* GUI : PyQt5

<br/>

## How It Works



1. For collecting data, execute stock_analysis/src/scraping/scraping.py.
2. stock_analysis/main.py can analze and predict the price.
3. Finally you can get the result like below in html format.

<br/>

<img src="./image/plot.png" width="700">

<br/>

## Reports

There are analyze, prediction, scraping, application error log in report directory. The result of ML is gonna be display like above in html format.

<br/>

## Config

src/system_trading/PARAMETER.py can control your hyper-parameters of ML model.

``` python
BATCH_SIZE = 7
TIME_STEPS = 15
EPOCH = 10
ITERATIONS = 50
LSTM_UNITS = 64

LEARNING_RATE = 0.0003
DROPOUT_SIZE = 0.3
```

<br/>
