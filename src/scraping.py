#%%

import requests
import traceback
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import json
import time
import FinanceDataReader as fdr
import csv
import os
from tqdm import tqdm
import random
import numpy as np

#%%

# 한국거래소 상장종목 전체
df_krx = fdr.StockListing('KRX')
codes = df_krx['Symbol'] 
keywords = df_krx['Name']
errors = []

#%%

def get_last_page(code, url, site, daum_headers=None):
    if site == "naver":
        res = requests.get(url)
        res.encoding = 'utf-8'
        soap = BeautifulSoup(res.text)
        el_table_navi = soap.find("table", class_="Nnavi")
        el_td_last = el_table_navi.find("td", class_="pgRR")
        pg_last = el_td_last.a.get('href').rsplit('&')[1]
        pg_last = pg_last.split('=')[1]
        pg_last = int(pg_last)
        return pg_last
    elif site == "daum":
        res = requests.get(url, headers=daum_headers)
        data = json.loads(res.text)
        total_pages = data['totalPages']
        return total_pages

#%%

def parse_naver_page(code, page):
    try:
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'.format(code=code, page=page)
        res = requests.get(url)
        _soap = BeautifulSoup(res.text)
        _df = pd.read_html(str(_soap.find("table")), header=0)[0]
        _df = _df.dropna()
        df = _df.drop(['전일비'], axis=1)
        df['날짜'] = pd.to_datetime(df['날짜'], format='%Y-%m-%d')
        df = df.set_index('날짜')
        return df
    except Exception as e:
        traceback.print_exc()
    return None

#%%

def parse_daum_page(code, page, daum_headers):
    url = 'https://finance.daum.net/api/investor/days?symbolCode=A{code}&perPage=10&page={page}&pagination=true'.format(code=code, page=page)
    res = requests.get(url, headers=daum_headers)
    if "<!DOCTYPE html>" in res.text:
        return None
    try:
        data = json.loads(res.text)
    except:
        print(res.text)
    _df = pd.DataFrame.from_dict(data['data'])
    if _df.empty:
        return pd.DataFrame()
    _df = _df.drop(['foreignOwnShares', 'tradePrice', 'accTradePrice','accTradeVolume'], axis=1)
    _df.columns = ['날짜', '외국인보유', '외국인순매수', '기관순매수','기관보유량','전일비','변화']
    _df['개인순매수'] = -(_df['외국인순매수'] + _df['기관순매수'])
    _df = _df[['날짜','개인순매수','외국인순매수','기관순매수','외국인보유','기관보유량','전일비','변화']]
    _df['날짜'] = pd.to_datetime(_df['날짜'], format='%Y-%m-%d', errors='ignore')
    df = _df.set_index('날짜')
    return df

#%%

def get_data(code, end, site, start=1, daum_headers=None):
    if site == "naver":
        for page in range(start, end+1):
            if page == 1:
                naver_df = parse_naver_page(code, page)
                continue
            try:
                naver_df = naver_df.append(parse_naver_page(code, page))
                time.sleep(0.1)
            except:
                continue
        return naver_df
    elif site == "daum":
        for page in range(start, end+1):
            if page == 1:
                daum_df = parse_daum_page(code, page, daum_headers)
                continue
            df = parse_daum_page(code, page, daum_headers)
            if df is None:
                return pd.DataFrame()
            daum_df = daum_df.append(df)
            time.sleep( random.uniform(0,0.1) )
        return daum_df

#%%

def make_csv(keyword):
    outname = '{keyword}.csv'.format(keyword=keyword)
    outdir = '../stock/{keyword}/'.format(keyword=keyword)
    if not os.path.exists(outdir):
        os.makedirs(outdir)        
    fullname = os.path.join(outdir, outname)    
    return fullname

def get_current_data(code, data, daum_headers):
    data = data.set_index('날짜').sort_index(ascending=False)
    last_date = pd.to_datetime(data.index[0])
    df = pd.DataFrame(columns=data.columns)
    for p in range(1, 10):
        daum_parse_data = parse_daum_page(code, p, daum_headers)
        naver_parse_data = parse_naver_page(code, p)
        stacked = pd.concat([naver_parse_data, daum_parse_data], axis=1)
        df = pd.concat([stacked, df])
        if True in stacked.index.isin([last_date]):
            df = df[:last_date]
            res = pd.concat([df[:-1], data])
            res.index = pd.to_datetime(res.index, format='%Y-%m-%d', errors='ignore')
            return res.drop_duplicates()

#%%

def make_result(code, keyword, naver_url, daum_url, daum_headers, batch):
    #naver_last_page = get_last_page(code, naver_url, "naver")
    #daum_last_page = get_last_page(code, daum_url, "daum", daum_headers)
    #print(naver_last_page,daum_last_page)
    daum_data = get_data(code, batch, "daum", daum_headers=daum_headers)
    if daum_data.empty:
        return None
    naver_data = get_data(code, batch, "naver")
    if naver_data.empty:
        return None
    naver_data = naver_data.drop_duplicates()
    daum_data = daum_data.drop_duplicates()
    data = pd.concat([naver_data, daum_data], axis=1)
    data.dropna()
    # start_year = data.index[-1].year
    # start_month = data.index[-1].month
    # now = datetime.today()
    # current_year = now.year
    # current_month = now.month

    #pytrends = TrendReq(hl='KR', tz=540)
    #google_trend = dailydata.get_daily_data(keyword,start_year, start_month, current_year, current_month, geo = 'KR')
    #google_trend = google_trend.rename(columns={keyword : '검색량'})
    
    #res = pd.concat([data, google_trend['검색량']], axis=1)
    #holid_trend = res[res["종가"].isnull()].dropna(axis=1)
    #res = res.dropna()
    
    #holid_trend.to_csv(make_csv('holid_trend', keyword), encoding='utf-8')
    data.to_csv(make_csv(keyword))
    return 1

#%%

def assign_vars(code):
    daum_headers = {
        'accept': "application/json, text/javascript, */*; q=0.01",
        'accept-encoding': "gzip, deflate, br",
        'accept-language': "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        'cookie': "webid=9f4fbc3082f611eab9a5000af759d440; _TI_NID=iok9mjEFfnWoNqMjDA5hhUChXUxRg7L6mzkKMklqD4+kmSqh6byoM36vbqm6RbiYX91qkV5GBPqt/eECdHNLAQ==; KAKAO_STOCK_CHART_ENABLED_INDICATORS=[%22sma%22%2C%22column%22]; KAKAO_STOCK_VIEW_MODE=pc; _ga=GA1.2.366528297.1589976234; _gid=GA1.2.470386811.1589976234; recentMenus=[{%22destination%22:%22influential_investors%22%2C%22title%22:%22%EC%99%B8%EC%9D%B8%C2%B7%EA%B8%B0%EA%B4%80%22}%2C{%22destination%22:%22news%22%2C%22title%22:%22%EB%89%B4%EC%8A%A4%C2%B7%EA%B3%B5%EC%8B%9C%22}%2C{%22destination%22:%22talks%22%2C%22title%22:%22%ED%86%A0%EB%A1%A0%22}%2C{%22destination%22:%22analysis%22%2C%22title%22:%22%EA%B8%B0%EC%97%85%EC%A0%95%EB%B3%B4%22}%2C{%22destination%22:%22investments%22%2C%22title%22:%22%ED%88%AC%EC%9E%90%EC%A0%95%EB%B3%B4%22}%2C{%22destination%22:%22chart%22%2C%22title%22:%22%EC%B0%A8%ED%8A%B8%22}%2C{%22destination%22:%22current%22%2C%22title%22:%22%ED%98%84%EC%9E%AC%EA%B0%80%22}]; webid_sync=1590046704785; _gat_gtag_UA_128578811_1=1; TIARA=gUveTXN62di13WTZ6ubgXq9ugi2XccYvzEq_smTln1lvl6UYoyFlrdh8EzmTwO9HFthCzMY6PO1Hmm8FQ92RuTZEwg8AoSDe; KAKAO_STOCK_RECENT=[%22A000660%22%2C%22A338100%22%2C%22A046440%22%2C%22A322780%22%2C%22A035420%22%2C%22A003000%22]; _gat_gtag_UA_74989022_11=1; _dfs=b0lWTW9qdkcxQmlqbHBla3FiM0NwSDNJSXZyRVpJMU1MTEZobms3ck9MQzRUUnR6VUYrOTBBYVVLQmk2OEFwZit3SVBjYW4vZXRFQlRlcFA5Y1Jkanc9PS0tQzRhNmxhWE82UUNvMDV5RG1EWEgzdz09--472c53167ed367f80b417c7c3c665d2b4ffc2e4e",
        'referer': "https://finance.daum.net/quotes/A{code}?view=pc".format(code=code),
        'sec-fetch-dest': "empty",
        'sec-fetch-mode': "cors",
        'sec-fetch-site': "same-origin",
        'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
        'x-requested-with': 'XMLHttpRequest'            
        }
    naver_url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
    daum_url = 'https://finance.daum.net/api/investor/days?symbolCode=A{code}&perPage=10&page=1&pagination=true'.format(code=code)
    return daum_headers, naver_url, daum_url


#%%

def check_search_result(keyword, daum_headers):
    url = 'https://finance.daum.net/api/search/quotes?q={keyword}'.format(keyword=keyword)
    res = requests.get(url, headers=daum_headers)
    data = json.loads(res.text)
    if data['quotes'] == None:
        return 0
    return 1

#%%
'''
batch = 150
idx = 1
total = len(codes)
for code, keyword in tqdm(zip(codes,keywords), total=len(codes)):
    daum_headers, naver_url, daum_url = assign_vars(code)
    
    if os.path.exists('./{keyword}/'.format(keyword=keyword)) or os.path.exists('./\'{keyword}\'/'.format(keyword=keyword))  :
        continue
    elif code in errors:
        print('{keyword} has error.'.format(keword=keyword))
        os.system('clear')
        continue
    elif not check_search_result(keyword, daum_headers):
        continue
    
    success = make_result(code, keyword, naver_url, daum_url, daum_headers, batch)

    time.sleep( random.uniform(0,1) )
    if success == None:
        print('{code} {keyword} error occured!'.format(code=code, keyword=keyword))
        errors.append(code)

    print('{code} {keyword} is completed...'.format(code=code, keyword=keyword))
#%%
'''
error = []
for code, keyword in zip(codes,keywords):
    if not os.path.exists('../stock/{keyword}/'.format(keyword=keyword)):
        continue
    daum_headers, naver_url, daum_url = assign_vars(code)

    try:
        data = pd.read_csv('../stock/{0}/{0}.csv'.format(keyword))
        res = get_current_data(code, data, daum_headers)
    except:
        error.append(keyword)
        continue
    res.to_csv(make_csv(keyword))
    print('{0} is completed!'.format(keyword))

print(error)
