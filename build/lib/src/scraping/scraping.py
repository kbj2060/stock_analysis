#%%

from src.utils import report_error
import requests
import traceback
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import json
import time
import FinanceDataReader as fdr
import rootpath
import os
from tqdm import tqdm
import random
from multiprocessing import Pool


class Scraping():
    def __init__(self, code, keyword):
        self.code = code
        self.keyword = keyword
        self.daum_headers = {
            'accept': "application/json, text/javascript, */*; q=0.01",
            'accept-encoding': "gzip, deflate, br",
            'accept-language': "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            'cookie': "webid=9f4fbc3082f611eab9a5000af759d440; _TI_NID=iok9mjEFfnWoNqMjDA5hhUChXUxRg7L6mzkKMklqD4+kmSqh6byoM36vbqm6RbiYX91qkV5GBPqt/eECdHNLAQ==; KAKAO_STOCK_CHART_ENABLED_INDICATORS=[%22sma%22%2C%22column%22]; KAKAO_STOCK_VIEW_MODE=pc; _ga=GA1.2.192986296.1590495978; _gid=GA1.2.285685164.1591113605; recentMenus=[{%22destination%22:%22influential_investors%22%2C%22title%22:%22%EC%99%B8%EC%9D%B8%C2%B7%EA%B8%B0%EA%B4%80%22}%2C{%22destination%22:%22chart%22%2C%22title%22:%22%EC%B0%A8%ED%8A%B8%22}%2C{%22destination%22:%22news%22%2C%22title%22:%22%EB%89%B4%EC%8A%A4%C2%B7%EA%B3%B5%EC%8B%9C%22}%2C{%22destination%22:%22talks%22%2C%22title%22:%22%ED%86%A0%EB%A1%A0%22}%2C{%22destination%22:%22analysis%22%2C%22title%22:%22%EA%B8%B0%EC%97%85%EC%A0%95%EB%B3%B4%22}%2C{%22destination%22:%22investments%22%2C%22title%22:%22%ED%88%AC%EC%9E%90%EC%A0%95%EB%B3%B4%22}%2C{%22destination%22:%22current%22%2C%22title%22:%22%ED%98%84%EC%9E%AC%EA%B0%80%22}]; webid_ts=1587382655732; _gat_gtag_UA_128578811_1=1; webid_sync=1591165804480; KAKAO_STOCK_RECENT=[%22A005930%22%2C%22A095570%22%2C%22A008350%22%2C%22A011930%22%2C%22A060310%22%2C%22A035420%22%2C%22A005720%22%2C%22A000660%22%2C%22A338100%22%2C%22A046440%22%2C%22A322780%22%2C%22A003000%22]; _gat_gtag_UA_74989022_11=1; TIARA=1YPVTAMkWXietds3H2uhBIcXtV9iM9jIu8aRwlXydVH1esaY1g45VJMrnxiWzqpX_RZZcIRl4_Vg46jsIoPIII2xMexGzNqu; _dfs=SkJDUTROVUZvYm5SMnYwc2FpNG55VWh2T1Zuc2tTOXp2Z0dnNUhFaTFTWXJTOEZNMVREd1hZZk9CQzk3OU8vTHZYdHQya24rMStacmxlOFhLZDdTNEE9PS0td054YWJScDYvUTZXaTcxY0V0TDQ1QT09--2c58e9a00eb52a126977b566ebf24aaa037cec65",
            'referer': "https://finance.daum.net/quotes/A{code}?view=pc".format(code=self.code),
            'sec-fetch-dest': "empty",
            'sec-fetch-mode': "cors",
            'sec-fetch-site': "same-origin",
            'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
            'x-requested-with': 'XMLHttpRequest'            
            }
        self.naver_url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=self.code)
        self.daum_url = 'https://finance.daum.net/api/investor/days?symbolCode=A{code}&perPage=10&page=1&pagination=true'.format(code=self.code)

    def get_last_page(self, code, url, site, daum_headers=None):
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

    def parse_naver_page(self, code, page):
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

    def parse_daum_page(self, code, page, daum_headers):
        url = 'https://finance.daum.net/api/investor/days?symbolCode=A{code}&perPage=10&page={page}&pagination=true'.format(code=code, page=page)
        res = requests.get(url, headers=self.daum_headers)
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

    def get_data(self, code, end, site, start=1, daum_headers=None):
        if site == "naver":
            for page in range(start, end+1):
                if page == 1:
                    naver_df = self.parse_naver_page(code, page)
                    continue
                try:
                    naver_df = naver_df.append(self.parse_naver_page(code, page))
                    time.sleep(0.1)
                except:
                    continue
            return naver_df
        elif site == "daum":
            for page in range(start, end+1):
                if page == 1:
                    daum_df = self.parse_daum_page(code, page, daum_headers)
                    continue
                df = self.parse_daum_page(code, page, daum_headers)
                if df is None:
                    return pd.DataFrame()
                daum_df = daum_df.append(df)
                time.sleep( random.uniform(0,0.1) )
            return daum_df

#%%

    def make_csv(self, keyword):
        root = rootpath.detect()
        outname = '{keyword}.csv'.format(keyword=keyword)
        outdir = str(root) + '/stock/{keyword}/'.format(keyword=keyword)
        if not os.path.exists(outdir):
            os.makedirs(outdir)        
        fullname = os.path.join(outdir, outname)
        return fullname

    def get_current_data(self, code, data):

        data = data.set_index(data.columns[0]).sort_index(ascending=False)
        last_date = pd.to_datetime(data.index[0])
        df = pd.DataFrame(columns=data.columns)
        for p in range(1, 10):
            daum_parse_data = self.parse_daum_page(code, p, self.daum_headers)
            naver_parse_data = self.parse_naver_page(code, p)
            stacked = pd.concat([naver_parse_data, daum_parse_data], axis=1)
            df = pd.concat([stacked, df]).sort_index(ascending=False)
            if last_date == df.index[0]:
                return pd.DataFrame()
            if True in stacked.index.isin([last_date]):
                df = df[:last_date]
                res = pd.concat([df[:-1], data])
                res.index = pd.to_datetime(res.index, format='%Y-%m-%d', errors='ignore')
                return res.dropna()


    def make_result(self, code, keyword, naver_url, daum_url, daum_headers, batch):
        daum_data = self.get_data(code, batch, "daum", daum_headers=daum_headers)
        if daum_data.empty:
            return None
        naver_data = self.get_data(code, batch, "naver")
        if naver_data.empty:
            return None
        naver_data = naver_data.drop_duplicates()
        daum_data = daum_data.drop_duplicates()
        data = pd.concat([naver_data, daum_data], axis=1)
        data.dropna()
        data.to_csv(self.make_csv(keyword))
        return 1


    def check_search_result(self, keyword, daum_headers):
        url = 'https://finance.daum.net/api/search/quotes?q={keyword}'.format(keyword=keyword)
        res = requests.get(url, headers=daum_headers)
        data = json.loads(res.text)
        if data['quotes'] == None:
            return 0
        return 1


if __name__ == '__main__':    # 프로그램의 시작점일 때만 아래 코드 실행
    df_krx = fdr.StockListing('KRX')
    codes = df_krx['Symbol'] 
    keywords = df_krx['Name']
    root = rootpath.detect()
    for code, keyword in zip(codes,keywords):
        try:
            sc = Scraping(code, keyword)
            data = pd.read_csv(root+'/stock/{0}/{0}.csv'.format(keyword))
            res = sc.get_current_data(code, data)
            if res.empty:
                print('{0} is already updated!'.format(keyword))
                continue
            res = res.loc[~res.index.duplicated(keep='first')]
            res.to_csv(sc.make_csv(keyword))
            print('{0} is completed!'.format(keyword))
        except Exception as e:
            print('SCRAPING ERROR')
            error = '[{0}] {1} Error. {2} \n'.format(datetime.now(), keyword, e)
            report_error(error, 'scraping.err')

