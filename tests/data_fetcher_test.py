import os
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from src.config import Config
from src.utils.logger import logger


class DataFetcher:
    def __init__(self, config: Config):
        """📌 설정을 활용하는 DataFetcher"""
        self.config = config
        self.fred_api_key = config.get_api_key("fred_api")
        self.fmp_api = config.get_api_key("fmp_api")
        self.data_dir = config.get_path("data")
        self.current_date = datetime.today().strftime('%Y-%m-%d')

    def fetch_data(self, filename_prefix, fetch_function):
        """📂 데이터 캐싱 및 수집"""
        filename = os.path.join(self.data_dir, f"{filename_prefix}_{self.current_date}.csv")

        if os.path.exists(filename):
            logger.info(f"📂 {filename}에서 캐시된 데이터를 불러옵니다.")
            return pd.read_csv(filename, index_col=0, parse_dates=True)

        logger.info(f"🔄 {filename_prefix} 데이터를 가져오는 중...")
        df = fetch_function()
        if df.empty:
            logger.warning(f"⚠️ {filename_prefix} 데이터를 찾을 수 없습니다.")
            return df

        df.to_csv(filename, index=True)
        logger.info(f"✅ {filename_prefix} 데이터를 {filename}에 저장했습니다.")
        return df

    def fetch_stock_data(self):
        """📈 주식 데이터 가져오기"""
        return self.fetch_data(
            f"{self.config.ticker}_{self.config.period}_{self.config.interval}",
            lambda: yf.Ticker(self.config.ticker).history(period=self.config.period, interval=self.config.interval)
        )

    def fetch_economic_data(self, start_year=2000, end_year=2025):
        """📊 경제 지표 가져오기"""

        def get_economic_data():
            fred_series = {
                "CPI": "CPIAUCSL",
                "PPI": "PPIACO",
                "GDP_Growth": "A191RL1Q225SBEA",
                "Unemployment_Rate": "UNRATE",
                "Interest_Rate": "FEDFUNDS",
            }
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            data = {}
            for key, series_id in fred_series.items():
                params = {
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "observation_start": f"{start_year}-01-01",
                    "observation_end": f"{end_year}-12-31",
                    "frequency": "a",
                }
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    observations = response.json().get("observations", [])
                    data[key] = {
                        obs["date"]: float(obs["value"]) if obs["value"] not in ["", ".", "NaN", "null"] else None for
                        obs in observations
                    }
            return pd.DataFrame(data).dropna()

        return self.fetch_data(f"economic_data_{start_year}_{end_year}", get_economic_data)

    def fetch_market_data(self):
        """📉 시장 데이터 가져오기"""
        return self.fetch_data("market_data", lambda: pd.DataFrame({
            ticker: yf.Ticker(ticker).history(period=self.config.period, interval=self.config.interval)["Close"]
            for ticker in ["^GSPC", "^VIX", "GC=F", "CL=F"]
        }).ffill())

    def fetch_fundamental_data(self):
        """📑 기업 재무 데이터 가져오기"""

        def get_fundamental_data():
            base_url = "https://financialmodelingprep.com/api/v3"
            endpoint = f"{base_url}/income-statement/{self.config.ticker}"
            params = {"apikey": self.fmp_api, "period": "year", "limit": 5}
            response = requests.get(endpoint, params=params)
            if response.status_code != 200 or not response.json():
                return pd.DataFrame()
            df = pd.DataFrame(response.json())
            df["acceptedDate"] = pd.to_datetime(df["acceptedDate"])
            df.set_index("acceptedDate", inplace=True)
            df.sort_index(inplace=True)
            return df

        return self.fetch_data(f"fundamental_data_{self.config.ticker}", get_fundamental_data)

    def fetch_combined_data(self):
        """📌 주가 + 경제지표 + 시장지표 + 재무제표(분기) 종합"""
        df_stock = self.fetch_stock_data()
        df_stock.index = pd.to_datetime(df_stock.index, utc=True).tz_convert(None)

        df_econ = self.fetch_economic_data()
        df_econ.index = pd.to_datetime(df_econ.index, utc=True).tz_convert(None)
        start_date, end_date = df_stock.index.min(), df_stock.index.max()
        df_econ = df_econ.loc[start_date:end_date].resample('D').ffill()

        df_market = self.fetch_market_data()
        df_market.index = pd.to_datetime(df_market.index, utc=True).tz_convert(None)
        df_market = df_market.loc[start_date:end_date]

        df_fund = self.fetch_fundamental_data()
        df_fund.index = pd.to_datetime(df_fund.index, utc=True).tz_convert(None)
        df_fund = df_fund.loc[start_date:end_date].resample('D').ffill()

        df_combined = pd.concat([df_stock, df_econ, df_market, df_fund], axis=1)
        df_combined.ffill(inplace=True)

        return df_combined


if __name__ == "__main__":
    config = Config()
    data_fetcher = DataFetcher(config)

    df_combined = data_fetcher.fetch_combined_data()
    print(df_combined.head())
