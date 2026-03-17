import os
from datetime import datetime
from io import StringIO
from typing import Callable, List, Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from src.config import Config
from src.utils.logger import logger


class DataFetcher:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.ticker = config.ticker
        self.fred_api_key = config.get_api_key("fred_api")
        self.fmp_api = config.get_api_key("fmp_api")
        self.alphavantage_api = config.get_api_key("alphavantage_api")
        self.data_dir = config.get_path("data")
        self.current_date = datetime.today().strftime("%Y-%m-%d")
        self.use_validation = config.use_validation

    def _get_cache_path(self, filename_prefix: str) -> str:
        return os.path.join(self.data_dir, f"{filename_prefix}_{self.current_date}.csv")

    def fetch_data(self, filename_prefix: str, fetch_function: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        """캐시된 데이터가 있으면 로드하고, 없으면 fetch_function을 호출하여 저장합니다."""
        cache_path = self._get_cache_path(filename_prefix)

        if os.path.exists(cache_path):
            logger.info(f"캐시에서 데이터 로드: {cache_path}")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

        logger.info(f"데이터 수집 중: {filename_prefix}")
        df = fetch_function()

        if df is None or df.empty:
            logger.warning(f"수집된 데이터가 비어 있습니다: {filename_prefix}")
            return pd.DataFrame()

        df.to_csv(cache_path, index=True)
        logger.info(f"데이터 저장 완료: {cache_path}")
        return df

    def get_top100_tickers(self) -> List[str]:
        """SlickCharts에서 NASDAQ-100 상위 종목 티커 목록을 반환합니다."""
        def _fetch() -> pd.DataFrame:
            url = "https://www.slickcharts.com/nasdaq100"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if table is None:
                raise ValueError("NASDAQ-100 테이블을 찾을 수 없습니다.")

            df = pd.read_html(StringIO(str(table)))[0]
            if "Symbol" not in df.columns:
                raise ValueError("테이블에 'Symbol' 컬럼이 없습니다.")
            return df

        try:
            df = self.fetch_data("top100_tickers", _fetch)
        except Exception as e:
            logger.error(f"NASDAQ-100 티커 수집 실패: {e}")
            return []

        if df.empty:
            return []

        return df["Symbol"].tolist()[:21]

    def fetch_stock_data(self) -> pd.DataFrame:
        """Yahoo Finance에서 주가 데이터(OHLCV)를 가져옵니다."""
        key = f"{self.config.ticker}_{self.config.period}_{self.config.interval}"

        def _fetch() -> pd.DataFrame:
            return yf.Ticker(self.config.ticker).history(
                period=self.config.period,
                interval=self.config.interval,
            )

        return self.fetch_data(key, _fetch)

    def fetch_economic_data(self, start_year: int = 2000, end_year: int = 2025) -> pd.DataFrame:
        """FRED API에서 거시경제 지표(CPI, PPI, GDP, 실업률, 금리)를 가져옵니다."""
        fred_series = {
            "CPI": "CPIAUCSL",
            "PPI": "PPIACO",
            "GDP_Growth": "A191RL1Q225SBEA",
            "Unemployment_Rate": "UNRATE",
            "Interest_Rate": "FEDFUNDS",
        }

        def _fetch() -> pd.DataFrame:
            if not self.fred_api_key:
                raise ValueError("FRED_API_KEY가 설정되지 않았습니다.")

            base_url = "https://api.stlouisfed.org/fred/series/observations"
            data: dict = {}

            for key, series_id in fred_series.items():
                try:
                    params = {
                        "series_id": series_id,
                        "api_key": self.fred_api_key,
                        "file_type": "json",
                        "observation_start": f"{start_year}-01-01",
                        "observation_end": f"{end_year}-12-31",
                        "frequency": "a",
                    }
                    response = requests.get(base_url, params=params, timeout=10)
                    response.raise_for_status()
                    observations = response.json().get("observations", [])
                    data[key] = {
                        obs["date"]: float(obs["value"])
                        for obs in observations
                        if obs["value"] not in ("", ".", "NaN", "null")
                    }
                except Exception as e:
                    logger.warning(f"FRED 지표 수집 실패 ({series_id}): {e}")

            return pd.DataFrame(data).dropna()

        return self.fetch_data(f"economic_data_{start_year}_{end_year}", _fetch)

    def fetch_market_data(self) -> pd.DataFrame:
        """Yahoo Finance에서 시장 지수(S&P 500, VIX, 금, 원유) 데이터를 가져옵니다."""
        tickers = ["^GSPC", "^VIX", "GC=F", "CL=F"]

        def _fetch() -> pd.DataFrame:
            series = {}
            for t in tickers:
                try:
                    series[t] = yf.Ticker(t).history(
                        period=self.config.period,
                        interval=self.config.interval,
                    )["Close"]
                except Exception as e:
                    logger.warning(f"시장 데이터 수집 실패 ({t}): {e}")
            return pd.DataFrame(series).ffill()

        return self.fetch_data("market_data", _fetch)

    def fetch_fundamental_data(self) -> pd.DataFrame:
        """Financial Modeling Prep API에서 기업 재무제표 데이터를 가져옵니다."""
        def _fetch() -> pd.DataFrame:
            if not self.fmp_api:
                raise ValueError("FMP_API_KEY가 설정되지 않았습니다.")

            url = f"https://financialmodelingprep.com/api/v3/income-statement/{self.config.ticker}"
            params = {"apikey": self.fmp_api, "period": "year", "limit": 5}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["acceptedDate"] = pd.to_datetime(df["acceptedDate"])
            df.set_index("acceptedDate", inplace=True)
            df.sort_index(inplace=True)
            return df

        try:
            return self.fetch_data(f"fundamental_data_{self.config.ticker}", _fetch)
        except Exception as e:
            logger.error(f"재무제표 데이터 수집 실패: {e}")
            return pd.DataFrame()

    def _fetch_alpha_vantage(self, function: str, interval: str = "annual") -> pd.DataFrame:
        """Alpha Vantage API를 호출하여 경제 지표 데이터를 가져옵니다."""
        if not self.alphavantage_api:
            raise ValueError("ALPHAVANTAGE_API_KEY가 설정되지 않았습니다.")

        params = {
            "function": function,
            "apikey": self.alphavantage_api,
            "interval": interval,
            "outputsize": "full",
            "datatype": "csv",
        }
        response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))

    def fetch_unemployment_data(self) -> pd.DataFrame:
        """Alpha Vantage에서 실업률 데이터를 일 단위로 가져옵니다."""
        def _fetch() -> pd.DataFrame:
            df = self._fetch_alpha_vantage("UNEMPLOYMENT")
            df.rename(columns={"timestamp": "date", "value": "unemployment_rate"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.set_index("date", inplace=True)
            return df.resample("D").ffill()

        try:
            return self.fetch_data("unemployment_data", _fetch)
        except Exception as e:
            logger.error(f"실업률 데이터 수집 실패: {e}")
            return pd.DataFrame()

    def fetch_gdp_data(self) -> pd.DataFrame:
        """Alpha Vantage에서 실질 GDP 데이터를 일 단위로 가져옵니다."""
        def _fetch() -> pd.DataFrame:
            df = self._fetch_alpha_vantage("REAL_GDP")
            df.rename(columns={"timestamp": "date", "value": "gdp"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.set_index("date", inplace=True)
            return df.resample("D").ffill()

        try:
            return self.fetch_data("gdp_data", _fetch)
        except Exception as e:
            logger.error(f"GDP 데이터 수집 실패: {e}")
            return pd.DataFrame()

    def fetch_combined_data(self) -> pd.DataFrame:
        """주가, 경제지표, 시장지수, 재무제표 데이터를 병합하여 반환합니다."""
        df_stock = self.fetch_stock_data()
        if df_stock.empty:
            raise ValueError(f"주가 데이터를 수집하지 못했습니다: {self.ticker}")

        df_stock.index = pd.to_datetime(df_stock.index, utc=True).tz_convert(None)
        start_date, end_date = df_stock.index.min(), df_stock.index.max()

        def _align(df: pd.DataFrame, resample: bool = False) -> pd.DataFrame:
            if df.empty:
                return df
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
            df = df.loc[start_date:end_date]
            return df.resample("D").ffill() if resample else df

        def _align_naive(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            df.index = pd.to_datetime(df.index, errors="coerce")
            return df.loc[start_date:end_date]

        df_econ = _align(self.fetch_economic_data(), resample=True)
        df_market = _align(self.fetch_market_data())
        df_fund = _align(self.fetch_fundamental_data(), resample=True)
        df_unemp = _align_naive(self.fetch_unemployment_data())
        df_gdp = _align_naive(self.fetch_gdp_data())

        df_combined = pd.concat(
            [df_stock, df_econ, df_market, df_fund, df_unemp, df_gdp],
            axis=1,
            join="outer",
        )
        df_combined.sort_index(inplace=True)
        df_combined.ffill(inplace=True)

        if self.use_validation:
            df_combined = df_combined[df_combined.index < pd.Timestamp("2025-02-01")]

        logger.info(f"데이터 병합 완료: shape={df_combined.shape}")
        return df_combined
