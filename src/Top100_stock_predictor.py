import os
import sys
from datetime import datetime
from typing import Optional, Dict

import pandas as pd

from src.config import Config
from src.data_fetcher import DataFetcher
from src.single_stock_predictor import SingleStockPredictor
from src.utils.logger import logger


class Top100EnsembleRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.output_dir = config.get_path("top100")

    @staticmethod
    def _process_ticker(ticker: str) -> Optional[Dict]:
        """단일 종목에 대해 예측 파이프라인을 실행하고 결과를 반환합니다."""
        logger.info(f"처리 중: {ticker}")
        try:
            return SingleStockPredictor(ticker=ticker).run(visualize=True)
        except Exception as e:
            logger.error(f"{ticker} 처리 실패: {e}")
            return None

    def _save_result(self, result: Dict) -> None:
        """예측 결과를 티커별 CSV 파일로 저장합니다."""
        ticker = result.get("Ticker", "UNKNOWN")
        future_dates = result.get("Future Dates", [])
        future_forecasts = result.get("Future Forecast", [])

        df = pd.DataFrame({
            "Ticker": ticker,
            "MAPE (%)": result.get("MAPE (%)", ""),
            "Last Prediction": result.get("Last Prediction", ""),
            "Future Date": future_dates,
            "Forecast Price": future_forecasts,
        })
        df = df[["Ticker", "MAPE (%)", "Last Prediction", "Future Date", "Forecast Price"]]

        filename = os.path.join(
            self.output_dir,
            f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        df.to_csv(filename, index=False)
        logger.info(f"{ticker} 예측 결과 저장: {filename}")

    def run(self) -> None:
        """NASDAQ-100 전체 종목에 대해 예측을 실행하고 결과를 저장합니다."""
        tickers = self.data_fetcher.get_top100_tickers()
        if not tickers:
            logger.error("NASDAQ-100 티커 목록을 불러오지 못했습니다.")
            sys.exit(1)

        logger.info(f"총 {len(tickers)}개 종목 처리 시작")
        for ticker in tickers:
            result = self._process_ticker(ticker)
            if result is not None:
                self._save_result(result)


if __name__ == "__main__":
    config = Config()
    Top100EnsembleRunner(config).run()
