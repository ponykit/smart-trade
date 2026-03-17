import os
from typing import Any, Optional

from dotenv import load_dotenv

from src.utils.logger import logger

load_dotenv()


class Config:
    def __init__(self, ticker: str = "AMD") -> None:
        self.ticker = ticker
        self.period = "5y"
        self.interval = "1d"
        self.seq_length = 60
        self.scaler_type = "minmax"

        self.use_validation = True

        # 학습 설정
        self.mape_threshold = 0.01
        self.max_iterations = 3
        self.forecast_steps = 60
        self.hp_epochs = 50
        self.training_epochs = 100

        # API 키 (환경 변수에서 로드)
        self.api_keys = {
            "news_api": os.environ.get("NEWS_API_KEY", ""),
            "fmp_api": os.environ.get("FMP_API_KEY", ""),
            "fred_api": os.environ.get("FRED_API_KEY", ""),
            "alphavantage_api": os.environ.get("ALPHAVANTAGE_API_KEY", ""),
        }

        # Optuna 하이퍼파라미터 검색 설정
        self.optuna_trials = 20
        self.optuna_search_space = {
            "lstm_units": (128, 192, 32),
            "dropout_rate": (0.1, 0.4, 0.1),
            "learning_rate": (1e-5, 5e-4),
            "batch_size": [16, 32, 64, 128],
        }

        # 기본 하이퍼파라미터
        self.model_params = {
            "lstm_units": 128,
            "dropout_rate": 0.2,
            "learning_rate": 1e-4,
            "batch_size": 32,
        }

        # 저장 경로 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.paths = {
            "top100": os.path.join(base_dir, "top100"),
            "model_dir": os.path.join(base_dir, "models"),
            "log": os.path.join(base_dir, "logs"),
            "data": os.path.join(base_dir, "data"),
            "visualize": os.path.join(base_dir, "visualize"),
        }

        self._create_directories()

    def _create_directories(self) -> None:
        """설정된 저장 경로의 디렉토리를 생성합니다."""
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        logger.info("저장 경로 초기화 완료.")

    def get_api_key(self, key_name: str) -> Optional[str]:
        """API 키를 반환합니다. 키가 없으면 None을 반환합니다."""
        value = self.api_keys.get(key_name)
        if not value:
            logger.warning(f"API 키 '{key_name}'가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return value or None

    def get_model_params(self) -> dict:
        """모델 하이퍼파라미터를 반환합니다."""
        return self.model_params

    def get_ticker_settings(self) -> dict:
        """주식 종목 관련 설정 값을 반환합니다."""
        return {
            "ticker": self.ticker,
            "period": self.period,
            "interval": self.interval,
            "seq_length": self.seq_length,
        }

    def get_path(self, key: str) -> Optional[str]:
        """저장 경로를 반환합니다."""
        return self.paths.get(key)

    def update_config(self, key: str, value: Any) -> None:
        """설정 값을 업데이트합니다."""
        if hasattr(self, key):
            setattr(self, key, value)
            logger.info(f"설정 업데이트: {key} = {value}")
        else:
            logger.warning(f"존재하지 않는 설정 키: {key}")
