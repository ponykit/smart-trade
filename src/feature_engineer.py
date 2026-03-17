from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import Config
from src.utils.logger import logger


class FeatureEngineer:
    FEATURE_COLUMNS = [
        # 가격 및 거래량
        "Close", "Open", "High", "Low", "Volume",
        # 기술적 지표
        "5_MA", "20_MA", "60_MA", "Volatility", "RSI", "MACD", "BB_Upper", "BB_Lower",
        # 거시경제 지표
        "GDP_Growth", "Unemployment_Rate", "Interest_Rate",
        # 시장 지수
        "^GSPC", "^VIX", "GC=F", "CL=F",
        # 기업 재무 지표
        "revenue", "operatingIncome", "netIncome", "researchAndDevelopmentExpenses",
    ]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.seq_length = config.seq_length

        if config.scaler_type.lower() == "standard":
            self.scaler: MinMaxScaler | StandardScaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표(이동평균, RSI, MACD, 볼린저 밴드 등)를 추가합니다."""
        logger.info("기술적 지표 계산 중...")

        # 이동평균 및 변동성
        df["5_MA"] = df["Close"].rolling(window=5).mean()
        df["20_MA"] = df["Close"].rolling(window=20).mean()
        df["60_MA"] = df["Close"].rolling(window=60).mean()
        df["Volatility"] = df["Close"].rolling(window=20).std()

        # RSI (Relative Strength Index)
        delta = df["Close"].diff()
        avg_gain = pd.Series(np.where(delta > 0, delta, 0), index=df.index).ewm(span=14, adjust=False).mean()
        avg_loss = pd.Series(np.where(delta < 0, -delta, 0), index=df.index).ewm(span=14, adjust=False).mean()
        df["RSI"] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))

        # MACD (Moving Average Convergence Divergence)
        df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()

        # 볼린저 밴드
        df["BB_Upper"] = df["20_MA"] + df["Volatility"] * 2
        df["BB_Lower"] = df["20_MA"] - df["Volatility"] * 2

        # EMA-50
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

        # ATR (Average True Range)
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
            axis=1,
        )
        df["ATR"] = tr.max(axis=1).rolling(window=14).mean()

        # 스토캐스틱 오실레이터
        low_min = df["Low"].rolling(window=14).min()
        high_max = df["High"].rolling(window=14).max()
        df["Stoch_K"] = (df["Close"] - low_min) / (high_max - low_min + 1e-9) * 100
        df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

        # CCI (Commodity Channel Index)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df["CCI"] = (typical_price - sma_tp) / (0.015 * mad + 1e-9)

        # OBV (On-Balance Volume)
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

        # Williams %R
        df["Williams_%R"] = -100 * (high_max - df["Close"]) / (high_max - low_min + 1e-9)

        df.bfill(inplace=True)
        df.fillna(0, inplace=True)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> Dict:
        """데이터 전처리, 피처 스케일링, 시퀀스 생성을 수행합니다.

        Returns:
            x_train, y_train, x_test, y_test, scaler, date_index를 담은 딕셔너리
        """
        logger.info("데이터 전처리 시작...")

        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

        # 결측치 처리
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        df.fillna(df.select_dtypes(include=["number"]).mean(), inplace=True)

        df = self.add_technical_indicators(df)

        available_features = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        logger.info(f"사용 피처 ({len(available_features)}개): {available_features}")

        x = df[available_features].to_numpy()
        y = df["Close"].shift(-self.seq_length).ffill().to_numpy()
        label_dates = df.index

        train_size = int(len(x) * 0.8)
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_train, dates_test = label_dates[:train_size], label_dates[train_size:]

        logger.info(f"데이터셋 분할 (80:20) - Train: {train_size}, Test: {len(x) - train_size}")

        # 학습 데이터 기준으로 스케일러 fit (데이터 누수 방지)
        self.scaler.fit(x_train)
        scaled_x_train = self.scaler.transform(x_train)
        scaled_x_test = self.scaler.transform(x_test)

        x_train_seq, y_train_seq, dates_train_seq = self._create_sequences(
            scaled_x_train, y_train, dates_train
        )
        x_test_seq, y_test_seq, dates_test_seq = self._create_sequences(
            scaled_x_test, y_test, dates_test
        )

        return {
            "x_train": x_train_seq,
            "y_train": y_train_seq,
            "x_test": x_test_seq,
            "y_test": y_test_seq,
            "scaler": self.scaler,
            "date_index": dates_test_seq,
        }

    def _create_sequences(
        self,
        scaled_data: np.ndarray,
        y_data: np.ndarray,
        dates: pd.Index,
    ):
        """스케일된 데이터로부터 (시퀀스, 레이블, 날짜) 배열을 생성합니다."""
        x_seq, y_seq, date_seq = [], [], []
        for i in range(len(scaled_data) - self.seq_length):
            x_seq.append(scaled_data[i: i + self.seq_length])
            y_seq.append(y_data[i + self.seq_length])
            date_seq.append(dates[i + self.seq_length])
        return np.array(x_seq), np.array(y_seq), np.array(date_seq)
