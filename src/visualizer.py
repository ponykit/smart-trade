from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.korean_font import set_korean_font
from src.utils.logger import logger


def _inverse_transform_column(
    values: np.ndarray,
    scaler: MinMaxScaler,
    n_features: int,
    col_idx: int = 0,
) -> np.ndarray:
    """정규화된 단일 컬럼 값을 원래 스케일로 복원합니다."""
    values = np.array(values).reshape(-1, 1)
    dummy = np.zeros((len(values), n_features))
    dummy[:, col_idx] = values[:, 0]
    return scaler.inverse_transform(dummy)[:, col_idx]


class Visualizer:
    @staticmethod
    def plot_predictions(
        dates: pd.Index,
        actual: np.ndarray,
        predicted: np.ndarray,
        stop_loss: Optional[np.ndarray] = None,
        take_profit: Optional[np.ndarray] = None,
        ticker: str = "Stock",
        save_path: Optional[str] = None,
        future_dates: Optional[List] = None,
        future_forecasts: Optional[np.ndarray] = None,
        scaler: Optional[MinMaxScaler] = None,
        total_feature_cols: int = 13,
        close_col_idx: int = 0,
    ) -> None:
        """주가 예측 결과와 미래 예측값을 그래프로 시각화합니다.

        Args:
            dates: 테스트 데이터 날짜 인덱스
            actual: 실제 주가 (정규화 값)
            predicted: 예측 주가 (정규화 값)
            stop_loss: 손절가 배열 (선택)
            take_profit: 익절가 배열 (선택)
            ticker: 종목 코드
            save_path: 그래프 저장 경로 (선택)
            future_dates: 미래 예측 날짜 목록 (선택)
            future_forecasts: 미래 예측 주가 배열 (선택)
            scaler: 역변환용 스케일러 (선택)
            total_feature_cols: 전체 피처 수
            close_col_idx: Close 컬럼 인덱스
        """
        set_korean_font()

        dates = pd.to_datetime(dates)
        if future_dates:
            future_dates = pd.to_datetime(future_dates)

        if scaler is not None:
            actual = _inverse_transform_column(actual, scaler, total_feature_cols, close_col_idx)
            predicted = _inverse_transform_column(predicted, scaler, total_feature_cols, close_col_idx)
            if stop_loss is not None and len(stop_loss) > 0:
                stop_loss = _inverse_transform_column(stop_loss, scaler, total_feature_cols, close_col_idx)
            if take_profit is not None and len(take_profit) > 0:
                take_profit = _inverse_transform_column(take_profit, scaler, total_feature_cols, close_col_idx)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, actual, label="실제 주가", color="steelblue", linewidth=1.5)
        ax.plot(dates, predicted, label="예측 주가", color="tomato", linestyle="--", linewidth=1.5)

        if stop_loss is not None and len(stop_loss) > 0:
            ax.plot(dates, stop_loss, label="Stop Loss (−5%)", color="orange", linestyle=":", linewidth=1)
        if take_profit is not None and len(take_profit) > 0:
            ax.plot(dates, take_profit, label="Take Profit (+5%)", color="green", linestyle=":", linewidth=1)
        if future_dates is not None and future_forecasts is not None:
            ax.plot(future_dates, future_forecasts, label="미래 예측", color="purple", linestyle="-.", linewidth=1.5)
            logger.info(f"미래 예측값 (첫 5개): {future_forecasts[:5]}")

        ax.set_title(f"{ticker} 주가 예측", fontsize=14, fontweight="bold")
        ax.set_xlabel("날짜")
        ax.set_ylabel("가격 (USD)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"그래프 저장 완료: {save_path}")

        plt.close(fig)
