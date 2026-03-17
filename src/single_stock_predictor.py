import os
import sys
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from src.config import Config
from src.data_fetcher import DataFetcher
from src.feature_engineer import FeatureEngineer
from src.forecast import multi_step_forecast
from src.risk_manager import RiskManager
from src.trainer import Trainer
from src.utils.logger import logger
from src.visualizer import Visualizer


class SingleStockPredictor:
    def __init__(self, ticker: str) -> None:
        self.config = Config(ticker=ticker)
        self.data_fetcher = DataFetcher(self.config)

    def run(self, visualize: bool = True) -> Dict:
        """주가 예측 파이프라인을 실행합니다.

        1. 데이터 수집 → 2. 전처리 → 3. 앙상블 학습 (반복) → 4. 리스크 계산 → 5. 미래 예측

        Args:
            visualize: True이면 예측 그래프를 저장합니다.

        Returns:
            Ticker, MAPE, 마지막 예측값, 미래 예측 정보를 담은 딕셔너리
        """
        start_time = datetime.now()
        logger.info(f"[{self.config.ticker}] 예측 파이프라인 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 데이터 수집
        try:
            df = self.data_fetcher.fetch_combined_data()
        except Exception as e:
            logger.error(f"데이터 수집 실패: {e}")
            sys.exit(1)

        # 2. 데이터 전처리
        processed = FeatureEngineer(self.config).preprocess_data(df)
        X_train, y_train = processed["x_train"], processed["y_train"]
        X_test, y_test = processed["x_test"], processed["y_test"]
        scaler = processed["scaler"]
        dates = processed["date_index"]

        trainer = Trainer(self.config, X_train, y_train, X_test, y_test)

        # 3. 앙상블 반복 학습 (목표 MAPE 또는 최대 반복 도달까지)
        best_mape = float("inf")
        best_adv_model = None
        best_tft_model = None

        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(f"앙상블 학습 {iteration}/{self.config.max_iterations}")
            adv_model, tft_model, _, _ = trainer.train_ensemble_model()
            y_pred = trainer.ensemble_predict(adv_model, tft_model)
            current_mape = mean_absolute_percentage_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            logger.info(f"Iteration {iteration} - MAPE: {current_mape:.2%}, RMSE: {np.sqrt(mse):.4f}")

            if current_mape < best_mape:
                best_mape = current_mape
                best_adv_model, best_tft_model = adv_model, tft_model

            if current_mape <= self.config.mape_threshold:
                logger.info(f"목표 MAPE({self.config.mape_threshold:.1%}) 달성. 학습 종료.")
                break

        # 4. 최종 모델 저장 및 평가
        trainer.save_model(best_adv_model, model_type="advanced")
        trainer.save_model(best_tft_model, model_type="tft")

        y_pred_final = trainer.ensemble_predict(best_adv_model, best_tft_model)
        final_mape = mean_absolute_percentage_error(y_test, y_pred_final)
        stop_loss, take_profit = RiskManager.stop_loss_take_profit(y_pred_final.flatten())

        logger.info(
            f"최종 MAPE: {final_mape:.2%}, "
            f"평균 Stop Loss: {np.mean(stop_loss):.2f}, "
            f"평균 Take Profit: {np.mean(take_profit):.2f}"
        )

        # 5. 미래 예측
        last_sequence = X_test[-1]
        future_adv = multi_step_forecast(best_adv_model, last_sequence, self.config.forecast_steps, scaler)
        future_tft = multi_step_forecast(best_tft_model, last_sequence, self.config.forecast_steps, scaler)
        future_forecasts = (future_adv + future_tft) / 2

        dates_test = dates[-len(y_test):]
        future_dates = [dates_test[-1] + timedelta(days=i) for i in range(1, self.config.forecast_steps + 1)]

        # 6. 시각화
        if visualize:
            plot_path = os.path.join(
                self.config.get_path("visualize"),
                f"{self.config.ticker}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            Visualizer.plot_predictions(
                dates=dates_test,
                actual=y_test.flatten(),
                predicted=y_pred_final.flatten(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                ticker=self.config.ticker,
                save_path=plot_path,
                future_dates=future_dates,
                future_forecasts=future_forecasts,
                scaler=scaler,
                total_feature_cols=scaler.data_min_.shape[0],
            )

        elapsed = datetime.now() - start_time
        logger.info(f"[{self.config.ticker}] 파이프라인 완료 (소요 시간: {elapsed})")

        return {
            "Ticker": self.config.ticker,
            "MAPE (%)": final_mape * 100,
            "Last Prediction": float(y_pred_final[-1, 0]) if y_pred_final.ndim == 2 else float(y_pred_final[-1]),
            "Future Forecast": future_forecasts,
            "Future Dates": future_dates,
        }


if __name__ == "__main__":
    predictor = SingleStockPredictor(ticker="AMD")
    result = predictor.run()
    print(result)
