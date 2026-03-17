import os
import sys
from datetime import datetime, timedelta

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
    def __init__(self, ticker):
        # 각 티커별로 별도의 Config 인스턴스를 생성
        self.config = Config(ticker=ticker)
        self.data_fetcher = DataFetcher(self.config)

    def run(self, visualize=True):
        """✅ 주가 예측 및 리스크 관리 실행 (앙상블 트레이닝 적용 및 반복 학습)"""
        start_time = datetime.now()
        logger.info(f"▶️ 프로세스 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 데이터 수집
        try:
            df = DataFetcher(self.config).fetch_combined_data()
            logger.info(f"✅ 데이터 수집 성공, 데이터 크기: {df.shape}")
        except Exception as e:
            logger.error(f"❌ 데이터 수집 실패: {e}")
            sys.exit(1)

        # 2. 데이터 전처리
        processed_data = FeatureEngineer(self.config).preprocess_data(df)
        X_train, y_train = processed_data["x_train"], processed_data["y_train"]
        X_test, y_test = processed_data["x_test"], processed_data["y_test"]
        scaler, dates = processed_data["scaler"], processed_data["date_index"]

        trainer = Trainer(self.config, X_train, y_train, X_test, y_test)

        target_mape = self.config.mape_threshold
        max_iterations = self.config.max_iterations
        iteration = 0
        current_mape = float('inf')
        best_ensemble_mape = float('inf')
        best_adv_model = None
        best_tft_model = None

        # 3. 앙상블 모델 반복 학습: 목표 MAPE 도달 또는 최대 반복까지
        while iteration < max_iterations and current_mape > target_mape:
            iteration += 1
            logger.info(f"🚀 앙상블 모델 학습 시작 (Iteration {iteration}/{max_iterations})")
            # Advanced와 TFT 모델을 hyperparameter 최적화 후 개별 학습
            adv_model, tft_model, history_adv, history_tft = trainer.train_ensemble_model()
            # 두 모델의 예측을 단순 평균으로 앙상블
            y_pred_ensemble = trainer.ensemble_predict(adv_model, tft_model)
            current_mape = mean_absolute_percentage_error(y_test, y_pred_ensemble)
            mse = mean_squared_error(y_test, y_pred_ensemble)
            rmse = np.sqrt(mse)
            logger.info(
                f"📊 Iteration {iteration}: Ensemble MAPE = {current_mape:.2%}, MSE = {mse:.4f}, RMSE = {rmse:.4f}")

            # 최적 앙상블 업데이트: 현재 앙상블 모델이 이전보다 개선되었으면 저장
            if current_mape < best_ensemble_mape:
                best_ensemble_mape = current_mape
                best_adv_model, best_tft_model = adv_model, tft_model
                logger.info("🏆 개선된 앙상블 모델 업데이트")
            if current_mape <= target_mape:
                logger.info("🏆 목표 MAPE 달성! 학습 종료")
                break
            else:
                logger.info("❌ 목표 MAPE 미달성, 추가 학습 진행...")

        # 4. 최종 앙상블 모델 선택 및 저장
        if best_adv_model is None or best_tft_model is None:
            best_adv_model, best_tft_model = adv_model, tft_model
        trainer.save_model(best_adv_model, model_type="advanced")
        trainer.save_model(best_tft_model, model_type="tft")

        # 최종 앙상블 예측 및 평가
        y_pred_ensemble = trainer.ensemble_predict(best_adv_model, best_tft_model)
        stop_loss, take_profit = RiskManager.stop_loss_take_profit(y_pred_ensemble.flatten(), threshold=0.05)
        final_mape = mean_absolute_percentage_error(y_test, y_pred_ensemble)
        stop_loss_mean = np.mean(stop_loss) if stop_loss is not None else 0
        take_profit_mean = np.mean(take_profit) if take_profit is not None else 0

        logger.info(
            f"🎯 최종 Ensemble MAPE: {final_mape:.2%}, 평균 Stop Loss: {stop_loss_mean:.2f}, 평균 Take Profit: {take_profit_mean:.2f}")

        # 5. 미래 예측 (향후 forecast_steps일 예측: 앙상블 미래 예측)
        dates_test = dates[-len(y_test):]
        forecast_steps = self.config.forecast_steps
        last_sequence = X_test[-1]
        future_forecasts_adv = multi_step_forecast(best_adv_model, last_sequence, forecast_steps, scaler=scaler)
        future_forecasts_tft = multi_step_forecast(best_tft_model, last_sequence, forecast_steps, scaler=scaler)
        future_forecasts = (future_forecasts_adv + future_forecasts_tft) / 2
        future_dates = [dates_test[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]

        # visualize 옵션이 True인 경우에만 시각화 수행
        if visualize:
            plot_path = os.path.join(self.config.get_path("data"),
                                     f"{self.config.ticker}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            Visualizer.plot_predictions(
                dates=dates_test,
                actual=y_test.flatten(),
                predicted=y_pred_ensemble.flatten(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                ticker=self.config.ticker,
                save_path=plot_path,
                future_dates=future_dates,
                future_forecasts=future_forecasts,
                scaler=scaler,
                total_feature_cols=scaler.data_min_.shape[0]
            )

        end_time = datetime.now()
        logger.info(f"▶️ 프로세스 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}, 소요 시간: {end_time - start_time}")

        result = {
            "Ticker": self.config.ticker,
            "MAPE (%)": final_mape * 100,
            "Last Prediction": y_pred_ensemble[-1, 0] if y_pred_ensemble.ndim == 2 else y_pred_ensemble[-1]
        }
        return result


if __name__ == "__main__":
    predictor = SingleStockPredictor(ticker="AMD")
    result = predictor.run()
    print(result)
