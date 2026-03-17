import os
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model

from src.config import Config
from src.model_builder import ModelBuilder
from src.risk_manager import RiskManager
from src.utils.logger import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Trainer:
    def __init__(
        self,
        config: Config,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params: Dict = config.model_params.copy()

        model_dir = config.get_path("model_dir")
        self.advanced_model_path = os.path.join(model_dir, f"{config.ticker}_advanced_model.h5")
        self.tft_model_path = os.path.join(model_dir, f"{config.ticker}_tft_model.h5")

    @property
    def _input_shape(self) -> Tuple[int, int]:
        return self.X_train.shape[1], self.X_train.shape[2]

    @property
    def _default_callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]

    def save_model(self, model: Model, model_type: str = "advanced") -> None:
        """모델을 지정된 경로에 저장합니다."""
        path = self.advanced_model_path if model_type == "advanced" else self.tft_model_path
        model.save(path)
        logger.info(f"{model_type.upper()} 모델 저장 완료: {path}")

    def load_best_model(self, model_type: str = "advanced") -> Optional[Model]:
        """저장된 모델을 로드합니다. 없으면 None을 반환합니다."""
        path = self.advanced_model_path if model_type == "advanced" else self.tft_model_path
        if not os.path.exists(path):
            logger.warning(f"저장된 {model_type.upper()} 모델이 없습니다. 새로 학습합니다.")
            return None
        try:
            model = load_model(path)
            logger.info(f"{model_type.upper()} 모델 로드 완료: {path}")
            return model
        except Exception as e:
            logger.error(f"{model_type.upper()} 모델 로드 실패: {e}")
            return None

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna 최적화 목적 함수: MAPE + VaR + CVaR + RMSE - Sharpe를 최소화합니다."""
        lstm_units = trial.suggest_int("lstm_units", 64, 256, step=32)
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", self.config.optuna_search_space["batch_size"])

        model = ModelBuilder.create_advanced_model(
            input_shape=self._input_shape,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=6, min_delta=1e-4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        model.fit(
            self.X_train, self.y_train,
            epochs=self.config.hp_epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=0,
        )
        y_pred = model.predict(self.X_test, verbose=0).clip(0, 1)
        residuals = y_pred.flatten() - self.y_test.flatten()
        var, cvar = RiskManager.calculate_var_cvar(residuals)
        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        sharpe = np.mean(residuals) / (np.std(residuals) + 1e-6)
        return mape + abs(var) + abs(cvar) + rmse - sharpe

    def hyperparameter_optimization(self) -> Dict:
        """Optuna를 사용하여 최적 하이퍼파라미터를 탐색합니다."""
        logger.info(f"하이퍼파라미터 최적화 시작 (trials={self.config.optuna_trials})")
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.config.optuna_trials)

        if study.best_params:
            self.best_params = study.best_params
            logger.info(f"최적 하이퍼파라미터: {self.best_params}")
        return self.best_params

    def _fine_tune_lstm_only(self, model: Model) -> None:
        """LSTM 레이어만 학습 가능하도록 설정합니다 (Fine-Tuning용)."""
        for layer in model.layers:
            layer.trainable = "lstm" in layer.name
        logger.info("Fine-Tuning: LSTM 레이어만 학습 활성화")

    def train_advanced_model(
        self, params: Optional[Dict] = None, pretrained_model: Optional[Model] = None
    ) -> Tuple[Model, object]:
        """Advanced CNN-BiLSTM-Attention 모델을 학습합니다."""
        if pretrained_model is None:
            params = params or self.hyperparameter_optimization()
            model = ModelBuilder.create_advanced_model(
                input_shape=self._input_shape,
                lstm_units=params.get("lstm_units", self.config.model_params["lstm_units"]),
                dropout_rate=params.get("dropout_rate", self.config.model_params["dropout_rate"]),
                learning_rate=params.get("learning_rate", self.config.model_params["learning_rate"]),
            )
            batch_size = params.get("batch_size", self.config.model_params["batch_size"])
            logger.info("Advanced 모델 신규 학습 시작")
        else:
            model = pretrained_model
            batch_size = self.config.model_params["batch_size"]
            self._fine_tune_lstm_only(model)

        history = model.fit(
            self.X_train, self.y_train,
            epochs=self.config.training_epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=self._default_callbacks,
            verbose=1,
        )
        return model, history

    def train_tft_model(
        self, params: Optional[Dict] = None, pretrained_model: Optional[Model] = None
    ) -> Tuple[Model, object]:
        """Temporal Fusion Transformer 모델을 학습합니다."""
        if pretrained_model is None:
            params = params or self.hyperparameter_optimization()
            model = ModelBuilder.create_tft_model(
                input_shape=self._input_shape,
                lstm_units=params.get("lstm_units", self.config.model_params["lstm_units"]),
                dropout_rate=params.get("dropout_rate", self.config.model_params["dropout_rate"]),
                learning_rate=params.get("learning_rate", self.config.model_params["learning_rate"]),
            )
            batch_size = params.get("batch_size", self.config.model_params["batch_size"])
            logger.info("TFT 모델 신규 학습 시작")
        else:
            model = pretrained_model
            batch_size = self.config.model_params["batch_size"]
            self._fine_tune_lstm_only(model)

        history = model.fit(
            self.X_train, self.y_train,
            epochs=self.config.training_epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=self._default_callbacks,
            verbose=1,
        )
        return model, history

    def train_ensemble_model(self) -> Tuple[Model, Model, object, object]:
        """Advanced와 TFT 모델을 하이퍼파라미터 최적화 후 각각 학습합니다."""
        logger.info("앙상블 모델 학습 시작")
        params = self.hyperparameter_optimization()
        adv_model, history_adv = self.train_advanced_model(params)
        tft_model, history_tft = self.train_tft_model(params)
        self.save_model(adv_model, model_type="advanced")
        self.save_model(tft_model, model_type="tft")
        return adv_model, tft_model, history_adv, history_tft

    def ensemble_predict(self, adv_model: Model, tft_model: Model) -> np.ndarray:
        """두 모델의 MAPE 기반 동적 가중치 앙상블 예측을 반환합니다."""
        y_pred_adv = adv_model.predict(self.X_test, verbose=0)
        y_pred_tft = tft_model.predict(self.X_test, verbose=0)

        mape_adv = mean_absolute_percentage_error(self.y_test, y_pred_adv)
        mape_tft = mean_absolute_percentage_error(self.y_test, y_pred_tft)

        inv_adv = 1.0 / (mape_adv + 1e-6)
        inv_tft = 1.0 / (mape_tft + 1e-6)
        total = inv_adv + inv_tft
        weight_adv = inv_adv / total
        weight_tft = inv_tft / total

        logger.info(f"앙상블 가중치 - Advanced: {weight_adv:.2f}, TFT: {weight_tft:.2f}")
        return weight_adv * y_pred_adv + weight_tft * y_pred_tft

    def evaluate_ensemble(self, y_pred_ensemble: np.ndarray) -> Tuple[float, float, float]:
        """앙상블 예측 결과의 MAPE, MSE, RMSE를 계산하고 로깅합니다."""
        mape = mean_absolute_percentage_error(self.y_test, y_pred_ensemble)
        mse = mean_squared_error(self.y_test, y_pred_ensemble)
        rmse = np.sqrt(mse)
        var, cvar = RiskManager.calculate_var_cvar(y_pred_ensemble.flatten() - self.y_test.flatten())
        sharpe = np.mean(y_pred_ensemble - self.y_test) / (np.std(y_pred_ensemble - self.y_test) + 1e-6)
        logger.info(
            f"앙상블 평가 - MAPE: {mape:.2%}, RMSE: {rmse:.4f}, VaR: {var:.4f}, CVaR: {cvar:.4f}, Sharpe: {sharpe:.4f}"
        )
        return mape, mse, rmse
