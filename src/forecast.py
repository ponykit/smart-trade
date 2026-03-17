from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model


def multi_step_forecast(
    model: Model,
    last_sequence: np.ndarray,
    forecast_steps: int,
    scaler: Optional[MinMaxScaler] = None,
) -> np.ndarray:
    """학습된 모델로 다중 스텝 자기회귀(autoregressive) 예측을 수행합니다.

    각 스텝에서 직전 예측값을 입력 시퀀스의 마지막 값으로 대체하며 반복 예측합니다.

    Args:
        model: 학습된 Keras 모델
        last_sequence: 마지막 입력 시퀀스 (shape: [seq_length, n_features])
        forecast_steps: 예측할 미래 시점 수
        scaler: 역변환에 사용할 스케일러 (없으면 정규화 값 그대로 반환)

    Returns:
        예측 주가 배열 (shape: [forecast_steps])
    """
    if model is None:
        raise ValueError("model이 None입니다.")
    if last_sequence is None or len(last_sequence) == 0:
        raise ValueError("last_sequence가 비어 있습니다.")

    current_seq = last_sequence.copy()
    forecasts = []

    for _ in range(forecast_steps):
        pred = model.predict(np.expand_dims(current_seq, axis=0), verbose=0)
        next_val = float(pred[0, 0])
        forecasts.append(next_val)

        current_seq[:-1] = current_seq[1:]
        current_seq[-1, 0] = next_val

    forecasts_arr = np.array(forecasts)

    if scaler is not None:
        n_features = last_sequence.shape[1]
        dummy = np.zeros((len(forecasts_arr), n_features))
        dummy[:, 0] = forecasts_arr
        return scaler.inverse_transform(dummy)[:, 0]

    return forecasts_arr
