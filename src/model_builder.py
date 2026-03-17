from typing import Tuple

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Add, BatchNormalization, Bidirectional, Conv1D, Dense, Dropout,
    Input, Lambda, LayerNormalization, LSTM, MaxPooling1D,
    MultiHeadAttention, Multiply, Permute, concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW

from src.utils.logger import logger

_L2 = regularizers.l2(1e-4)


class ModelBuilder:
    @staticmethod
    def create_advanced_model(
        input_shape: Tuple[int, int],
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        use_adamw: bool = False,
    ) -> Model:
        """CNN-BiLSTM-Attention 모델을 생성합니다.

        단기 패턴 추출(CNN) → 장기 트렌드 학습(BiLSTM) → 중요 시점 강조(Attention)
        → 변수 선택(VSN) → 잔차 연결(GRN) → 최종 예측 순서로 구성됩니다.
        """
        logger.info(f"Advanced 모델 생성 - lstm_units={lstm_units}, dropout={dropout_rate}, lr={learning_rate}")

        inputs = Input(shape=input_shape)

        # Multi-kernel CNN: 다양한 단기 패턴 동시 포착
        conv_branches = []
        for kernel_size in [2, 3, 5, 7]:
            conv = Conv1D(filters=64, kernel_size=kernel_size, activation="relu", padding="same")(inputs)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv_branches.append(conv)
        cnn_out = concatenate(conv_branches, axis=-1)
        cnn_out = BatchNormalization()(cnn_out)
        cnn_out = Dropout(dropout_rate)(cnn_out)

        # Stacked Bidirectional LSTM: 장기 시계열 의존성 학습
        lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.2))(cnn_out)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(dropout_rate)(lstm_out)
        lstm_out = Bidirectional(LSTM(lstm_units // 2, return_sequences=True, recurrent_dropout=0.2))(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(dropout_rate + 0.1)(lstm_out)

        # Multi-Head Attention: 시퀀스 내 중요 시점 강조
        attn_in = Permute((2, 1))(lstm_out)
        attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(attn_in, attn_in)
        attn_out = Permute((2, 1))(attn_out)

        # Variable Selection Network: 특징 중요도 학습
        vsn = Dense(attn_out.shape[-1], activation="sigmoid", kernel_regularizer=_L2)(attn_out)
        vsn_out = Multiply()([attn_out, vsn])

        # Gated Residual Network: 선택적 정보 전달
        grn_dense = Dense(64, activation="relu", kernel_regularizer=_L2)(vsn_out)
        grn_dense = BatchNormalization()(grn_dense)
        grn_gate = Dense(64, activation="sigmoid", kernel_regularizer=_L2)(vsn_out)
        grn_gate = BatchNormalization()(grn_gate)
        grn_out = Multiply()([grn_dense, grn_gate])
        grn_out = Dense(64, activation="relu", kernel_regularizer=_L2)(grn_out)
        grn_out = BatchNormalization()(grn_out)

        # Residual Connection: 마지막 시점 기준 합산
        residual = Add()([attn_out[:, -1, :], lstm_out[:, -1, :]])
        residual = LayerNormalization(epsilon=1e-6)(residual)
        residual = Dropout(0.2)(residual)

        grn_last = Dense(residual.shape[-1], activation="relu", kernel_regularizer=_L2)(grn_out[:, -1, :])
        final = Add()([residual, grn_last])

        # 출력 레이어
        out = Dense(64, activation="relu", kernel_regularizer=_L2)(final)
        out = BatchNormalization()(out)
        out = Dropout(dropout_rate)(out)
        out = Dense(32, activation="relu", kernel_regularizer=_L2)(out)
        out = BatchNormalization()(out)
        outputs = Dense(1)(out)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4) if use_adamw else Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"])
        return model

    @staticmethod
    def create_tft_model(
        input_shape: Tuple[int, int],
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        use_adamw: bool = False,
    ) -> Model:
        """Temporal Fusion Transformer 기반 모델을 생성합니다.

        Lim et al. (2020) 논문을 참고하여 구현한 간략화 버전입니다.
        변수 선택(VSN) → LSTM 시계열 처리 → Attention → GRN → 예측 순서로 구성됩니다.
        """
        logger.info(f"TFT 모델 생성 - lstm_units={lstm_units}, dropout={dropout_rate}, lr={learning_rate}")

        inputs = Input(shape=input_shape)

        # Variable Selection Network
        vsn = Dense(input_shape[-1], activation="sigmoid")(inputs)
        selected = Multiply()([inputs, vsn])

        # LSTM 시계열 처리
        lstm_out = LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.2)(selected)
        lstm_out = Dropout(dropout_rate)(lstm_out)

        # Interpretable Multi-Head Attention
        attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)
        attn_out = LayerNormalization()(attn_out)

        # Gated Residual Network
        grn_dense = Dense(lstm_units, activation="relu", kernel_regularizer=_L2)(attn_out)
        grn_dense = BatchNormalization()(grn_dense)
        grn_gate = Dense(lstm_units, activation="sigmoid", kernel_regularizer=_L2)(attn_out)
        grn_gate = BatchNormalization()(grn_gate)
        grn_out = Add()([lstm_out, Multiply()([grn_dense, grn_gate])])
        grn_out = LayerNormalization()(grn_out)

        last = Lambda(lambda x: x[:, -1, :])(grn_out)

        # 출력 레이어
        out = Dense(64, activation="relu", kernel_regularizer=_L2)(last)
        out = BatchNormalization()(out)
        out = Dropout(dropout_rate)(out)
        outputs = Dense(1)(out)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4) if use_adamw else Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=["mean_absolute_error"])
        return model
