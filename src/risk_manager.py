from typing import Tuple

import numpy as np

from src.utils.logger import logger


class RiskManager:
    @staticmethod
    def calculate_var_cvar(
        returns: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """VaR(Value at Risk)와 CVaR(Conditional Value at Risk)을 계산합니다.

        Args:
            returns: 예측값과 실제값의 잔차 배열
            confidence_level: 신뢰 수준 (기본값: 95%)

        Returns:
            (var, cvar) 튜플
        """
        if returns is None or len(returns) == 0:
            logger.warning("수익률 데이터가 비어 있습니다. VaR, CVaR = 0.0 반환")
            return 0.0, 0.0

        var = np.quantile(returns, 1 - confidence_level)
        tail_values = returns[returns <= var]
        cvar = np.nanmean(tail_values) if len(tail_values) > 0 else var
        if np.isnan(cvar):
            cvar = var

        logger.info(f"VaR({confidence_level:.0%}): {var:.4f}, CVaR: {cvar:.4f}")
        return var, cvar

    @staticmethod
    def stop_loss_take_profit(
        predictions: np.ndarray, threshold: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """예측 가격 기준으로 손절가와 익절가를 계산합니다.

        Args:
            predictions: 예측된 주가 배열
            threshold: 손절/익절 비율 (기본값: 5%)

        Returns:
            (stop_loss, take_profit) 배열 튜플
        """
        if predictions is None or len(predictions) == 0:
            logger.warning("예측 데이터가 비어 있습니다. 빈 배열 반환")
            return np.array([]), np.array([])

        prices = np.array(predictions)
        return prices * (1 - threshold), prices * (1 + threshold)
