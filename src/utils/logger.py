import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import colorlog


def setup_logging() -> logging.Logger:
    # 로그 디렉토리 경로 설정
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 로그 파일 이름 설정
    log_filename = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

    # 로거 생성
    logger = logging.getLogger("smart-trade")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 파일 핸들러 설정 (로그 파일 회전)
    file_handler = TimedRotatingFileHandler(
        log_filename, when="midnight", interval=1, encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 설정
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(asctime)s - %(levelname)-8s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# 로깅 설정을 최우선으로 수행하여 다른 모듈에서도 적용되도록 함
logger = setup_logging()

if __name__ == "__main__":
    # 로깅 설정 테스트
    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")
