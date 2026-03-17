import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.utils.logger import logger


def set_korean_font():
    plt.rcParams['axes.unicode_minus'] = False
    font_list = [f.name for f in font_manager.fontManager.ttflist]
    preferred_fonts = ["Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "AppleGothic"]

    for font in preferred_fonts:
        if font in font_list:
            plt.rcParams['font.family'] = font
            logger.info(f"한글 폰트 적용: {font}")
            return
    logger.warning("한글 폰트를 찾을 수 없습니다. 기본 설정 사용")