import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader


def download_financial_statements(company_name: str, email: str, ticker: str,
                                  form_type: str = "10-K", limit: int = 1):
    """
    특정 기업의 SEC 공시 데이터를 다운로드하는 함수

    :param company_name: 사용자 또는 회사명
    :param email: 사용자 이메일
    :param ticker: 기업의 종목 코드 (예: "AAPL" for Apple)
    :param form_type: 다운로드할 문서 유형 (기본값: "10-K")
    :param limit: 다운로드할 문서 개수 (기본값: 1)
    """
    dl = Downloader(company_name, email)
    print(f"Downloading {form_type} reports for {ticker}...")
    dl.get(form_type, ticker, limit=limit)
    print(f"Download completed! Check the 'sec-edgar-filings/{ticker}/{form_type}' directory.")


def is_financial_table(df: pd.DataFrame) -> bool:
    """
    DataFrame이 실제 재무 데이터(숫자가 일정 이상 있는지)인지 간단히 판별하는 예시 함수.
    필요하다면 조건을 더 추가할 수 있음.
    """
    # (1) 행/열이 너무 작으면 탈락 (목차 표일 가능성이 높음)
    if df.shape[0] < 3 or df.shape[1] < 2:
        return False

    # (2) 숫자 비중 검사: 전체 셀 중 숫자로 변환 가능한 값이 일정 비율 이상이어야 함
    numeric_count = 0
    total_count = df.shape[0] * df.shape[1]

    for col in df.columns:
        for val in df[col]:
            # 이미 int/float 인지 확인
            if isinstance(val, (int, float, np.number)):
                numeric_count += 1
            else:
                # 문자열로 변환 후, 콤마 등 제거하고 float 변환 시도
                try:
                    v = str(val).replace(',', '')
                    float(v)
                    numeric_count += 1
                except:
                    pass

    numeric_ratio = numeric_count / total_count
    # 기준 예: 0.2(20%) 미만이면 재무표가 아닐 가능성이 큼
    if numeric_ratio < 0.2:
        return False

    return True


def parse_financial_statements(ticker, form_type="10-K"):
    """
    다운로드한 SEC 공시 문서에서 재무 데이터를 추출하여 DataFrame을 생성하는 함수

    :param ticker: 기업의 종목 코드
    :param form_type: 다운로드한 문서 유형 (기본값: "10-K")
    :return: pandas DataFrame 형태의 재무 데이터
    """
    filings_dir = f"sec-edgar-filings/{ticker}/{form_type}/"
    data_frames = []

    if not os.path.exists(filings_dir):
        print(f"[Error] No filings found in '{filings_dir}'! Run the download function first.")
        return None

    # 루프를 돌며 HTML/텍스트 파일을 하나씩 읽는다
    for root, _, files in os.walk(filings_dir):
        for file in files:
            # HTML(.html/.htm) 또는 텍스트(.txt) 파일만 처리
            if file.lower().endswith((".html", ".htm", ".txt")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    # 파일 읽기 (인코딩 에러 무시)
                    content = f.read()

                # BeautifulSoup 파싱
                soup = BeautifulSoup(content, "lxml")

                # 모든 <table> 태그 순회
                tables = soup.find_all("table")
                for table in tables:
                    # <table> 전체 텍스트
                    table_text = table.get_text(separator=" ", strip=True).lower()

                    # 1) "Consolidated Statements of Operations" 키워드 검색
                    if "consolidated statements of operations" in table_text:
                        # 2) 목차(Index to...) 여부 체크 (목차 표라면 건너뛰기)
                        if "index to consolidated" in table_text:
                            # 'Index to Consolidated Financial Statements' 등은 스킵
                            continue

                        try:
                            # pandas.read_html()로 HTML -> DF 변환
                            dfs_in_table = pd.read_html(str(table), flavor="lxml")
                            for df_sub in dfs_in_table:
                                # 3) 실제 재무 테이블인지 확인
                                if is_financial_table(df_sub):
                                    data_frames.append(df_sub)
                                else:
                                    # print("Skipping a non-financial table.")
                                    pass
                        except Exception as e:
                            print(f"[Warning] Error processing table in file {file_path}: {e}")
                            continue

    # 파싱된 DataFrame이 없으면 안내
    if not data_frames:
        print("No financial tables found containing 'Consolidated Statements of Operations'.")
        return None

    # 여러 DataFrame을 하나로 합침 (sort=False -> 컬럼 충돌 시 경고 방지)
    combined_df = pd.concat(data_frames, ignore_index=True, sort=False)
    return combined_df


# 실행 예시
if __name__ == "__main__":
    COMPANY_NAME = "MyCompanyName"
    EMAIL = "my.email@domain.com"
    TICKER = "AAPL"  # 애플의 종목 코드
    FORM_TYPE = "10-K"  # 연간 보고서
    LIMIT = 2  # 최신 2개 문서 다운로드

    # 다운로드 (필요 시 주석 해제)
    # download_financial_statements(COMPANY_NAME, EMAIL, TICKER, FORM_TYPE, LIMIT)

    # 파싱
    df = parse_financial_statements(TICKER, FORM_TYPE)
    if df is not None:
        print(df.head())  # 상위 5개 데이터 미리보기
        output_name = f"{TICKER}_financial_data.csv"
        df.to_csv(output_name, index=False)
        print(f"Dataset saved as {output_name}")