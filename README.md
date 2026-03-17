# Smart-Trade: AI 기반 주가 예측 시스템

딥러닝 앙상블 모델을 활용하여 주가를 예측하고 리스크를 관리하는 시스템입니다.
CNN-BiLSTM-Attention 모델과 Temporal Fusion Transformer(TFT) 두 가지 모델을 결합하여 최대 60일 후의 주가를 예측합니다.

---

## 주요 기능

- **앙상블 예측**: CNN-BiLSTM-Attention + TFT 모델의 동적 가중치 앙상블
- **자동 하이퍼파라미터 최적화**: Optuna 기반 베이지안 최적화 (20회 탐색)
- **다중 데이터 소스 통합**: 주가, 거시경제 지표, 시장 지수, 기업 재무제표
- **리스크 관리**: VaR, CVaR, 손절가/익절가 자동 계산
- **배치 처리**: NASDAQ-100 전체 종목 자동 예측
- **데이터 캐싱**: API 중복 호출 방지를 위한 일자별 CSV 캐시

---

## 시스템 아키텍처

```
데이터 수집
    ├── Yahoo Finance (OHLCV 주가 데이터)
    ├── FRED API (CPI, PPI, GDP 성장률, 실업률, 기준금리)
    ├── Alpha Vantage (실업률, 실질 GDP - 일별 해상도)
    ├── Financial Modeling Prep (손익계산서, 연구개발비)
    └── 시장 지수 (S&P 500, VIX, 금, 원유)
            │
            ▼
    피처 엔지니어링
    ├── 20+ 기술적 지표 계산 (이동평균, RSI, MACD, 볼린저 밴드,
    │   EMA, ATR, 스토캐스틱, CCI, OBV, Williams %R)
    └── 시계열 순서 유지 학습/테스트 분할 (80:20)
            │
            ▼
    하이퍼파라미터 최적화 (Optuna, 20 trials)
            │
            ▼
    앙상블 학습
    ├── 모델 A: CNN-BiLSTM-Attention
    │   ├── Multi-kernel CNN (k=2,3,5,7) - 단기 패턴 추출
    │   ├── Stacked BiLSTM - 장기 트렌드 학습
    │   ├── Multi-Head Attention (4 heads) - 중요 시점 강조
    │   ├── Variable Selection Network (VSN)
    │   └── Gated Residual Network (GRN) + 잔차 연결
    └── 모델 B: Temporal Fusion Transformer (TFT)
        ├── Variable Selection Network
        ├── LSTM 시계열 처리
        ├── Interpretable Multi-Head Attention
        └── Gated Residual Network
            │
            ▼
    동적 가중치 앙상블
    (각 모델의 MAPE 역수 비율로 가중치 결정)
            │
            ▼
    결과 출력
    ├── 60일 미래 주가 예측
    ├── 리스크 지표 (VaR, CVaR, Sharpe Ratio)
    └── 손절가 / 익절가 계산
```

---

## 프로젝트 구조

```
smart-trade/
├── src/
│   ├── config.py                  # 전체 설정 중앙 관리 (환경변수 기반 API 키)
│   ├── data_fetcher.py            # 다중 소스 데이터 수집 및 캐싱
│   ├── feature_engineer.py        # 기술적 지표 계산 및 데이터 전처리
│   ├── model_builder.py           # Advanced 모델 및 TFT 모델 정의
│   ├── trainer.py                 # 학습, Optuna 최적화, 앙상블 로직
│   ├── forecast.py                # 자기회귀 다중 스텝 예측
│   ├── risk_manager.py            # VaR, CVaR, 손절가/익절가 계산
│   ├── visualizer.py              # 예측 결과 시각화 및 그래프 저장
│   ├── single_stock_predictor.py  # 단일 종목 엔드-투-엔드 파이프라인
│   ├── Top100_stock_predictor.py  # NASDAQ-100 배치 처리
│   └── utils/
│       ├── logger.py              # 컬러 콘솔 + 회전 파일 로거
│       └── korean_font.py         # matplotlib 한글 폰트 설정
├── tests/
│   ├── main_test.py
│   └── data_fetcher_test.py
├── .env.example                   # API 키 설정 템플릿
├── requirements.txt
└── README.md
```

---

## 설치 방법

**요구 환경:** Python 3.9 이상

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/smart-trade.git
cd smart-trade

# 2. 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# 3. 패키지 설치
pip install -r requirements.txt

# 4. API 키 설정
cp .env.example .env
# .env 파일을 열어 API 키를 입력하세요
```

---

## API 키 발급

본 프로젝트는 아래 무료 API 키를 필요로 합니다.

| 서비스 | 용도 | 발급 |
|--------|------|------|
| [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | CPI, GDP, 기준금리 등 거시경제 지표 | 무료 |
| [Alpha Vantage](https://www.alphavantage.co/support/#api-key) | 실업률, 실질 GDP (일별) | 무료 |
| [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs) | 기업 손익계산서 | 무료 티어 제공 |

발급 후 `.env` 파일에 입력하세요 (`.env.example` 참고):

```env
FRED_API_KEY=발급받은_키
ALPHAVANTAGE_API_KEY=발급받은_키
FMP_API_KEY=발급받은_키
```

---

## 사용 방법

### 단일 종목 예측

```python
from src.single_stock_predictor import SingleStockPredictor

predictor = SingleStockPredictor(ticker="AAPL")
result = predictor.run(visualize=True)

print(f"최종 MAPE: {result['MAPE (%)']:.2f}%")
print(f"60일 미래 예측: {result['Future Forecast']}")
```

### NASDAQ-100 배치 예측

```python
from src.config import Config
from src.Top100_stock_predictor import Top100EnsembleRunner

runner = Top100EnsembleRunner(Config())
runner.run()  # 결과를 src/top100/ 폴더에 종목별 CSV로 저장
```

### 설정 커스터마이징

```python
from src.config import Config

config = Config(ticker="NVDA")
config.update_config("forecast_steps", 30)    # 30일 예측
config.update_config("mape_threshold", 0.05)  # MAPE 5% 달성 시 조기 종료
config.update_config("max_iterations", 5)     # 최대 5회 반복 학습
```

---

## 모델 상세

### CNN-BiLSTM-Attention (Advanced Model)

| 구성 요소 | 상세 내용 |
|-----------|---------|
| Multi-kernel CNN | 4개 병렬 Conv1D (kernel=2,3,5,7) + MaxPooling + BatchNorm |
| Bidirectional LSTM | 2층 스택 BiLSTM + recurrent dropout |
| Multi-Head Attention | 4 heads, key_dim=64 |
| Variable Selection Network | Attention 출력에 sigmoid 게이팅 |
| Gated Residual Network | Dense + sigmoid 게이트 + 잔차 연결 |
| 출력 레이어 | FC(64) → FC(32) → Dense(1) |
| 손실 함수 | Mean Squared Error |
| 옵티마이저 | Adam (clipnorm=1.0) 또는 AdamW |

### Temporal Fusion Transformer (TFT Model)

| 구성 요소 | 상세 내용 |
|-----------|---------|
| Variable Selection | 피처별 sigmoid 게이팅 |
| LSTM 처리기 | 단층 LSTM + recurrent dropout |
| Interpretable Attention | 4-head attention + LayerNorm |
| Gated Residual Network | LSTM 출력에 가산적 잔차 연결 |
| 출력 레이어 | FC(64) → Dense(1) |
| 손실 함수 | Huber Loss (이상치에 강건) |
| 옵티마이저 | Adam 또는 AdamW |

### 앙상블 전략

두 모델을 각각 독립적으로 학습한 후, 검증 MAPE의 역수 비율로 가중치를 계산합니다.
MAPE가 낮을수록 (더 정확할수록) 높은 가중치를 부여합니다.

```
weight_A = (1 / MAPE_A) / ((1 / MAPE_A) + (1 / MAPE_B))
weight_B = (1 / MAPE_B) / ((1 / MAPE_A) + (1 / MAPE_B))

최종 예측 = weight_A × pred_A + weight_B × pred_B
```

---

## 사용 피처

**가격 및 거래량 (5개)**
`Close, Open, High, Low, Volume`

**기술적 지표 (15개)**
`5_MA, 20_MA, 60_MA, Volatility, RSI, MACD, BB_Upper, BB_Lower, EMA_50, ATR, Stoch_K, Stoch_D, CCI, OBV, Williams_%R`

**거시경제 지표 (3개)**
`GDP_Growth, Unemployment_Rate, Interest_Rate`

**시장 지수 (4개)**
`^GSPC (S&P 500), ^VIX (변동성 지수), GC=F (금), CL=F (원유)`

**기업 재무 지표 (4개)**
`revenue, operatingIncome, netIncome, researchAndDevelopmentExpenses`

---

## 리스크 관리 지표

| 지표 | 설명 |
|------|------|
| **VaR (95%)** | 95% 신뢰 수준에서의 최대 예상 손실 |
| **CVaR (95%)** | VaR 초과 손실의 조건부 기댓값 |
| **Sharpe Ratio** | 리스크 대비 수익률 지표 |
| **Stop-Loss** | 예측 가격 × (1 − 0.05) |
| **Take-Profit** | 예측 가격 × (1 + 0.05) |

---

## 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 딥러닝 | TensorFlow / Keras |
| 데이터 처리 | NumPy, Pandas, scikit-learn |
| 하이퍼파라미터 최적화 | Optuna |
| 데이터 수집 | yfinance, requests, BeautifulSoup |
| 시각화 | Matplotlib |
| 로깅 | colorlog |

---

## 참고 논문

| 논문 | 적용 내용 |
|------|----------|
| Vaswani et al. (2017) *Attention Is All You Need* | BiLSTM 이후 Multi-Head Attention 적용 |
| Borovykh et al. (2017) *Conditional Time Series Forecasting with CNNs* | 다양한 커널 크기의 병렬 Conv1D 레이어 |
| Lim et al. (2020) *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* | VSN, GRN 구성 요소의 TFT 아키텍처 |

---

## 라이선스

MIT License
