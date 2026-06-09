# 🛒 K-Pick: Instacart 재구매 예측 AI 모델

<p align="center">
  <img src="https://img.shields.io/badge/기간-2026.03.03%20~%202026.06.04-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.x-yellow?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/LightGBM-GBDT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/type-Capstone%20Project-red?style=flat-square"/>
</p>

---

## 프로젝트 소개

**K-Pick**은 미국 최대 식료품 배달 플랫폼 **Instacart**의 실제 구매 이력 데이터를 기반으로,  
사용자가 **특정 상품을 다시 구매할 것인지 예측하는 AI 모델**을 개발한 캡스톤 프로젝트입니다.

단순한 재구매 여부(Yes/No) 예측에서 나아가, 아래 두 가지 차별화 요소를 통해 예측 품질을 고도화했습니다.

- **구매 주기 예측**: 언제 살 것인지를 4단계 구간으로 분류
- **동시 구매 패턴(Co-purchase)**: 함께 자주 구매되는 상품 간 연관성을 Lift 지표로 정량화

> 📅 **프로젝트 기간**: 2026.03.03 ~ 2026.06.04 (총 3개월)

---

## 팀원 소개

| 이름 | 역할 |
|:---:|---|
| **이종호** | 유저 구매 패턴 분석 기반 피처 엔지니어링 |
| **고태원** | 유저-상품 상호작용 분석 기반 피처 엔지니어링 |
| **한민석** | 상품 간 상관관계 분석 기반 피처 엔지니어링 |
| **곽세희** | 협업 인프라 구축 및 데이터 엔지니어링 |

---

## 데이터셋

Kaggle [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis) 공개 데이터셋을 사용했습니다.

| 파일 | 설명 | 행 수 |
|---|---|---:|
| `orders.csv` | 주문 메타데이터 (사용자, 요일, 시간, 주문 순서) | 3,421,083 |
| `order_products__prior.csv` | 과거 주문 상세 내역 및 재주문 여부 | 32,434,489 |
| `order_products__train.csv` | 예측 대상 주문 (타깃 레이블) | 1,384,617 |
| `products.csv` | 상품명, 아이슬, 부서 정보 | 49,688 |
| `aisles.csv` | 진열대(aisle) 카테고리 134종 | 134 |
| `departments.csv` | 부서(department) 카테고리 21종 | 21 |

> 총 원본 데이터 용량: **약 681 MB**  
> 전처리 후 통합 피처 테이블: **약 4.4 GB**

---

## 데이터 분석

### EDA 주요 인사이트

<!-- 01_eda_integrity_check 결과 이미지 -->

- 전체 주문 중 재구매 비율: **약 59%** (타깃 클래스 불균형 존재)
- 사용자 1인당 평균 주문 횟수: **약 17회**
- 재주문이 가장 많은 시간대: **오전 10시 ~ 오후 2시**
- 재구매율이 높은 카테고리: produce(신선식품), dairy eggs(유제품·달걀)

### 결측값 처리

- `days_since_prior_order` 컬럼: 첫 주문 사용자에 한해 결측값 발생  
  → 해당 사용자는 이전 주문이 없으므로 **0으로 대체** (의미적으로 유효한 처리)

---

## 피처 엔지니어링

총 4가지 유형의 피처를 생성하고 하나의 통합 테이블로 병합했습니다.

### 1. 유저 피처 (이종호)

사용자의 전반적인 구매 습관을 수치화했습니다.

| 피처명 | 설명 |
|---|---|
| `u_total_orders` | 총 주문 횟수 |
| `u_reorder_ratio` | 전체 주문 대비 재주문 비율 |
| `u_avg_days_between_orders` | 평균 주문 간격 (일) |
| `u_avg_basket_size` | 평균 장바구니 상품 수 |

### 2. 유저-상품 상호작용 피처 (고태원)

특정 사용자가 특정 상품을 얼마나, 어떻게 구매했는지를 수치화했습니다.

| 피처명 | 설명 |
|---|---|
| `up_order_count` | 해당 사용자의 해당 상품 구매 횟수 |
| `up_reorder_rate` | 해당 사용자-상품 쌍의 재주문 비율 |
| `up_orders_since_last` | 마지막 구매 이후 경과한 주문 수 |
| `up_avg_position` | 장바구니 내 평균 등록 순서 |

### 3. 상품 피처 및 동시 구매 패턴 (한민석)

상품 자체의 인기도와 함께 구매되는 상품 간 연관성을 수치화했습니다.

| 피처명 | 설명 |
|---|---|
| `p_reorder_rate` | 전체 사용자 기준 상품 재주문율 |
| `p_total_orders` | 상품 총 주문 횟수 |
| `copurchase_lift` | 아이슬 기반 동시 구매 연관성 지표 (Lift 값) |

> **Lift 지표**: Lift = P(A∩B) / (P(A) × P(B))  
> Lift > 1이면 두 상품이 무작위보다 더 자주 함께 구매됨을 의미

### 4. 구매 타이밍 피처 (곽세희)

언제 다시 살 가능성이 높은지를 수치화했습니다.

| 피처명 | 설명 |
|---|---|
| `up_avg_interval` | 해당 사용자-상품 쌍의 평균 구매 주기 |
| `timing_ratio` | (마지막 구매 후 경과일) / (평균 구매 주기) — 1 이상이면 "이미 살 때 됐음" |
| `hazard_proxy` | 구매 이력이 1회뿐인 상품에 대해 사용자 평균 주기를 대리 적용한 값 |

---

## 모델링

### 모델 발전 과정

```
[1단계] Logistic Regression (베이스라인)
         ↓ F1 한계 확인
[2단계] LightGBM 기본 모델
         ↓ 임계값 최적화, 클래스 불균형 보정
[3단계] LightGBM + 통합 피처 테이블 (v2, v3)
         ↓ 피처 수 확대
[4단계] LightGBM + 타이밍 피처 + Co-purchase 추가
         ↓ 차별화 피처 도입
[5단계] 2-서브모델 구조 (최종)
         구매 시기 예측 모델 + 재구매 이진 분류 모델
```

### 최종 모델: 2-서브모델 구조

| 서브모델 | 입력 | 출력 |
|---|---|---|
| **Model 1** (구매 시기 분류) | 타이밍 피처 | 4구간: ≤7일 / 8-15일 / 16-30일 / >30일 |
| **Model 2** (재구매 예측) | 전체 피처 + 타이밍 + Co-purchase Lift | 재구매 여부 (0/1) |

### 클래스 불균형 처리

- 전체 데이터의 재구매:비재구매 비율 ≈ **1 : 9.2**
- `scale_pos_weight = 9.2` 적용으로 소수 클래스 가중치 보정
- 예측 임계값(Threshold)을 0.1~0.8 범위에서 탐색하여 **F1-Score 최대화 지점** 선택

---

## 모델 성능 비교

| 모델 | Accuracy | F1-Score | AUC | 비고 |
|---|:---:|:---:|:---:|---|
| Logistic Regression | 0.834 | 0.416 | - | 베이스라인 |
| LightGBM (기본) | 0.785 | 0.423 | - | 임계값 0.2 기준 |
| LightGBM + 통합 피처 (v2) | 0.795 | 0.438 | 0.853 | 70/20/10 분할 |
| LightGBM + 통합 피처 (v3) | 0.795 | 0.438 | 0.852 | prior+train 통합 |
| LightGBM + 타이밍 + Co-purchase | 0.796 | 0.442 | 0.855 | 차별화 피처 추가 |
| **최종 (Lift 기반 Co-purchase)** | **0.796** | **0.443** | **0.855** | **최종 제출 모델** |

> 구매 시기 분류 모델(4구간) 정확도: **0.64 ~ 0.70**

### 피처 중요도 (최종 모델 기준)

<!-- 04_model_train_06 Feature Importance -->

- 상위 피처: `up_order_count`, `up_orders_since_last`, `up_reorder_rate`
- 차별화 피처 기여: `timing_ratio`, `hazard_proxy`, `copurchase_lift` 모두 상위권 진입
- Lift 기반 Co-purchase 점수가 단순 빈도 기반 점수 대비 피처 중요도 순위 **유의미하게 상승**

---

## 분석 결과 요약

### 재구매 패턴 분석

<!-- 재구매율 분포 차트 이미지 -->

- **구매 횟수가 많을수록** 재구매율 증가 (충성 고객 효과)
- **마지막 구매 이후 경과 시간**이 평균 구매 주기에 근접할수록 재구매 확률 급상승
- **신선식품·유제품** 카테고리는 재구매율이 전체 평균 대비 약 15% 이상 높음

### 타이밍 피처 분석

<!-- timing_ratio 분포 및 재구매율 관계 이미지 -->

- `timing_ratio ≥ 1.0` (즉, 평균 구매 주기를 초과한 경우) 구간에서 재구매 확률이 현저히 증가
- 구매 주기가 짧은 사용자일수록 타이밍 피처의 예측 기여도 상승

### 동시 구매 패턴 분석

<!-- Co-purchase Lift 히트맵 이미지 -->

- produce(신선식품)와 dairy eggs(유제품)의 Lift 값이 전체 아이슬 중 최상위
- Lift > 3.0인 아이슬 조합은 함께 구매될 확률이 무작위 대비 3배 이상

---

## 결론

1. **타이밍 피처의 예측력 검증**  
   구매 주기와 현재 경과일을 비율화한 `timing_ratio`와 `hazard_proxy`는 단순 구매 횟수만큼이나 강력한 예측 신호임을 확인했습니다.

2. **Lift 기반 Co-purchase의 유효성**  
   단순 빈도 기반 동시 구매 점수보다 Lift로 정규화된 지표가 피처 중요도와 F1-Score 모두에서 더 우수한 성능을 보였습니다.

3. **클래스 불균형 극복**  
   `scale_pos_weight` 조정과 임계값 최적화를 결합함으로써, 심각한 클래스 불균형(1:9.2) 하에서도 안정적인 F1-Score를 달성했습니다.

4. **2-서브모델 구조의 실용성**  
   재구매 여부뿐 아니라 언제 살 것인지를 예측함으로써, 실제 서비스에서 타이밍 기반 마케팅 자동화(푸시 알림, 재구매 유도 쿠폰)에 직접 활용 가능한 구조를 설계했습니다.

---

## 확장 가능성

| 방향 | 내용 |
|---|---|
| **실시간 추론** | 사용자 최신 주문 발생 시 타이밍 피처를 즉시 갱신하여 재구매 가능성 실시간 알림 |
| **개인화 마케팅** | timing_ratio 기반으로 "살 때 됐어요" 알림 발송 타이밍 자동화 |
| **상품 추천 연계** | Co-purchase Lift 행렬을 활용한 "함께 구매하면 좋은 상품" 추천 시스템 확장 |
| **딥러닝 모델 적용** | 구매 시퀀스를 Transformer 계열 모델로 처리하여 장기 패턴 학습 |
| **다국적 플랫폼 적용** | 국내 커머스 데이터(쿠팡, 마켓컬리 등)에 동일 파이프라인 적용 |

---

## 기술 스택

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=flat-square"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
</p>

| 분류 | 라이브러리 |
|---|---|
| 데이터 처리 | `pandas`, `numpy` |
| 머신러닝 | `scikit-learn`, `lightgbm` |
| 희소 행렬 연산 | `scipy.sparse` (Co-purchase 행렬 계산) |
| 시각화 | `matplotlib`, `seaborn` |
| 개발 환경 | `Jupyter Notebook` |

---

## 성능 최적화

대용량 CSV(최대 1.9 GB) 처리를 위해 `notebooks/utils.py`의 `reduce_memory_usage()` 함수를 전처리 단계에 적용했습니다.

- `int64` → `int8 / int16 / int32` 다운캐스팅
- `float64` → `float32` 다운캐스팅
- **메모리 사용량 약 60% 절감** 달성
