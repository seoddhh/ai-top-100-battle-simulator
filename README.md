# AI TOP 100 — Battle Simulator
링크:https://challenge.aitop100.org/



## 📋 문제 개요

신규 전투 시뮬레이션 게임의 밸런스 패치를 위해, 실제 전투를 실행하지 않고 **유닛의 종류(type)와 2D 좌표(x, y)만으로 승패를 예측**하는 ML 모델을 개발합니다.

- **훈련 데이터**: 29,000건의 전투 시뮬레이션 (1v1 ~ 4v4)
- **테스트 데이터**: 500건 (승자 예측 필요)
- **유닛 종류**: `aleo`, `bras`, `cbene`, `dgreg`, `eyanoo` (5종)
- **총점**: 85점 (문제 1~5 객관식 + 문제 6 예측)

## 🏆 결과

| 문항 | 내용 | 배점 | 정답 |
|------|------|------|------|
| 1 | 1v1 최강 유닛 | 5점 | `dgreg` (승률 75.7%) |
| 2 | 전방/후방 배치 효과 | 5점 | `eyanoo` (차이 35.3%) |
| 3 | 진형 우세 | 5점 | x 방향 진형 |
| 4 | 상성 관계 검증 | 10점 | 4개 선지 |
| 5 | 데이터 사실 검증 | 10점 | 2개 선지 |
| 6 | 최종 승패 예측 | 50점 | **74% accuracy** (5-Fold CV) |

## 📁 프로젝트 구조

```
├── ai_top_100_modeling/
│   ├── train_battles.json    # 훈련 데이터 (29,000건)
│   └── test_battles.json     # 테스트 데이터 (500건)
├── analysis.py               # 문제 1~5 데이터 분석
├── predict.py                # 문제 6 ML 예측 모델
├── predictions.json          # 최종 예측 결과
├── challenge-overview.md     # 대회 문제 설명
└── requirements.txt
```

## 🚀 실행 방법

```bash
# 의존성 설치
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 문제 1~5 분석 실행
python3 analysis.py

# 문제 6 예측 모델 실행
python3 predict.py
```

## 🔍 접근 방법

### 문제 1~5: 데이터 분석

- **전방/후방 판별**: 두 팀 중심의 수직이등분선 기준 dot product로 분류
- **진형 판정**: range + std 이중 기준으로 교차 검증 (일치율 97.1%)
- **상성 분석**: Wilson score interval로 95% 신뢰구간 + 표본 수 고려
- **사실 검증**: 구간별 승률, 상관계수, 정확한 전적 매칭으로 각 보기 검증

### 문제 6: ML 예측 모델

**Feature Engineering** (102개 특성):

| 카테고리 | 주요 피처 |
|----------|----------|
| 유닛 구성 | 타입별 수, 다양성, 구성 차이 |
| 상성 점수 | 1v1 매치업 테이블 기반 pairwise 상성 |
| 조합 승률 | 팀 조합별 과거 승률 통계 |
| 위치 특성 | 팀 중심, 좌표 중심 거리, 진형 spread |
| 거리 특성 | 팀 내 거리, 적 중심 거리, 최근접 적 거리, 응집도 |
| 전방/후방 | 전방 유닛 비율, 전방 유닛 상성 점수 |

**모델 비교** (Stratified 5-Fold CV):

| 모델 | 정확도 |
|------|--------|
| **GradientBoosting** | **74.0% ± 0.6%** |
| RandomForest | 71.4% ± 0.5% |
| Ensemble (Soft Voting) | 73.6% ± 0.5% |

## 🛠️ 기술 스택

- **Python 3**
- **NumPy** — 수치 연산
- **scikit-learn** — ML 모델 (GradientBoosting, RandomForest)
