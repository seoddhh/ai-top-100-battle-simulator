# AI TOP 100 - 전투 시뮬레이션 승패 예측 풀이 계획

## 개요

전투 시뮬레이션 게임의 유닛 초기 배치 정보(유닛 종류 + 좌표)만으로 승패를 예측하는 문제입니다.  
총 6문항(85점 만점), 훈련 데이터 29,000건, 테스트 데이터 500건.

---

## 데이터 요약

| 전투 규모 | 훈련 데이터 | 테스트 데이터 |
|-----------|------------|-------------|
| 1v1 | 250 | 50 |
| 2v2 | 1,250 | 100 |
| 3v3 | 6,250 | 150 |
| 4v4 | 21,250 | 200 |
| **합계** | **29,000** | **500** |

유닛 타입: `aleo`, `bras`, `cbene`, `dgreg`, `eyanoo` (총 5종)

---

## 문제별 풀이 전략

### 문제 1 (5점) — 1v1 최강 유닛

> 1대1 전투에서 가장 높은 승률을 자랑하는 유닛 타입

**풀이 방법:**
1. 1v1 전투(250건)만 필터링
2. 각 유닛 타입별로 **참전 횟수** 및 **승리 횟수** 집계
3. 승률 = 승리 / 참전 으로 계산하여 최고 승률 유닛 선택

**구현:**
```python
# 1v1 전투 필터 → 유닛별 승률 계산
for battle in battles_1v1:
    blue_type = battle['blue'][0]['type']
    red_type = battle['red'][0]['type']
    winner_type = blue_type if winner=='blue' else red_type
    # 승리/참전 카운트 업데이트
```

---

### 문제 2 (5점) — 배치 효과 (전방 vs 후방 승률 차이)

> 전방 배치 vs 후방 배치 승률 차이가 가장 큰 유닛

**핵심 개념 정리:**
- **팀 중심**: 해당 팀 유닛 좌표의 평균 (mean_x, mean_y)
- **두 팀 중심을 잇는 선분**의 **수직이등분선**이 경계
- **전방**: 경계선 기준 **상대 팀 중심 쪽** 반평면
- **후방**: 반대쪽 반평면

**풀이 방법:**
1. 2v2 이상 전투 데이터 사용 (1v1은 전방/후방 구분 불가 - 유닛이 1개뿐)
2. 각 전투에서 양 팀 중심 계산
3. 두 팀 중심의 중점(수직이등분선 위의 점)과 방향벡터 계산
4. 각 유닛이 전방/후방 어디에 위치하는지 판별
5. 유닛 타입별 전방 승률, 후방 승률 계산 → 차이가 가장 큰 유닛 선택

**수학적 구현:**
```python
# 팀 중심: blue_center, red_center
# 방향벡터: d = red_center - blue_center (blue 기준)
# 중점: mid = (blue_center + red_center) / 2
# blue 유닛의 전방/후방 판별:
#   dot((unit_pos - mid), d) > 0 → 전방 (상대팀 쪽)
#   dot((unit_pos - mid), d) < 0 → 후방
# red 유닛은 방향을 반대로
```

---

### 문제 3 (5점) — 진형 우세 (x축 vs y축)

> x축 방향으로 넓은 진형 vs y축 방향으로 긴 진형, 어느 쪽이 승률이 높은가?

**풀이 방법:**
1. 2인 이상 전투에서 각 팀의 x좌표 분산(spread)과 y좌표 분산 계산
2. x_spread > y_spread → x방향 진형, 반대는 y방향 진형
3. 각 진형 유형별 승률 비교

**구현:**
```python
for battle in all_battles:
    for team in ['blue', 'red']:
        x_coords = [u['x'] for u in units]
        y_coords = [u['y'] for u in units]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        formation = 'x_wide' if x_range > y_range else 'y_long'
```

---

### 문제 4 (10점) — 상성 관계 (옳지 않은 것 고르기, 복수 선택)

> 주어진 10개 상성 관계 중 **틀린 것**을 모두 선택

**풀이 방법:**
1. 1v1 전투 데이터(250건)에서 모든 유닛 타입 매치업 승률 계산
2. A vs B 전적표를 만들어 각 매치업의 승률 확인
3. "A > B" (A가 B를 이김) 형태로 주어진 10개 선지를 검증
4. 승률 50% 이하인 상성은 틀린 것

**핵심:** 5개 유닛 타입 → C(5,2) = 10개 매치업 존재, 정확히 10개 선지가 주어짐

---

### 문제 5 (10점) — 데이터 검증 (올바르지 않은 것 고르기, 복수 선택)

> 5개 보기 중 train_battles.json 데이터와 **맞지 않는** 것을 모두 선택

**각 보기별 검증 방법:**

| # | 보기 | 검증 방법 |
|---|------|----------|
| 1 | 4v4에서 aleo+bras+dgreg+eyanoo 조합 승률 60%+ | 4v4 필터 → 해당 조합 찾기 → 승률 계산 |
| 2 | 팀 중심이 좌표 중심(10.5,10.5)에 가까울수록 승률 높음 | 각 팀 중심↔(10.5,10.5) 거리 계산 → 거리 구간별 승률 분석 |
| 3 | 2v2에서 aleo+dgreg vs bras+eyanoo 전적이 26전 25승 | 정확한 매치업 필터링 → 전적 확인 |
| 4 | dgreg는 전방이 후방보다 승률 높음 | 문제2 로직 재활용 → dgreg 전방/후방 승률 비교 |
| 5 | 같은 팀 유닛 간 거리 가까울수록 승률 높음 | 팀 내 유닛 간 평균거리 계산 → 거리-승률 상관관계 분석 |

---

### 문제 6 (50점) — 최종 승자 예측 (test_battles.json)

> 500건 테스트 데이터의 승자 예측, 정확도 60%~80% 기준 2%당 5점 (최대 50점)

**풀이 전략 — Feature Engineering + ML Classification:**

#### Feature 추출 (양 팀 각각):
1. **유닛 구성**: 각 타입별 유닛 수 (5개 피처 × 2팀)
2. **유닛 수 차이**: blue 유닛 수 - red 유닛 수
3. **상성 점수**: 1v1 매치업 승률표 기반 팀 상성 총합
4. **위치 특성**: 팀 중심 좌표, 좌표 중심과의 거리
5. **진형 특성**: x/y 방향 spread (분산, 범위)
6. **팀 내 거리**: 유닛 간 평균 거리, 최소/최대 거리
7. **전방 유닛 비율**: 전방에 배치된 유닛의 비율
8. **유닛 상대적 위치**: 상대 팀 중심까지의 평균 거리

#### 모델:
- **Primary**: `scikit-learn`의 `GradientBoostingClassifier` 또는 `RandomForestClassifier`
- **Alternative**: `XGBoost` / `LightGBM` (설치 가능한 경우)
- Train/Val split (80:20)으로 검증 후 전체 훈련 데이터로 최종 학습

#### 출력:
```json
[
  {"id": "test_001", "winner": "blue"},
  {"id": "test_002", "winner": "red"},
  ...
]
```

---

## 구현 계획

### [NEW] [solve_all.py](file:///Users/seodonghwi/Desktop/ai_top100/solve_all.py)

문제 1~5의 답을 분석하고, 문제 6의 예측 결과를 생성하는 통합 Python 스크립트.

**구조:**
1. 데이터 로드 및 전처리
2. 문제 1~5 각각의 분석 함수 구현
3. 문제 6 Feature Engineering + 모델 학습 + 예측
4. 결과 출력 및 JSON 파일 저장

### [NEW] [predictions.json](file:///Users/seodonghwi/Desktop/ai_top100/predictions.json)

문제 6의 최종 예측 결과 (test_battles.json의 500건 승자 예측)

---

## Verification Plan

### Automated Tests
```bash
cd /Users/seodonghwi/Desktop/ai_top100
python3 solve_all.py
```

- 문제 1~5: 각 분석 결과와 선지 대조하여 정답 확정
- 문제 6: train 데이터 80:20 split에서 validation accuracy 확인 (목표: 80%+)
  - predictions.json 파일 생성 확인 (500건, 올바른 JSON 형식)

### Manual Verification
- 문제 1~5의 분석 결과 출력을 사용자에게 공유하여 정답 확인 요청
- 문제 6의 validation accuracy 수치 확인
