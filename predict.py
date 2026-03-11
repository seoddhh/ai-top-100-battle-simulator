#!/usr/bin/env python3
"""
AI TOP 100 - 전투 결과 최종 예측 (문제 6) v2
Enhanced Feature Engineering + Ensemble + Stratified K-Fold CV
"""

import json
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 데이터 로드 및 전처리
# ============================================================

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_coord(at_str):
    parts = at_str.split(',')
    return float(parts[0]), float(parts[1])

def get_team_units(battle, team):
    units = []
    for u in battle[team]:
        x, y = parse_coord(u['at'])
        units.append({'type': u['type'], 'x': x, 'y': y})
    return units

def team_center(units):
    xs = [u['x'] for u in units]
    ys = [u['y'] for u in units]
    return np.array([np.mean(xs), np.mean(ys)])

# ============================================================
# 매치업 & 승률 테이블 구축
# ============================================================

UNIT_TYPES = ['aleo', 'bras', 'cbene', 'dgreg', 'eyanoo']
CENTER_REF = np.array([10.5, 10.5])

def build_matchup_table(train_data):
    """1v1 전투 데이터에서 매치업 승률표 생성"""
    win_count = defaultdict(lambda: defaultdict(int))
    total_count = defaultdict(lambda: defaultdict(int))
    
    battles_1v1 = [b for b in train_data if len(b['blue']) == 1 and len(b['red']) == 1]
    
    for b in battles_1v1:
        bt = b['blue'][0]['type']
        rt = b['red'][0]['type']
        winner_t = bt if b['winner'] == 'blue' else rt
        loser_t = rt if b['winner'] == 'blue' else bt
        
        total_count[bt][rt] += 1
        total_count[rt][bt] += 1
        win_count[winner_t][loser_t] += 1
    
    win_rate = {}
    for a in UNIT_TYPES:
        win_rate[a] = {}
        for b in UNIT_TYPES:
            if a == b:
                win_rate[a][b] = 0.5
            elif total_count[a][b] > 0:
                win_rate[a][b] = win_count[a][b] / total_count[a][b]
            else:
                win_rate[a][b] = 0.5
    return win_rate

def build_combo_stats(train_data):
    """팀 조합별 승률 통계 구축 (2v2, 3v3, 4v4)"""
    combo_stats = defaultdict(lambda: {'win': 0, 'total': 0})
    
    for b in train_data:
        for team in ['blue', 'red']:
            types = tuple(sorted(u['type'] for u in b[team]))
            size = len(b[team])
            key = (size, types)
            combo_stats[key]['total'] += 1
            if b['winner'] == team:
                combo_stats[key]['win'] += 1
    
    return combo_stats

# ============================================================
# Enhanced Feature Engineering
# ============================================================

def extract_features(battle, matchup_table, combo_stats):
    """전투 데이터에서 강화된 피처 벡터 추출"""
    blue_units = get_team_units(battle, 'blue')
    red_units = get_team_units(battle, 'red')
    n = len(blue_units)
    
    features = []
    
    # ========== 기본 정보 ==========
    features.append(n)  # 전투 규모
    
    # ========== 유닛 구성 ==========
    blue_counts = Counter(u['type'] for u in blue_units)
    red_counts = Counter(u['type'] for u in red_units)
    
    for t in UNIT_TYPES:
        features.append(blue_counts.get(t, 0))
    for t in UNIT_TYPES:
        features.append(red_counts.get(t, 0))
    for t in UNIT_TYPES:
        features.append(blue_counts.get(t, 0) - red_counts.get(t, 0))
    
    # 유닛 종류 다양성
    features.append(len(blue_counts))
    features.append(len(red_counts))
    features.append(len(blue_counts) - len(red_counts))
    
    # ========== 상성 점수 ==========
    # 전체 pairwise 매치업 점수
    matchup_score = 0.0
    matchup_max = -1.0
    matchup_min = 1.0
    pairwise_scores = []
    for bu in blue_units:
        for ru in red_units:
            s = matchup_table[bu['type']][ru['type']] - 0.5
            pairwise_scores.append(s)
            matchup_score += s
    
    features.append(matchup_score)
    features.append(np.max(pairwise_scores) if pairwise_scores else 0)
    features.append(np.min(pairwise_scores) if pairwise_scores else 0)
    features.append(np.std(pairwise_scores) if len(pairwise_scores) > 1 else 0)
    
    # 최강/최약 유닛의 상성 점수
    blue_power = {}
    for bu in blue_units:
        bt = bu['type']
        power = sum(matchup_table[bt][ru['type']] for ru in red_units) / n
        blue_power[bt] = blue_power.get(bt, 0) + power
    red_power = {}
    for ru in red_units:
        rt = ru['type']
        power = sum(matchup_table[rt][bu['type']] for bu in blue_units) / n
        red_power[rt] = red_power.get(rt, 0) + power
    
    features.append(max(blue_power.values()) if blue_power else 0.5)
    features.append(min(blue_power.values()) if blue_power else 0.5)
    features.append(max(red_power.values()) if red_power else 0.5)
    features.append(min(red_power.values()) if red_power else 0.5)
    
    # ========== 조합 승률 ==========
    blue_combo = tuple(sorted(u['type'] for u in blue_units))
    red_combo = tuple(sorted(u['type'] for u in red_units))
    
    blue_combo_key = (n, blue_combo)
    red_combo_key = (n, red_combo)
    
    blue_combo_rate = combo_stats[blue_combo_key]['win'] / combo_stats[blue_combo_key]['total'] \
        if combo_stats[blue_combo_key]['total'] > 10 else 0.5
    red_combo_rate = combo_stats[red_combo_key]['win'] / combo_stats[red_combo_key]['total'] \
        if combo_stats[red_combo_key]['total'] > 10 else 0.5
    
    features.append(blue_combo_rate)
    features.append(red_combo_rate)
    features.append(blue_combo_rate - red_combo_rate)
    
    # ========== 위치 특성 ==========
    bc = team_center(blue_units)
    rc = team_center(red_units)
    
    features.extend([bc[0], bc[1], rc[0], rc[1]])
    
    # 좌표 중심과의 거리
    blue_dist_center = np.linalg.norm(bc - CENTER_REF)
    red_dist_center = np.linalg.norm(rc - CENTER_REF)
    features.append(blue_dist_center)
    features.append(red_dist_center)
    features.append(blue_dist_center - red_dist_center)
    
    # 두 팀 중심 간 거리
    inter_dist = np.linalg.norm(bc - rc)
    features.append(inter_dist)
    
    # ========== 진형 특성 ==========
    blue_xs = [u['x'] for u in blue_units]
    blue_ys = [u['y'] for u in blue_units]
    red_xs = [u['x'] for u in red_units]
    red_ys = [u['y'] for u in red_units]
    
    if n >= 2:
        bxs, bys = np.std(blue_xs), np.std(blue_ys)
        bxr, byr = max(blue_xs)-min(blue_xs), max(blue_ys)-min(blue_ys)
        rxs, rys = np.std(red_xs), np.std(red_ys)
        rxr, ryr = max(red_xs)-min(red_xs), max(red_ys)-min(red_ys)
    else:
        bxs = bys = bxr = byr = rxs = rys = rxr = ryr = 0.0
    
    features.extend([bxs, bys, bxr, byr, rxs, rys, rxr, ryr])
    features.extend([bxs - bys, rxs - rys, bxr - byr, rxr - ryr])
    
    # 전체 spread (team area proxy)
    features.append(bxr * byr if n >= 2 else 0)
    features.append(rxr * ryr if n >= 2 else 0)
    
    # ========== 거리 특성 ==========
    # 팀 내 유닛 간 거리
    def calc_intra_dists(units):
        if len(units) < 2:
            return 0.0, 0.0, 0.0
        dists = []
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                d = np.sqrt((units[i]['x']-units[j]['x'])**2 + (units[i]['y']-units[j]['y'])**2)
                dists.append(d)
        return np.mean(dists), np.min(dists), np.max(dists)
    
    b_intra_mean, b_intra_min, b_intra_max = calc_intra_dists(blue_units)
    r_intra_mean, r_intra_min, r_intra_max = calc_intra_dists(red_units)
    features.extend([b_intra_mean, b_intra_min, b_intra_max])
    features.extend([r_intra_mean, r_intra_min, r_intra_max])
    features.extend([b_intra_mean - r_intra_mean, b_intra_min - r_intra_min, b_intra_max - r_intra_max])
    
    # 유닛별 적 팀 중심까지의 거리
    b2r = [np.linalg.norm(np.array([u['x'], u['y']]) - rc) for u in blue_units]
    r2b = [np.linalg.norm(np.array([u['x'], u['y']]) - bc) for u in red_units]
    features.extend([np.mean(b2r), np.min(b2r), np.max(b2r)])
    features.extend([np.mean(r2b), np.min(r2b), np.max(r2b)])
    features.append(np.mean(b2r) - np.mean(r2b))
    
    # 응집도 (유닛별 자기 팀 중심까지의 거리)
    b_coh = [np.linalg.norm(np.array([u['x'], u['y']]) - bc) for u in blue_units]
    r_coh = [np.linalg.norm(np.array([u['x'], u['y']]) - rc) for u in red_units]
    features.extend([np.mean(b_coh), np.mean(r_coh)])
    features.append(np.mean(b_coh) - np.mean(r_coh))
    
    # 최근접 적 거리
    def min_enemy_dists(team_units, enemy_units):
        dists = []
        for u in team_units:
            d = min(np.sqrt((u['x']-e['x'])**2 + (u['y']-e['y'])**2) for e in enemy_units)
            dists.append(d)
        return np.mean(dists), np.min(dists), np.max(dists)
    
    b_enemy_mean, b_enemy_min, b_enemy_max = min_enemy_dists(blue_units, red_units)
    r_enemy_mean, r_enemy_min, r_enemy_max = min_enemy_dists(red_units, blue_units)
    features.extend([b_enemy_mean, b_enemy_min, b_enemy_max])
    features.extend([r_enemy_mean, r_enemy_min, r_enemy_max])
    features.append(b_enemy_mean - r_enemy_mean)
    
    # ========== 전방/후방 비율 ==========
    if n >= 2:
        mid = (bc + rc) / 2.0
        d_blue = rc - bc
        d_red = bc - rc
        
        blue_front = sum(1 for u in blue_units if np.dot(np.array([u['x'], u['y']]) - mid, d_blue) > 0)
        red_front = sum(1 for u in red_units if np.dot(np.array([u['x'], u['y']]) - mid, d_red) > 0)
        
        features.append(blue_front / n)
        features.append(red_front / n)
        features.append(blue_front / n - red_front / n)
    else:
        features.extend([0.5, 0.5, 0.0])
    
    # ========== 유닛 타입별 위치 특성 ==========
    # 각 타입별 평균 좌표 (존재하지 않으면 0)
    for t in UNIT_TYPES:
        b_of_type = [u for u in blue_units if u['type'] == t]
        if b_of_type:
            features.extend([np.mean([u['x'] for u in b_of_type]), np.mean([u['y'] for u in b_of_type])])
        else:
            features.extend([0.0, 0.0])
    
    for t in UNIT_TYPES:
        r_of_type = [u for u in red_units if u['type'] == t]
        if r_of_type:
            features.extend([np.mean([u['x'] for u in r_of_type]), np.mean([u['y'] for u in r_of_type])])
        else:
            features.extend([0.0, 0.0])
    
    # ========== 전방 유닛 상성 점수 ==========
    if n >= 2:
        mid = (bc + rc) / 2.0
        d_blue = rc - bc
        d_red = bc - rc
        
        blue_front_units = [u for u in blue_units if np.dot(np.array([u['x'], u['y']]) - mid, d_blue) > 0]
        red_front_units = [u for u in red_units if np.dot(np.array([u['x'], u['y']]) - mid, d_red) > 0]
        
        # 전방 유닛들 간의 상성 점수
        front_matchup = 0.0
        front_count = 0
        for bu in (blue_front_units if blue_front_units else blue_units):
            for ru in (red_front_units if red_front_units else red_units):
                front_matchup += matchup_table[bu['type']][ru['type']] - 0.5
                front_count += 1
        features.append(front_matchup)
    else:
        features.append(matchup_score)
    
    return features

# ============================================================
# 메인
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("문제 6: 전투 결과 최종 예측 (v2 Enhanced)")
    print("=" * 60)
    
    train_data = load_data("ai_top_100_modeling/train_battles.json")
    test_data = load_data("ai_top_100_modeling/test_battles.json")
    print(f"훈련: {len(train_data)}건, 테스트: {len(test_data)}건")
    
    matchup_table = build_matchup_table(train_data)
    combo_stats = build_combo_stats(train_data)
    
    print("Feature 추출 중...")
    X_train = np.array([extract_features(b, matchup_table, combo_stats) for b in train_data])
    y_train = np.array([1 if b['winner'] == 'blue' else 0 for b in train_data])
    X_test = np.array([extract_features(b, matchup_table, combo_stats) for b in test_data])
    test_ids = [b['id'] for b in test_data]
    
    print(f"Feature shape: {X_train.shape}")
    
    # NaN/Inf 처리
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Stratified 5-Fold CV
    print("\n" + "-" * 40)
    print("Stratified 5-Fold Cross-Validation")
    print("-" * 40)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Model 1: GradientBoosting (tuned)
    gb = GradientBoostingClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42
    )
    
    gb_scores = cross_val_score(gb, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"\nGradientBoosting (tuned):")
    for i, s in enumerate(gb_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  평균: {gb_scores.mean():.4f} ± {gb_scores.std():.4f}")
    
    # Model 2: RandomForest (tuned)
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=20,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    rf_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"\nRandomForest (tuned):")
    for i, s in enumerate(rf_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  평균: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    
    # Model 3: Ensemble (Voting)
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf)],
        voting='soft'
    )
    
    ens_scores = cross_val_score(ensemble, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"\nEnsemble (GB+RF Soft Voting):")
    for i, s in enumerate(ens_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  평균: {ens_scores.mean():.4f} ± {ens_scores.std():.4f}")
    
    # 최고 성능 모델 선택
    model_scores = {'GB': gb_scores.mean(), 'RF': rf_scores.mean(), 'ENS': ens_scores.mean()}
    best_name = max(model_scores, key=model_scores.get)
    print(f"\n최고 성능: {best_name} ({model_scores[best_name]:.4f})")
    
    # 전체 데이터로 학습 + 예측
    if best_name == 'GB':
        final_model = GradientBoostingClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=20, max_features='sqrt', random_state=42
        )
    elif best_name == 'RF':
        final_model = RandomForestClassifier(
            n_estimators=800, max_depth=20, min_samples_leaf=3,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
    else:
        gb_f = GradientBoostingClassifier(
            n_estimators=800, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=20, max_features='sqrt', random_state=42
        )
        rf_f = RandomForestClassifier(
            n_estimators=800, max_depth=20, min_samples_leaf=3,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        final_model = VotingClassifier(estimators=[('gb', gb_f), ('rf', rf_f)], voting='soft')
    
    print("전체 데이터로 학습 중...")
    final_model.fit(X_train, y_train)
    
    y_pred = final_model.predict(X_test)
    
    predictions = []
    for i, tid in enumerate(test_ids):
        predictions.append({'id': tid, 'winner': 'blue' if y_pred[i] == 1 else 'red'})
    
    pred_blue = sum(1 for p in predictions if p['winner'] == 'blue')
    pred_red = sum(1 for p in predictions if p['winner'] == 'red')
    print(f"\n예측 분포: blue={pred_blue}, red={pred_red}")
    
    with open("predictions.json", 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"predictions.json 저장 완료 ({len(predictions)}건)")
    
    print("\n--- 샘플 출력 (처음 10건) ---")
    for p in predictions[:10]:
        print(f"  {p['id']}: {p['winner']}")
    
    print("\n완료!")
