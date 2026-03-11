#!/usr/bin/env python3
"""
AI TOP 100 - 전투 시뮬레이션 분석 (문제 1~5)
유닛 초기 배치 정보로 전투 데이터를 분석하여 각 문항의 정답을 도출합니다.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations

# ============================================================
# 데이터 로드 및 전처리
# ============================================================

def load_data(path="ai_top_100_modeling/train_battles.json"):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def parse_coord(at_str):
    """좌표 문자열 "x,y"를 (x, y) float 튜플로 변환"""
    parts = at_str.split(',')
    return float(parts[0]), float(parts[1])

def get_team_units(battle, team):
    """전투에서 특정 팀의 유닛 리스트를 좌표 파싱 포함하여 반환"""
    units = []
    for u in battle[team]:
        x, y = parse_coord(u['at'])
        units.append({
            'unit_id': u['unit_id'],
            'type': u['type'],
            'x': x,
            'y': y
        })
    return units

def team_center(units):
    """팀 유닛들의 중심 좌표 계산"""
    xs = [u['x'] for u in units]
    ys = [u['y'] for u in units]
    return np.array([np.mean(xs), np.mean(ys)])

# ============================================================
# 전방/후방 판별 유틸리티
# ============================================================

def classify_position(unit, my_center, enemy_center):
    """
    유닛이 전방/후방/경계에 위치하는지 판별.
    
    정의:
    - 두 팀 중심을 잇는 선분의 수직이등분선이 경계
    - 전방: 경계선 기준 상대 팀 중심 쪽 반평면
    - 후방: 반대쪽 반평면
    
    수학:
    - mid = (my_center + enemy_center) / 2
    - d = enemy_center - my_center  (방향벡터: 내 팀 → 상대 팀)
    - dot((unit_pos - mid), d) > 0 → 전방 (상대팀 쪽)
    - dot((unit_pos - mid), d) < 0 → 후방
    - dot == 0 → 경계 (제외)
    """
    mid = (my_center + enemy_center) / 2.0
    d = enemy_center - my_center
    unit_pos = np.array([unit['x'], unit['y']])
    dot = np.dot(unit_pos - mid, d)
    
    if dot > 1e-9:
        return 'front'
    elif dot < -1e-9:
        return 'rear'
    else:
        return 'boundary'

# ============================================================
# 진형 판정 유틸리티
# ============================================================

def classify_formation(units):
    """
    팀 유닛의 진형을 판정.
    
    range 기반과 std 기반 두 가지로 판정하여 교차 검증.
    
    Returns:
        (range_result, std_result, x_range, y_range, x_std, y_std)
        result: 'x_wide', 'y_long', 'equal'
    """
    if len(units) < 2:
        return ('equal', 'equal', 0, 0, 0, 0)
    
    xs = [u['x'] for u in units]
    ys = [u['y'] for u in units]
    
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    x_std = np.std(xs)
    y_std = np.std(ys)
    
    # range 기반
    if x_range > y_range:
        range_result = 'x_wide'
    elif y_range > x_range:
        range_result = 'y_long'
    else:
        range_result = 'equal'
    
    # std 기반
    if x_std > y_std:
        std_result = 'x_wide'
    elif y_std > x_std:
        std_result = 'y_long'
    else:
        std_result = 'equal'
    
    return (range_result, std_result, x_range, y_range, x_std, y_std)

# ============================================================
# 신뢰구간 계산 (이항분포)
# ============================================================

def binomial_ci(wins, total, confidence=0.95):
    """이항분포 기반 승률 신뢰구간 (Wilson score interval)"""
    if total == 0:
        return (0, 0, 0)
    
    from math import sqrt
    z = 1.96  # 95% confidence
    p_hat = wins / total
    
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator
    
    return (p_hat, max(0, center - margin), min(1, center + margin))

# ============================================================
# 문제 1: 1v1 최강 유닛
# ============================================================

def solve_q1(data):
    """1대1 전투에서 가장 높은 승률을 자랑하는 유닛 타입"""
    print("=" * 60)
    print("문제 1: 1v1 최강자는?")
    print("=" * 60)
    
    battles_1v1 = [b for b in data if len(b['blue']) == 1 and len(b['red']) == 1]
    print(f"1v1 전투 수: {len(battles_1v1)}")
    
    # 유닛별 참전/승리 카운트
    appeared = Counter()
    won = Counter()
    
    for b in battles_1v1:
        blue_type = b['blue'][0]['type']
        red_type = b['red'][0]['type']
        appeared[blue_type] += 1
        appeared[red_type] += 1
        
        winner_type = blue_type if b['winner'] == 'blue' else red_type
        won[winner_type] += 1
    
    print(f"\n{'유닛':<10} {'참전':<6} {'승리':<6} {'승률':<10}")
    print("-" * 35)
    
    results = {}
    for unit_type in sorted(appeared.keys()):
        win_rate = won[unit_type] / appeared[unit_type]
        results[unit_type] = win_rate
        print(f"{unit_type:<10} {appeared[unit_type]:<6} {won[unit_type]:<6} {win_rate:.4f}")
    
    best = max(results, key=results.get)
    print(f"\n★ 정답: {best} (승률: {results[best]:.4f})")
    return best

# ============================================================
# 문제 2: 배치 효과 (전방/후방 승률 차이)
# ============================================================

def solve_q2(data):
    """전방 배치 vs 후방 배치 승률 차이가 가장 큰 유닛"""
    print("\n" + "=" * 60)
    print("문제 2: 배치 효과 (전방 vs 후방 승률 차이)")
    print("=" * 60)
    
    # 2v2 이상 전투만 사용
    battles_multi = [b for b in data if len(b['blue']) >= 2]
    print(f"2인 이상 전투 수: {len(battles_multi)}")
    
    # 전방/후방 검증: 샘플 5건 출력
    print("\n--- 전방/후방 판별 검증 (샘플 5건) ---")
    for i, b in enumerate(battles_multi[:5]):
        blue_units = get_team_units(b, 'blue')
        red_units = get_team_units(b, 'red')
        bc = team_center(blue_units)
        rc = team_center(red_units)
        
        print(f"\n전투 {b['id']}:")
        print(f"  Blue 중심: ({bc[0]:.1f}, {bc[1]:.1f}), Red 중심: ({rc[0]:.1f}, {rc[1]:.1f})")
        print(f"  중점: ({(bc[0]+rc[0])/2:.1f}, {(bc[1]+rc[1])/2:.1f})")
        
        for u in blue_units:
            pos = classify_position(u, bc, rc)
            print(f"  Blue {u['type']}({u['x']},{u['y']}): {pos}")
        for u in red_units:
            pos = classify_position(u, rc, bc)
            print(f"  Red {u['type']}({u['x']},{u['y']}): {pos}")
    
    # 전체 데이터에서 유닛별 전방/후방 승률 계산
    front_wins = defaultdict(int)
    front_total = defaultdict(int)
    rear_wins = defaultdict(int)
    rear_total = defaultdict(int)
    
    for b in battles_multi:
        blue_units = get_team_units(b, 'blue')
        red_units = get_team_units(b, 'red')
        bc = team_center(blue_units)
        rc = team_center(red_units)
        
        is_blue_win = b['winner'] == 'blue'
        is_red_win = b['winner'] == 'red'
        
        # Blue 팀 유닛 처리
        for u in blue_units:
            pos = classify_position(u, bc, rc)
            if pos == 'front':
                front_total[u['type']] += 1
                if is_blue_win:
                    front_wins[u['type']] += 1
            elif pos == 'rear':
                rear_total[u['type']] += 1
                if is_blue_win:
                    rear_wins[u['type']] += 1
        
        # Red 팀 유닛 처리 (방향 반대)
        for u in red_units:
            pos = classify_position(u, rc, bc)
            if pos == 'front':
                front_total[u['type']] += 1
                if is_red_win:
                    front_wins[u['type']] += 1
            elif pos == 'rear':
                rear_total[u['type']] += 1
                if is_red_win:
                    rear_wins[u['type']] += 1
    
    print(f"\n{'유닛':<10} {'전방승률':<12} {'(N)':<8} {'후방승률':<12} {'(N)':<8} {'차이(절대값)':<12}")
    print("-" * 65)
    
    results = {}
    for unit_type in sorted(set(list(front_total.keys()) + list(rear_total.keys()))):
        fr = front_wins[unit_type] / front_total[unit_type] if front_total[unit_type] > 0 else 0
        rr = rear_wins[unit_type] / rear_total[unit_type] if rear_total[unit_type] > 0 else 0
        diff = abs(fr - rr)
        results[unit_type] = {
            'front_rate': fr, 'front_n': front_total[unit_type],
            'rear_rate': rr, 'rear_n': rear_total[unit_type],
            'diff': diff, 'direction': '전방>후방' if fr > rr else '후방>전방'
        }
        print(f"{unit_type:<10} {fr:.4f}      {front_total[unit_type]:<8} {rr:.4f}      {rear_total[unit_type]:<8} {diff:.4f}")
    
    best = max(results, key=lambda k: results[k]['diff'])
    r = results[best]
    print(f"\n★ 정답: {best} (차이: {r['diff']:.4f}, {r['direction']})")
    return best

# ============================================================
# 문제 3: 진형 우세 (x축 vs y축)
# ============================================================

def solve_q3(data):
    """x방향 진형 vs y방향 진형 승률 비교"""
    print("\n" + "=" * 60)
    print("문제 3: 진형 우세 예측 (x축 vs y축)")
    print("=" * 60)
    
    # 2인 이상 전투만 사용
    battles_multi = [b for b in data if len(b['blue']) >= 2]
    
    # range 기반 집계
    range_stats = {'x_wide': {'win': 0, 'total': 0}, 'y_long': {'win': 0, 'total': 0}, 'equal': {'win': 0, 'total': 0}}
    # std 기반 집계
    std_stats = {'x_wide': {'win': 0, 'total': 0}, 'y_long': {'win': 0, 'total': 0}, 'equal': {'win': 0, 'total': 0}}
    # 일치 여부
    agree_count = 0
    disagree_count = 0
    
    for b in battles_multi:
        for team in ['blue', 'red']:
            units = get_team_units(b, team)
            r_result, s_result, xr, yr, xs, ys = classify_formation(units)
            
            is_win = b['winner'] == team
            
            range_stats[r_result]['total'] += 1
            if is_win:
                range_stats[r_result]['win'] += 1
            
            std_stats[s_result]['total'] += 1
            if is_win:
                std_stats[s_result]['win'] += 1
            
            if r_result == s_result:
                agree_count += 1
            else:
                disagree_count += 1
    
    print("\n--- Range 기반 진형 승률 ---")
    for k in ['x_wide', 'y_long', 'equal']:
        if range_stats[k]['total'] > 0:
            rate = range_stats[k]['win'] / range_stats[k]['total']
            print(f"  {k:<10}: {rate:.4f} ({range_stats[k]['win']}/{range_stats[k]['total']})")
    
    print("\n--- Std 기반 진형 승률 ---")
    for k in ['x_wide', 'y_long', 'equal']:
        if std_stats[k]['total'] > 0:
            rate = std_stats[k]['win'] / std_stats[k]['total']
            print(f"  {k:<10}: {rate:.4f} ({std_stats[k]['win']}/{std_stats[k]['total']})")
    
    print(f"\n두 기준 일치: {agree_count}, 불일치: {disagree_count} "
          f"(일치율: {agree_count / (agree_count + disagree_count) * 100:.1f}%)")
    
    # 종합 판단
    r_x = range_stats['x_wide']['win'] / range_stats['x_wide']['total'] if range_stats['x_wide']['total'] > 0 else 0
    r_y = range_stats['y_long']['win'] / range_stats['y_long']['total'] if range_stats['y_long']['total'] > 0 else 0
    s_x = std_stats['x_wide']['win'] / std_stats['x_wide']['total'] if std_stats['x_wide']['total'] > 0 else 0
    s_y = std_stats['y_long']['win'] / std_stats['y_long']['total'] if std_stats['y_long']['total'] > 0 else 0
    
    print(f"\n종합:")
    print(f"  Range: x={r_x:.4f} vs y={r_y:.4f} → {'x 우세' if r_x > r_y else 'y 우세'}")
    print(f"  Std:   x={s_x:.4f} vs y={s_y:.4f} → {'x 우세' if s_x > s_y else 'y 우세'}")
    
    # 두 기준 모두 동의하는 결과
    if (r_x > r_y) and (s_x > s_y):
        answer = "x 방향으로 긴 진형"
    elif (r_y > r_x) and (s_y > s_x):
        answer = "y 방향으로 긴 진형"
    else:
        answer = "두 기준이 불일치 - 추가 분석 필요"
    
    print(f"\n★ 정답: {answer}")
    return answer

# ============================================================
# 문제 4: 상성 관계 (옳지 않은 것 고르기)
# ============================================================

def solve_q4(data):
    """1v1 매치업 승률표로 상성 관계 검증"""
    print("\n" + "=" * 60)
    print("문제 4: 상성 관계 (옳지 않은 것 고르기)")
    print("=" * 60)
    
    battles_1v1 = [b for b in data if len(b['blue']) == 1 and len(b['red']) == 1]
    
    # 매치업별 전적 (A vs B에서 A의 승리 기준)
    matchup_wins = defaultdict(int)  # (A, B) → A의 승수
    matchup_total = defaultdict(int)  # (A, B) → 총 경기수
    
    for b in battles_1v1:
        blue_type = b['blue'][0]['type']
        red_type = b['red'][0]['type']
        
        if blue_type == red_type:
            continue  # 동일 유닛 대전은 무의미
        
        # 항상 알파벳 순서로 정렬된 키 사용
        a, bb = sorted([blue_type, red_type])
        matchup_total[(a, bb)] += 1
        
        winner_type = blue_type if b['winner'] == 'blue' else red_type
        if winner_type == a:
            matchup_wins[(a, bb)] += 1
    
    # 전적표 출력
    unit_types = sorted(set(u['type'] for b in battles_1v1 for u in b['blue'] + b['red']))
    
    print(f"\n--- 1v1 매치업 전적표 (행이 이기는 쪽) ---")
    print(f"{'A vs B':<20} {'전적(A승/총)':<12} {'A승률':<10} {'95% CI':<15} {'표본수':<6} {'판정'}")
    print("-" * 80)
    
    matchup_results = {}
    for a, b in combinations(unit_types, 2):
        total = matchup_total[(a, b)]
        wins_a = matchup_wins[(a, b)]
        wins_b = total - wins_a
        
        if total > 0:
            rate_a = wins_a / total
            p_hat, ci_low, ci_high = binomial_ci(wins_a, total)
            
            # 불확실성 판정
            if ci_low > 0.5:
                verdict = f"{a} > {b} ✓"
            elif ci_high < 0.5:
                verdict = f"{b} > {a} ✓"
            else:
                verdict = "불확실 (?)"
            
            matchup_results[(a, b)] = {
                'a_wins': wins_a, 'b_wins': wins_b, 'total': total,
                'a_rate': rate_a, 'b_rate': 1 - rate_a,
                'ci_low': ci_low, 'ci_high': ci_high,
                'verdict': verdict
            }
            
            print(f"{a} vs {b:<12} {wins_a:>3}/{total:<8} {rate_a:.4f}    [{ci_low:.3f}, {ci_high:.3f}]  {total:<6} {verdict}")
    
    # 선지 검증
    claims = [
        ("dgreg", "aleo", "dgreg > aleo"),
        ("cbene", "eyanoo", "cbene > eyanoo"),
        ("bras", "cbene", "bras > cbene"),
        ("aleo", "bras", "aleo > bras"),
        ("cbene", "aleo", "cbene > aleo"),
        ("aleo", "eyanoo", "aleo > eyanoo"),
        ("bras", "dgreg", "bras > dgreg"),
        ("eyanoo", "bras", "eyanoo > bras"),
        ("dgreg", "cbene", "dgreg > cbene"),
        ("eyanoo", "dgreg", "eyanoo > dgreg"),
    ]
    
    print(f"\n--- 선지 검증 ---")
    wrong_claims = []
    
    for winner_claim, loser_claim, label in claims:
        a, b = sorted([winner_claim, loser_claim])
        result = matchup_results.get((a, b))
        
        if result is None:
            print(f"  {label}: 데이터 없음")
            continue
        
        # winner_claim이 a인지 b인지 확인
        if winner_claim == a:
            actual_rate = result['a_rate']
            ci_l, ci_h = result['ci_low'], result['ci_high']
        else:
            actual_rate = result['b_rate']
            ci_l, ci_h = 1 - result['ci_high'], 1 - result['ci_low']
        
        is_correct = actual_rate > 0.5
        is_uncertain = ci_l <= 0.5 <= ci_h if winner_claim == a else (1 - ci_h) <= 0.5 <= (1 - ci_l)
        
        status = "✓ 맞음" if is_correct else "✗ 틀림"
        if not is_correct:
            wrong_claims.append(label)
        
        uncertainty = " (불확실)" if is_uncertain else " (확실)"
        print(f"  {label}: 실제 승률 {actual_rate:.4f} [{ci_l:.3f},{ci_h:.3f}] → {status}{uncertainty}")
    
    print(f"\n★ 옳지 않은 선지: {wrong_claims}")
    return wrong_claims

# ============================================================
# 문제 5: 데이터 검증 (올바르지 않은 것 고르기)
# ============================================================

def solve_q5(data):
    """5개 보기 중 train_battles.json 데이터와 맞지 않는 것 선택"""
    print("\n" + "=" * 60)
    print("문제 5: 데이터 검증 (올바르지 않은 것 고르기)")
    print("=" * 60)
    
    wrong_statements = []
    
    # --------------------------------------------------------
    # 보기 1: 4v4에서 aleo+bras+dgreg+eyanoo 조합 승률 60%+
    # --------------------------------------------------------
    print("\n--- 보기 1: 4v4 aleo+bras+dgreg+eyanoo 조합 승률 ---")
    battles_4v4 = [b for b in data if len(b['blue']) == 4]
    
    combo_target = {'aleo', 'bras', 'dgreg', 'eyanoo'}
    combo_wins = 0
    combo_total = 0
    
    for b in battles_4v4:
        for team in ['blue', 'red']:
            team_types = set(u['type'] for u in b[team])
            if team_types == combo_target:
                combo_total += 1
                if b['winner'] == team:
                    combo_wins += 1
    
    if combo_total > 0:
        combo_rate = combo_wins / combo_total
        print(f"  전적: {combo_wins}/{combo_total} = {combo_rate:.4f}")
        print(f"  60% 이상? {'예 ✓' if combo_rate >= 0.6 else '아니오 ✗'}")
        if combo_rate < 0.6:
            wrong_statements.append("보기 1")
    else:
        print("  해당 조합 없음")
        wrong_statements.append("보기 1")
    
    # --------------------------------------------------------
    # 보기 2: 팀 중심이 (10.5,10.5)에 가까울수록 승률 높음
    # --------------------------------------------------------
    print("\n--- 보기 2: 팀 중심 ↔ 좌표 중심 거리 vs 승률 ---")
    center_ref = np.array([10.5, 10.5])
    
    dist_win = []
    dist_lose = []
    
    for b in data:
        for team in ['blue', 'red']:
            units = get_team_units(b, team)
            tc = team_center(units)
            dist = np.linalg.norm(tc - center_ref)
            is_win = b['winner'] == team
            if is_win:
                dist_win.append(dist)
            else:
                dist_lose.append(dist)
    
    avg_dist_win = np.mean(dist_win)
    avg_dist_lose = np.mean(dist_lose)
    print(f"  승리팀 평균 거리: {avg_dist_win:.4f}")
    print(f"  패배팀 평균 거리: {avg_dist_lose:.4f}")
    
    # 구간별 승률 분석
    all_dists = []
    for b in data:
        for team in ['blue', 'red']:
            units = get_team_units(b, team)
            tc = team_center(units)
            dist = np.linalg.norm(tc - center_ref)
            is_win = b['winner'] == team
            all_dists.append((dist, is_win))
    
    all_dists.sort(key=lambda x: x[0])
    n = len(all_dists)
    quartiles = [all_dists[:n//4], all_dists[n//4:n//2], all_dists[n//2:3*n//4], all_dists[3*n//4:]]
    
    print(f"\n  거리 구간별 승률:")
    closer_wins_more = True
    prev_rate = None
    for i, q in enumerate(quartiles):
        wins = sum(1 for _, w in q if w)
        rate = wins / len(q) if len(q) > 0 else 0
        dist_range = f"[{q[0][0]:.2f}, {q[-1][0]:.2f}]"
        print(f"  Q{i+1} {dist_range}: {rate:.4f} ({wins}/{len(q)})")
        if prev_rate is not None and rate > prev_rate:
            closer_wins_more = False
        prev_rate = rate
    
    trend = "거리 가까울수록 승률 높음 ✓" if closer_wins_more else "단조 감소 아님 ✗"
    print(f"  판정: {trend}")
    
    # 상관계수로도 확인
    dists_arr = np.array([d for d, _ in all_dists])
    wins_arr = np.array([1.0 if w else 0.0 for _, w in all_dists])
    corr = np.corrcoef(dists_arr, wins_arr)[0, 1]
    print(f"  거리-승리 상관계수: {corr:.4f} ({'음의 상관=가까울수록 승률↑' if corr < 0 else '양의 상관=가까울수록 승률↓'})")
    
    if corr >= 0:
        wrong_statements.append("보기 2")
    
    # --------------------------------------------------------
    # 보기 3: 2v2에서 aleo+dgreg vs bras+eyanoo 26전 25승
    # --------------------------------------------------------
    print("\n--- 보기 3: 2v2 aleo+dgreg vs bras+eyanoo 전적 ---")
    battles_2v2 = [b for b in data if len(b['blue']) == 2]
    
    target_a = {'aleo', 'dgreg'}
    target_b = {'bras', 'eyanoo'}
    
    matchup_count = 0
    matchup_a_wins = 0
    
    for b in battles_2v2:
        blue_types = set(u['type'] for u in b['blue'])
        red_types = set(u['type'] for u in b['red'])
        
        if blue_types == target_a and red_types == target_b:
            matchup_count += 1
            if b['winner'] == 'blue':
                matchup_a_wins += 1
        elif blue_types == target_b and red_types == target_a:
            matchup_count += 1
            if b['winner'] == 'red':
                matchup_a_wins += 1
    
    print(f"  aleo+dgreg vs bras+eyanoo: {matchup_count}전 {matchup_a_wins}승")
    is_claim3_correct = (matchup_count == 26 and matchup_a_wins == 25)
    print(f"  26전 25승? {'예 ✓' if is_claim3_correct else '아니오 ✗'}")
    if not is_claim3_correct:
        wrong_statements.append("보기 3")
    
    # --------------------------------------------------------
    # 보기 4: dgreg는 전방이 후방보다 승률 높음
    # --------------------------------------------------------
    print("\n--- 보기 4: dgreg 전방 vs 후방 승률 ---")
    
    dgreg_front_wins = 0
    dgreg_front_total = 0
    dgreg_rear_wins = 0
    dgreg_rear_total = 0
    
    battles_multi = [b for b in data if len(b['blue']) >= 2]
    
    for b in battles_multi:
        blue_units = get_team_units(b, 'blue')
        red_units = get_team_units(b, 'red')
        bc = team_center(blue_units)
        rc = team_center(red_units)
        
        # Blue 팀 dgreg
        for u in blue_units:
            if u['type'] == 'dgreg':
                pos = classify_position(u, bc, rc)
                if pos == 'front':
                    dgreg_front_total += 1
                    if b['winner'] == 'blue':
                        dgreg_front_wins += 1
                elif pos == 'rear':
                    dgreg_rear_total += 1
                    if b['winner'] == 'blue':
                        dgreg_rear_wins += 1
        
        # Red 팀 dgreg
        for u in red_units:
            if u['type'] == 'dgreg':
                pos = classify_position(u, rc, bc)
                if pos == 'front':
                    dgreg_front_total += 1
                    if b['winner'] == 'red':
                        dgreg_front_wins += 1
                elif pos == 'rear':
                    dgreg_rear_total += 1
                    if b['winner'] == 'red':
                        dgreg_rear_wins += 1
    
    dgreg_fr = dgreg_front_wins / dgreg_front_total if dgreg_front_total > 0 else 0
    dgreg_rr = dgreg_rear_wins / dgreg_rear_total if dgreg_rear_total > 0 else 0
    print(f"  dgreg 전방: {dgreg_fr:.4f} ({dgreg_front_wins}/{dgreg_front_total})")
    print(f"  dgreg 후방: {dgreg_rr:.4f} ({dgreg_rear_wins}/{dgreg_rear_total})")
    print(f"  전방 > 후방? {'예 ✓' if dgreg_fr > dgreg_rr else '아니오 ✗'}")
    if dgreg_fr <= dgreg_rr:
        wrong_statements.append("보기 4")
    
    # --------------------------------------------------------
    # 보기 5: 같은 팀 유닛 간 거리 가까울수록 승률 높음
    # --------------------------------------------------------
    print("\n--- 보기 5: 팀 내 유닛 간 거리 vs 승률 ---")
    
    intra_dist_win = []
    intra_dist_lose = []
    
    for b in data:
        for team in ['blue', 'red']:
            units = get_team_units(b, team)
            if len(units) < 2:
                continue
            
            # 유닛 간 평균 거리
            dists = []
            for i in range(len(units)):
                for j in range(i + 1, len(units)):
                    d = np.sqrt((units[i]['x'] - units[j]['x'])**2 + 
                                (units[i]['y'] - units[j]['y'])**2)
                    dists.append(d)
            avg_dist = np.mean(dists)
            
            if b['winner'] == team:
                intra_dist_win.append(avg_dist)
            else:
                intra_dist_lose.append(avg_dist)
    
    avg_win_dist = np.mean(intra_dist_win)
    avg_lose_dist = np.mean(intra_dist_lose)
    print(f"  승리팀 평균 유닛간 거리: {avg_win_dist:.4f}")
    print(f"  패배팀 평균 유닛간 거리: {avg_lose_dist:.4f}")
    
    # 구간별 분석
    all_intra = [(d, True) for d in intra_dist_win] + [(d, False) for d in intra_dist_lose]
    all_intra.sort(key=lambda x: x[0])
    n = len(all_intra)
    quartiles = [all_intra[:n//4], all_intra[n//4:n//2], all_intra[n//2:3*n//4], all_intra[3*n//4:]]
    
    print(f"\n  유닛간 거리 구간별 승률:")
    closer_wins_more = True
    prev_rate = None
    for i, q in enumerate(quartiles):
        wins = sum(1 for _, w in q if w)
        rate = wins / len(q) if len(q) > 0 else 0
        dist_range = f"[{q[0][0]:.2f}, {q[-1][0]:.2f}]"
        print(f"  Q{i+1} {dist_range}: {rate:.4f} ({wins}/{len(q)})")
        if prev_rate is not None and rate > prev_rate:
            closer_wins_more = False
        prev_rate = rate
    
    corr = np.corrcoef(
        [d for d, _ in all_intra],
        [1.0 if w else 0.0 for _, w in all_intra]
    )[0, 1]
    print(f"  거리-승리 상관계수: {corr:.4f}")
    
    trend = "가까울수록 승률↑ ✓" if corr < 0 and closer_wins_more else "경향 아님 ✗"
    print(f"  판정: {trend}")
    if corr >= 0 or not closer_wins_more:
        wrong_statements.append("보기 5")
    
    print(f"\n★ 올바르지 않은 보기: {wrong_statements}")
    return wrong_statements

# ============================================================
# 메인 실행
# ============================================================

if __name__ == '__main__':
    print("데이터 로드 중...")
    data = load_data()
    print(f"전투 {len(data)}건 로드 완료.\n")
    
    q1 = solve_q1(data)
    q2 = solve_q2(data)
    q3 = solve_q3(data)
    q4 = solve_q4(data)
    q5 = solve_q5(data)
    
    print("\n" + "=" * 60)
    print("종합 결과")
    print("=" * 60)
    print(f"문제 1 정답: {q1}")
    print(f"문제 2 정답: {q2}")
    print(f"문제 3 정답: {q3}")
    print(f"문제 4 정답 (틀린 것): {q4}")
    print(f"문제 5 정답 (틀린 것): {q5}")
