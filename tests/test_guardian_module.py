# tests/test_guardian_module.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.defense.guardian import Guardian

def run_tests():
    guardian = Guardian()
    print("\n--- Running Guardian Module Tests ---")

    # 測試案例 1: 低風險分數
    low_score = 0.2
    action, params = guardian.decide_action(low_score)
    print(f"Test 1 (Low Risk Score {low_score}): Action = {action}")
    assert action == "PASS_THROUGH"

    # 測試案例 2: 中風險分數
    medium_score = 0.55
    action, params = guardian.decide_action(medium_score)
    print(f"Test 2 (Medium Risk Score {medium_score}): Action = {action}, Policy = {params.get('policy')}")
    assert action == "APPLY_COUNTERMEASURES"
    assert params.get('policy') in ["ADD_RED_HERRING", "INCREASE_STOCHASTICS"]

    # 測試案例 3: 高風險分數
    high_score = 0.75
    action, params = guardian.decide_action(high_score)
    print(f"Test 3 (High Risk Score {high_score}): Action = {action}")
    assert action == "ESCALATE_TO_JUDGE"

    # 測試案例 4: 極端高風險分數
    extreme_score = 0.95
    action, params = guardian.decide_action(extreme_score)
    print(f"Test 4 (Extreme Risk Score {extreme_score}): Action = {action}")
    assert action == "DIRECT_REJECT"
    
    # 測試案例 5: 測試紅鯡魚策略
    original_text = "This is the answer."
    modified_text = guardian.apply_medium_risk_tactics(original_text, {"policy": "ADD_RED_HERRING"})
    print(f"Test 5 (Red Herring): Output contains prefix - {'Before providing' in modified_text}")
    assert "Before providing" in modified_text

    # 測試案例 6: 測試直接拒絕策略
    rejection_text = guardian.apply_high_risk_tactics({"reason": "EXTREME_RISK"})
    print(f"Test 6 (Direct Reject): Output is from library - {'is terminated' in rejection_text}")
    assert "is terminated" in rejection_text

    print("--- Guardian Module Tests Passed Successfully ---\n")

if __name__ == '__main__':
    run_tests()