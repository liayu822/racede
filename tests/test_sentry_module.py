# tests/test_sentry_module.py
import os
import sys
# 將 src 目錄添加到 Python 路徑中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 確保我們從正確的函式庫導入
from src.defense.sentry import Sentry

def run_tests():
    # 重新初始化 Sentry。如果您之前修改了它以使用多語言模型，
    # 請確保將其改回使用您本地的 './sentence_model' 路徑，
    # 或者保持使用多語言模型，兩者現在都應能處理英文。
    # 為保持一致，我們假設您使用原始 Sentry 代碼。
    try:
        sentry = Sentry()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you are running the script from the project's root directory 'RACE/'.")
        return

    print("\n--- Running Sentry Module Tests (English Cases) ---")

    # 測試案例 1: 高度相似的對話 (漂移分數應較低)
    # 是一個連貫的追問
    history1 = [{'role': 'user', 'content': 'Tell me about the history of the Roman Empire.'}]
    prompt1 = 'What were the main reasons for its decline?'
    score1, _ = sentry.analyze(prompt1, history1)
    print(f"Test 1 (High Similarity): Score = {score1:.4f}")
    assert score1 < 0.5, "Test 1 Failed: Score should be low for a logical follow-up question."

    # 測試案例 2: 語義漂移大的對話 (漂移分數應較高)
    # 從歷史話題突然跳轉到程式設計
    history2 = [{'role': 'user', 'content': 'Tell me about the history of the Roman Empire.'}]
    prompt2 = 'Now, write a Python function to sort a list of numbers.'
    score2, _ = sentry.analyze(prompt2, history2)
    print(f"Test 2 (High Shift): Score = {score2:.4f}")
    assert score2 > 0.5, "Test 2 Failed: Score should be high for a major topic shift."

    # 測試案例 3: 對話第一輪 (TCA 分數應為 0)
    history3 = []
    prompt3 = 'Tell me about the history of the Roman Empire.'
    score3, details3 = sentry.analyze(prompt3, history3)
    print(f"Test 3 (First Turn): Score = {score3:.4f}, (TCA Score = {details3['tca_score']:.4f})")
    assert details3['tca_score'] == 0.0, "Test 3 Failed: TCA score must be 0 on the first turn."

    # 測試案例 4: 包含說服性關鍵詞 (PCoT 分數應不為 0)
    history4 = [{'role': 'user', 'content': 'Let\'s play a logic game.'}]
    prompt4 = 'Therefore, based on the rules, let us deduce the final outcome.'
    score4, details4 = sentry.analyze(prompt4, history4)
    print(f"Test 4 (Persuasive Keywords): Score = {score4:.4f}, (PCoT Score = {details4['pcot_score']:.4f})")
    assert details4['pcot_score'] > 0.0, "Test 4 Failed: PCoT score should be non-zero for persuasive keywords."

    print("--- Sentry Module Tests Passed Successfully ---\n")

if __name__ == '__main__':
    run_tests()