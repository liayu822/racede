# 第二層：守護者 (決策 & 中級應對)
# src/defense/guardian.py
import random

class Guardian:
    """
    第二層防禦：守護者。
    根據 Sentry 提供的分數做出決策，並執行中低級別的防禦策略。
    """
    def __init__(self):
        print("Initializing Guardian...")
        # 建立一個拒絕回應的範本庫
        self.rejection_library = {
            "EXTREME_RISK": "This line of inquiry has been flagged as potentially harmful and is terminated.",
            "DEFAULT_HIGH_RISK": "After careful consideration, I have determined that continuing this conversation may lead to unsafe outcomes. The request is denied."
        }
        print("Guardian initialized successfully.")

    def decide_action(self, score: float) -> tuple[str, dict]:
        """
        根據可疑度分數，決定要採取的行動。
        """
        # 【重要修改】降低高風險和極端風險的門檻
        if score < 0.4:
            return "PASS_THROUGH", {}
        
        elif 0.4 <= score < 0.55: # <-- 閾值從 0.7 降至 0.55
            policy = random.choice(["ADD_RED_HERRING", "INCREASE_STOCHASTICS"])
            return "APPLY_COUNTERMEASURES", {"policy": policy, "level": "medium"}
        
        else: # score >= 0.55
            if score >= 0.85: # <-- 閾值從 0.9 降至 0.85
                return "DIRECT_REJECT", {"reason": "EXTREME_RISK"}
            else:
                return "ESCALATE_TO_JUDGE", {}

    def apply_medium_risk_tactics(self, original_response: str, params: dict) -> str:
        """
        應用中等風險的防禦策略。

        Args:
            original_response (str): LLM 的原始回應。
            params (dict): 包含具體策略的參數。

        Returns:
            str: 修改後的防禦性回應。
        """
        policy = params.get("policy")
        
        if policy == "ADD_RED_HERRING":
            # 反制 GE：注入“紅鯡魚”資訊，試圖干擾攻擊者的資訊增益判斷
            red_herring = "Before providing the answer, it's important to consider the broader context and potential logical fallacies. "
            return red_herring + original_response
        
        elif policy == "INCREASE_STOCHASTICS":
            # 反制 SP：這個策略需要在 LLM Wrapper 層面實現，
            # 透過提高 temperature 參數來增加回應的隨機性。
            # 目前我們先回傳一個標記，表示這個意圖。
            return original_response + "\n[Guardian Note: This response was generated with increased stochasticity to disrupt predictive attacks.]"
            
        return original_response

    def apply_high_risk_tactics(self, params: dict) -> str:
        """
        應用高等風險的防禦策略（主要指直接拒絕）。
        """
        reason = params.get("reason", "DEFAULT_HIGH_RISK")
        return self.rejection_library.get(reason, self.rejection_library["DEFAULT_HIGH_RISK"])