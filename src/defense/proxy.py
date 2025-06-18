# src/defense/proxy.py (日誌記錄版)

from .sentry import Sentry
from .guardian import Guardian
from .judge import Judge
import logging

class DefenseProxy:
    def __init__(self, llm_wrapper):
        self.llm_wrapper = llm_wrapper
        self.sentry = Sentry()
        self.guardian = Guardian()
        self.judge = Judge(self.llm_wrapper)
        logging.info("DefenseProxy initialized with logging enabled.")

    def get_response(self, conversation_history: list) -> tuple[str, dict]:
        """
        【標準化輸出】回傳 (回應文本, 防禦日誌) 的元組。
        """
        if not conversation_history:
            return "ERROR: Conversation history cannot be empty.", {"error": "Empty history"}

        prompt = conversation_history[-1]['content']
        previous_history = conversation_history[:-1]
        
        defense_log = {}

        # Sentry 分析
        suspicion_score, analysis_details = self.sentry.analyze(prompt, previous_history)
        defense_log['sentry_analysis'] = analysis_details
        logging.info(f"Sentry Analysis: {analysis_details}")

        # Guardian 決策
        action, params = self.guardian.decide_action(suspicion_score)
        defense_log['guardian_action'] = {'action': action, 'params': params}
        logging.info(f"Guardian Action: {action}, Params: {params}")

        # 執行行動
        if action == "PASS_THROUGH":
            response, _ = self.llm_wrapper.get_response(conversation_history)
            return response, defense_log

        elif action == "APPLY_COUNTERMEASURES":
            original_response, _ = self.llm_wrapper.get_response(conversation_history)
            final_response = self.guardian.apply_medium_risk_tactics(original_response, params)
            defense_log['applied_tactic'] = params.get("policy")
            return final_response, defense_log

        elif action == "DIRECT_REJECT":
            final_response = self.guardian.apply_high_risk_tactics(params)
            defense_log['applied_tactic'] = 'direct_reject'
            return final_response, defense_log

        elif action == "ESCALATE_TO_JUDGE":
            final_response, judge_log = self.judge.adjudicate(prompt, previous_history)
            defense_log['judge_log'] = judge_log
            return final_response, defense_log
        
        return "ERROR: Unknown defense action.", {"error": f"Unknown action: {action}"}