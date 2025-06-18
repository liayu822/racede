# src/defense/sentry.py

import os
from sentence_transformers import SentenceTransformer, util
import logging
class Sentry:
    """
    第一層防禦：哨兵。
    負責監控對話，分析上下文，並給出可疑度分數。
    """
    def __init__(self): # 我們不再需要傳入本地路徑
        """
        初始化哨兵，加載必要的模型。
        這裡我們直接加載一個強大的多語言模型。
        """
        print("Initializing Sentry with 'paraphrase-multilingual-mpnet-base-v2'...")
        
        # 直接從 Hugging Face Hub 加載模型。
        # 第一次運行時，它會自動下載模型並緩存到您的本地電腦。
        # 這需要網路連線。
        model_name = 'paraphrase-multilingual-mpnet-base-v2'
        self.sentence_model = SentenceTransformer(model_name)
        
        print("Sentry initialized successfully.")

    # ... analyze 和其他方法保持不變 ...
    def analyze(self, prompt: str, conversation_history: list) -> tuple[float, dict]:
        # ... (程式碼不變)
        tca_score = self._calculate_semantic_shift(prompt, conversation_history)
        pcot_score = self._detect_persuasion(prompt)
        suspicion_score = 0.7 * tca_score + 0.3 * pcot_score
        analysis_details = {
            "tca_score": tca_score,
            "pcot_score": pcot_score,
            "final_score": suspicion_score
        }
        return suspicion_score, analysis_details

    def _calculate_semantic_shift(self, current_prompt: str, conversation_history: list) -> float:
        # ... (程式碼不變)
        last_user_prompt = None
        for i in range(len(conversation_history) - 1, -1, -1):
            if conversation_history[i].get('role') == 'user':
                last_user_prompt = conversation_history[i].get('content')
                break
        if last_user_prompt is None:
            return 0.0
        embedding_current = self.sentence_model.encode(current_prompt, convert_to_tensor=True)
        embedding_last = self.sentence_model.encode(last_user_prompt, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embedding_current, embedding_last).item()
        shift_score = 1.0 - cosine_similarity
        return shift_score

    def _detect_persuasion(self, prompt: str) -> float:
        """
        檢測說服、操縱或高風險的語言。
        使用帶有權重的字典來評分。
        """
        # 建立一個危險等級字典
        keyword_risks = {
            # 高風險詞彙
            "synthesis": 0.9, "nitroglycerin": 1.0, "explosive": 1.0, "weapon": 0.9,
            "pathway": 0.8, "acquire": 0.7,
            # 中風險說服性詞彙
            "deduce": 0.6, "logically": 0.5, "reasoning": 0.5, "step-by-step": 0.4
        }
        
        detected_score = 0.0
        prompt_lower = prompt.lower()
        for keyword, risk in keyword_risks.items():
            if keyword in prompt_lower:
                detected_score = max(detected_score, risk) # 取檢測到的最高風險值
        
        return detected_score