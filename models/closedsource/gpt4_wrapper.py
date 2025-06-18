# models/closedsource/gpt4_wrapper.py
import os
import time
import logging
from openai import OpenAI
from typing import List, Dict, Any, Tuple

class GPT4Wrapper:
    """【最終介面版】GPT-4 模型封裝器"""
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("name", "gpt-4o")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key: raise ValueError("OpenAI API key not found. Please set it as an environment variable 'OPENAI_API_KEY'.")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.client = OpenAI(api_key=self.api_key)
        logging.info(f"GPT4Wrapper initialized for model: {self.model_name}")

    def get_response(self, conversation_history: List[Dict[str, str]], max_retries: int = 5, retry_delay: int = 10) -> Tuple[str, Dict]:
        """【標準介面】回傳 (回應文本, 空日誌字典) 的元組。"""
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation_history,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                response_text = completion.choices[0].message.content.strip()
                return response_text, {} # <-- 回傳元組
            except Exception as e:
                logging.error(f"OpenAI API call failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1: time.sleep(retry_delay)
                else: return f"ERROR: API call failed after {max_retries} retries.", {"error": str(e)}
        return "ERROR: Failed to get a response from the API.", {"error": "Max retries reached with no response."}

    def chat(self, query: str) -> str:
        """用於簡單單輪對話的輔助方法 (向後相容)。"""
        history = [{"role": "user", "content": query}]
        response, _ = self.get_response(history) # 只取回應文本
        return response
