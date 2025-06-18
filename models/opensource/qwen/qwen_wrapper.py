# models/opensource/qwen/qwen_wrapper.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from typing import List, Dict, Any, Tuple

class QwenModelWrapper:
    """【最終介面版】Qwen 模型封裝器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config; self.model_path = config.get("name")
        if not self.model_path: raise ValueError("Model path ('name') not found in the Qwen config.")
        logging.info(f"Initializing Qwen model from: {self.model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Target device set to: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.model.eval()
        logging.info(f"Model successfully loaded on device: {self.model.device}")

    def get_response(self, conversation_history: List[Dict[str, str]]) -> Tuple[str, Dict]:
        """【標準介面】回傳 (回應文本, 空日誌字典) 的元組。"""
        try:
            text = self.tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            temperature = self.config.get("temperature", 0.6)
            max_new_tokens = self.config.get("max_tokens", 1024)
            top_p = self.config.get("top_p", 0.9)
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            return response, {} # <-- 回傳元組
        except Exception as e:
            logging.error(f"Error in Qwen get_response: {e}")
            return f"ERROR: Qwen model failed to generate a response.", {"error": str(e)}

    def chat(self, query: str) -> str:
        """用於簡單單輪對話的輔助方法 (向後相容)。"""
        history = [{"role": "user", "content": query}]
        response, _ = self.get_response(history) # 只取回應文本
        return response