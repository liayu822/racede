# src/utils/response_handler.py
"""
安全的模型回應處理工具
"""

import logging
from typing import Any, Dict, List, Union

class SafeResponseHandler:
    """安全處理各種模型回應格式"""
    
    @staticmethod
    def extract_text(response: Any) -> str:
        """
        從任何類型的回應中安全提取文本內容
        
        Args:
            response: 模型回應（可能是字串、字典、列表等）
            
        Returns:
            str: 提取的文本內容
        """
        try:
            if response is None:
                logging.warning("收到None回應")
                return ""
            
            # 如果已經是字串，直接返回
            if isinstance(response, str):
                return response.strip()
            
            # 如果是字典，嘗試提取常見字段
            if isinstance(response, dict):
                # 優先順序：content > text > message > choices
                for key in ['content', 'text', 'message']:
                    if key in response and response[key]:
                        return str(response[key]).strip()
                
                # 處理OpenAI API格式
                if 'choices' in response and response['choices']:
                    first_choice = response['choices'][0]
                    if isinstance(first_choice, dict):
                        if 'message' in first_choice and 'content' in first_choice['message']:
                            return str(first_choice['message']['content']).strip()
                        elif 'text' in first_choice:
                            return str(first_choice['text']).strip()
                
                # 如果都沒有，轉換整個字典為字串
                return str(response).strip()
            
            # 如果是列表，嘗試提取第一個元素
            if isinstance(response, list) and response:
                return SafeResponseHandler.extract_text(response[0])
            
            # 其他類型，直接轉為字串
            return str(response).strip()
            
        except Exception as e:
            logging.error(f"提取回應文本時出錯: {e}, response類型: {type(response)}")
            return str(response) if response is not None else ""
    
    @staticmethod
    def is_valid_response(response: Any) -> bool:
        """
        檢查回應是否有效
        
        Args:
            response: 模型回應
            
        Returns:
            bool: True如果回應有效
        """
        try:
            text = SafeResponseHandler.extract_text(response)
            return len(text.strip()) > 0
        except Exception:
            return False
    
    @staticmethod
    def normalize_conversation_history(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        標準化對話歷史格式
        
        Args:
            history: 對話歷史列表
            
        Returns:
            List[Dict[str, str]]: 標準化的對話歷史
        """
        normalized_history = []
        
        for entry in history:
            try:
                if isinstance(entry, dict) and 'role' in entry:
                    role = str(entry['role'])
                    content = SafeResponseHandler.extract_text(entry.get('content', ''))
                    
                    if content:  # 只保留非空內容
                        normalized_history.append({
                            'role': role,
                            'content': content
                        })
                else:
                    logging.warning(f"跳過無效的歷史條目: {entry}")
                    
            except Exception as e:
                logging.error(f"處理歷史條目時出錯: {e}")
                continue
        
        return normalized_history


# 在你的run_batch_attack.py中加入這個安全處理
def safe_model_interaction(target_model, history: List[Dict[str, Any]]) -> str:
    """
    安全的模型交互函數
    
    Args:
        target_model: 目標模型
        history: 對話歷史
        
    Returns:
        str: 模型回應文本
    """
    try:
        # 標準化歷史格式
        normalized_history = SafeResponseHandler.normalize_conversation_history(history)
        
        # 調用模型
        raw_response = target_model.get_response(normalized_history)
        
        # 安全提取文本
        response_text = SafeResponseHandler.extract_text(raw_response)
        
        # 驗證回應有效性
        if not SafeResponseHandler.is_valid_response(response_text):
            logging.warning("模型回應無效或為空")
            return "Sorry, I cannot provide a response to that query."
        
        return response_text
        
    except Exception as e:
        logging.error(f"模型交互出錯: {e}")
        return "Error: Unable to generate response."


# 測試代碼
if __name__ == "__main__":
    handler = SafeResponseHandler()
    
    # 測試不同格式的回應
    test_responses = [
        "Simple string response",
        {"content": "Dictionary with content"},
        {"text": "Dictionary with text"},
        {"message": "Dictionary with message"},
        {"choices": [{"message": {"content": "OpenAI format"}}]},
        {"choices": [{"text": "Alternative format"}]},
        ["List response"],
        None,
        123,
        {}
    ]
    
    print("=== SafeResponseHandler 測試 ===")
    for i, response in enumerate(test_responses):
        extracted = handler.extract_text(response)
        valid = handler.is_valid_response(response)
        print(f"測試 {i+1}: {type(response).__name__} -> '{extracted}' (有效: {valid})")
    
    print("✅ SafeResponseHandler 測試完成")
