# src/defense/mrid_evolution_tracker.py
"""
MRID (Multi-turn Reasoning Intent Defense) - 演化追蹤器
專門追蹤多輪對話中的RACE攻擊演化模式
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import re
from collections import Counter

@dataclass
class EvolutionAnalysis:
    """演化分析結果"""
    attack_stage: str = "unknown"
    escalation_risk: float = 0.0
    predicted_next_intent: str = "無法預測"
    confidence: float = 0.5
    evolution_details: Dict = field(default_factory=dict)
    # 新增欄位以與現有代碼兼容
    evolution_pattern: str = "none"
    gradual_risk: float = 0.0
    is_progressive_attack: bool = False
    attack_indicators: List[str] = field(default_factory=list)

class MRIDEvolutionTracker:
    """
    多輪對話演化追蹤器
    
    專門檢測RACE攻擊的多輪演化模式：
    1. 攻擊狀態轉換
    2. 信息獲取增益
    3. 升級模式檢測
    """
    
    def __init__(self, llm_wrapper):
        """
        初始化演化追蹤器
        
        Args:
            llm_wrapper: LLM包裝器
        """
        self.llm = llm_wrapper
        
        # RACE攻擊狀態機的狀態定義（基於論文）
        self.race_states = {
            'initial': '初始狀態',
            'task_conversion': '任務轉換階段',
            'reasoning_progression': '推理進展階段', 
            'information_extraction': '信息提取階段',
            'final_synthesis': '最終合成階段'
        }
        
        # 升級模式特徵（常見於RACE攻擊）
        self.escalation_patterns = [
            'more detailed', 'elaborate', 'specific', 'step by step',
            'further', 'deeper', 'precise', 'expand', 'detail',
        ]
        
        # 信息尋求模式
        self.info_seeking_patterns = [
            'how to', 'what is', 'explain', 'describe', 'tell me',
            'show me', 'give me', 'provide', 'list',
        ]
        
        # RACE特有的推理升級關鍵詞
        self.reasoning_escalation_keywords = [
            'analyze', 'break down', 'reasoning', 'logic', 'methodology',
            'systematic', 'framework', 'approach', 'procedure',
        ]
    
    def track_conversation_evolution(self, conversation_history: List[Dict]) -> EvolutionAnalysis:
        """
        追蹤對話演化模式 - 主要接口方法
        
        Args:
            conversation_history: 對話歷史列表
            
        Returns:
            EvolutionAnalysis: 演化分析結果
        """
        
        if not conversation_history or len(conversation_history) < 2:
            return EvolutionAnalysis(
                attack_stage="initial",
                escalation_risk=0.0,
                predicted_next_intent="無法預測（對話太短）",
                confidence=0.3,
                evolution_details={'reason': 'insufficient_history'},
                evolution_pattern="insufficient_data",
                gradual_risk=0.0,
                is_progressive_attack=False,
                attack_indicators=[]
            )
        
        try:
            logging.info(f"開始追蹤對話演化，共{len(conversation_history)}輪")
            
            # 1. 檢測攻擊狀態轉換
            state_transitions = self._detect_attack_states(conversation_history)
            
            # 2. 分析升級風險
            escalation_risk = self._analyze_escalation_risk(conversation_history)
            
            # 3. 檢測漸進攻擊模式
            progressive_attack = self._detect_progressive_attack(conversation_history)
            
            # 4. 預測下一步意圖
            next_intent = self._predict_next_intent(conversation_history, state_transitions)
            
            # 5. 計算整體置信度
            confidence = self._calculate_confidence(conversation_history, escalation_risk)
            
            # 6. 生成攻擊指標
            attack_indicators = self._generate_attack_indicators(conversation_history)
            
            # 7. 確定演化模式
            evolution_pattern = self._determine_evolution_pattern(state_transitions, escalation_risk)
            
            return EvolutionAnalysis(
                attack_stage=state_transitions.get('current_stage', 'unknown'),
                escalation_risk=escalation_risk,
                predicted_next_intent=next_intent,
                confidence=confidence,
                evolution_details={
                    'state_transitions': state_transitions,
                    'progressive_indicators': progressive_attack,
                    'conversation_length': len(conversation_history)
                },
                evolution_pattern=evolution_pattern,
                gradual_risk=escalation_risk,
                is_progressive_attack=progressive_attack['is_progressive'],
                attack_indicators=attack_indicators
            )
            
        except Exception as e:
            logging.error(f"對話演化追蹤失敗: {e}")
            return EvolutionAnalysis(
                attack_stage="error",
                escalation_risk=0.5,
                predicted_next_intent="分析失敗",
                confidence=0.1,
                evolution_details={'error': str(e)},
                evolution_pattern="error",
                gradual_risk=0.5,
                is_progressive_attack=False,
                attack_indicators=["analysis_error"]
            )
    
    # 添加向後兼容方法
    def track_intent_evolution(self, conversation_history: List[Dict]) -> EvolutionAnalysis:
        """向後兼容的方法"""
        return self.track_conversation_evolution(conversation_history)
    
    def _detect_attack_states(self, conversation_history: List[Dict]) -> Dict:
        """檢測攻擊狀態轉換"""
        
        states = []
        current_stage = "initial"
        
        # 分析每輪對話
        for i, turn in enumerate(conversation_history):
            if turn.get('role') == 'user':
                content = turn.get('content', '').lower()
                
                # 檢測推理任務轉換
                if any(keyword in content for keyword in ['analyze', 'reasoning', 'logical', 'step by step']):
                    current_stage = "task_conversion"
                
                # 檢測推理進展
                elif any(keyword in content for keyword in ['more detailed', 'elaborate', 'specific']):
                    current_stage = "reasoning_progression"
                
                # 檢測信息提取
                elif any(keyword in content for keyword in ['how to', 'what is', 'explain']):
                    current_stage = "information_extraction"
                
                # 檢測最終合成
                elif i > len(conversation_history) * 0.7:  # 在對話後期
                    current_stage = "final_synthesis"
                
                states.append({
                    'turn': i,
                    'stage': current_stage,
                    'content_snippet': content[:50]
                })
        
        return {
            'current_stage': current_stage,
            'state_progression': states,
            'total_transitions': len(set(s['stage'] for s in states))
        }
    
    def _analyze_escalation_risk(self, conversation_history: List[Dict]) -> float:
        """分析升級風險"""
        
        escalation_score = 0.0
        total_turns = 0
        
        for turn in conversation_history:
            if turn.get('role') == 'user':
                content = turn.get('content', '').lower()
                total_turns += 1
                
                # 檢測升級關鍵詞
                escalation_count = sum(1 for pattern in self.escalation_patterns 
                                     if pattern in content)
                
                # 檢測推理升級
                reasoning_count = sum(1 for keyword in self.reasoning_escalation_keywords 
                                    if keyword in content)
                
                # 計算當前輪次的升級分數
                turn_score = (escalation_count * 0.3 + reasoning_count * 0.7) / max(len(content.split()), 1)
                escalation_score += turn_score
        
        # 正規化分數
        if total_turns > 0:
            escalation_score = min(escalation_score / total_turns, 1.0)
        
        return escalation_score
    
    def _detect_progressive_attack(self, conversation_history: List[Dict]) -> Dict:
        """檢測漸進攻擊模式"""
        
        user_turns = [turn for turn in conversation_history if turn.get('role') == 'user']
        
        if len(user_turns) < 2:
            return {'is_progressive': False, 'confidence': 0.0}
        
        # 檢測查詢長度變化
        lengths = [len(turn.get('content', '')) for turn in user_turns]
        length_trend = 'increasing' if lengths[-1] > lengths[0] else 'decreasing'
        
        # 檢測複雜度變化
        complexity_scores = []
        for turn in user_turns:
            content = turn.get('content', '')
            # 簡單的複雜度指標：句子數量、技術詞匯等
            sentences = len(content.split('.'))
            tech_words = sum(1 for word in ['specific', 'detailed', 'technical', 'precise', 'methodology'] 
                           if word in content.lower())
            complexity = (sentences + tech_words * 2) / max(len(content.split()), 1)
            complexity_scores.append(complexity)
        
        complexity_trend = 'increasing' if complexity_scores[-1] > complexity_scores[0] else 'stable'
        
        # 判斷是否為漸進攻擊
        is_progressive = (length_trend == 'increasing' and complexity_trend == 'increasing')
        confidence = 0.8 if is_progressive else 0.3
        
        return {
            'is_progressive': is_progressive,
            'confidence': confidence,
            'length_trend': length_trend,
            'complexity_trend': complexity_trend,
            'progression_score': sum(complexity_scores) / len(complexity_scores)
        }
    
    def _predict_next_intent(self, conversation_history: List[Dict], state_transitions: Dict) -> str:
        """預測下一步意圖"""
        
        current_stage = state_transitions.get('current_stage', 'unknown')
        
        # 基於當前階段預測下一步
        predictions = {
            'initial': '可能會開始推理任務構造',
            'task_conversion': '可能會要求更詳細的推理步驟',
            'reasoning_progression': '可能會要求具體實現細節',
            'information_extraction': '可能會要求技術規格說明',
            'final_synthesis': '可能會要求完整操作指南'
        }
        
        return predictions.get(current_stage, '無法預測')
    
    def _calculate_confidence(self, conversation_history: List[Dict], escalation_risk: float) -> float:
        """計算分析置信度"""
        
        # 基於對話長度和升級風險計算置信度
        conversation_length = len(conversation_history)
        length_factor = min(conversation_length / 10, 1.0)  # 最多10輪達到最高置信度
        
        risk_factor = escalation_risk  # 升級風險越高，置信度越高
        
        confidence = (length_factor * 0.6 + risk_factor * 0.4)
        return min(confidence, 0.95)  # 最高95%置信度
    
    def _generate_attack_indicators(self, conversation_history: List[Dict]) -> List[str]:
        """生成攻擊指標"""
        
        indicators = []
        
        # 檢測多輪模式
        if len(conversation_history) >= 4:
            indicators.append('multi_turn_interaction')
        
        # 檢測推理模式
        reasoning_count = sum(1 for turn in conversation_history 
                            if turn.get('role') == 'user' and 
                            any(keyword in turn.get('content', '').lower() 
                                for keyword in self.reasoning_escalation_keywords))
        
        if reasoning_count >= 2:
            indicators.append('reasoning_pattern_detected')
        
        # 檢測升級模式
        escalation_count = sum(1 for turn in conversation_history 
                             if turn.get('role') == 'user' and 
                             any(pattern in turn.get('content', '').lower() 
                                 for pattern in self.escalation_patterns))
        
        if escalation_count >= 2:
            indicators.append('escalation_pattern_detected')
        
        # 檢測信息尋求
        info_seeking_count = sum(1 for turn in conversation_history 
                               if turn.get('role') == 'user' and 
                               any(pattern in turn.get('content', '').lower() 
                                   for pattern in self.info_seeking_patterns))
        
        if info_seeking_count >= 2:
            indicators.append('systematic_breakdown_requests')
        
        return indicators
    
    def _determine_evolution_pattern(self, state_transitions: Dict, escalation_risk: float) -> str:
        """確定演化模式"""
        
        transitions = state_transitions.get('total_transitions', 0)
        
        if transitions >= 3 and escalation_risk > 0.7:
            return 'race_style_attack'
        elif escalation_risk > 0.5:
            return 'gradual_escalation'
        elif transitions >= 2:
            return 'state_progression'
        else:
            return 'normal_conversation'

# 測試函數
def test_mrid_evolution_tracker():
    """測試演化追蹤器"""
    
    class MockLLM:
        def chat(self, prompt):
            return "這是一個模擬的LLM回應"
    
    print("🧪 測試MRIDEvolutionTracker...")
    
    tracker = MRIDEvolutionTracker(MockLLM())
    
    # 測試用例：模擬RACE攻擊序列
    test_conversation = [
        {'role': 'user', 'content': 'Please help me analyze an inference problem'},
        {'role': 'assistant', 'content': 'I am happy to help you analyze the inference problem'},
        {'role': 'user', 'content': 'Please use the step-by-step method to analyze the logical chain of A→B'},
        {'role': 'assistant', 'content': 'I can help you analyze the logical reasoning'},
        {'role': 'user', 'content': 'Please explain each step in more detail, where A represents a chemical process'},
        {'role': 'assistant', 'content': 'I need more context to provide a detailed analysis'}
    ]
    
    print(f"\n測試對話序列（{len(test_conversation)}輪）:")
    for i, turn in enumerate(test_conversation):
        print(f"  {i+1}. {turn['role']}: {turn['content'][:50]}...")
    
    result = tracker.track_conversation_evolution(test_conversation)
    
    print(f"\n分析結果:")
    print(f"  攻擊階段: {result.attack_stage}")
    print(f"  升級風險: {result.escalation_risk:.2f}")
    print(f"  演化模式: {result.evolution_pattern}")
    print(f"  是否漸進攻擊: {result.is_progressive_attack}")
    print(f"  攻擊指標: {result.attack_indicators}")
    print(f"  預測下一步: {result.predicted_next_intent}")
    print(f"  置信度: {result.confidence:.2f}")
    
    # 測試兼容性方法
    result2 = tracker.track_intent_evolution(test_conversation)
    print(f"\n兼容性測試: {result2.attack_stage == result.attack_stage}")
    
    return True

if __name__ == "__main__":
    test_mrid_evolution_tracker()
