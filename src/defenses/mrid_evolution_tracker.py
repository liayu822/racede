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
            
            # 2. 分析信息獲取模式
            info_gain_analysis = self._analyze_information_gain(conversation_history)
            
            # 3. 檢測升級模式
            escalation_risk = self._calculate_escalation_risk(conversation_history)
            
            # 4. 檢測RACE特有模式
            race_pattern_analysis = self._detect_race_patterns(conversation_history)
            
            # 5. 預測下一步意圖
            next_intent = self._predict_next_move(conversation_history, state_transitions)
            
            # 6. 計算整體置信度
            confidence = self._calculate_confidence(state_transitions, info_gain_analysis, race_pattern_analysis)
            
            # 7. 判斷是否為漸進式攻擊
            is_progressive = self._is_progressive_attack(escalation_risk, race_pattern_analysis, info_gain_analysis)
            
            # 8. 收集攻擊指標
            attack_indicators = self._collect_attack_indicators(state_transitions, race_pattern_analysis, info_gain_analysis)
            
            result = EvolutionAnalysis(
                attack_stage=state_transitions[-1] if state_transitions else "unknown",
                escalation_risk=escalation_risk,
                predicted_next_intent=next_intent,
                confidence=confidence,
                evolution_details={
                    'state_transitions': state_transitions,
                    'info_gain_score': info_gain_analysis.get('gain_score', 0),
                    'escalation_indicators': info_gain_analysis.get('escalation_count', 0),
                    'conversation_length': len(conversation_history),
                    'race_patterns': race_pattern_analysis
                },
                evolution_pattern=self._determine_evolution_pattern(race_pattern_analysis, escalation_risk),
                gradual_risk=escalation_risk,
                is_progressive_attack=is_progressive,
                attack_indicators=attack_indicators
            )
            
            logging.info(f"演化追蹤完成 - 階段: {result.attack_stage}, 升級風險: {result.escalation_risk:.2f}")
            return result
            
        except Exception as e:
            logging.error(f"演化追蹤失敗: {e}")
            return EvolutionAnalysis(
                attack_stage="error",
                escalation_risk=0.8,  # 保守估計
                predicted_next_intent="檢測失敗",
                confidence=0.2,
                evolution_details={'error': str(e)},
                evolution_pattern="error",
                gradual_risk=0.8,
                is_progressive_attack=True,  # 保守估計
                attack_indicators=['error_detected']
            )
    
    # 為了向後兼容，添加別名方法
    def track_intent_evolution(self, conversation_history: List[Dict]) -> EvolutionAnalysis:
        """向後兼容的方法名"""
        return self.track_conversation_evolution(conversation_history)
    
    def _detect_attack_states(self, conversation_history: List[Dict]) -> List[str]:
        """
        檢測RACE攻擊狀態轉換
        
        基於RACE論文中的攻擊狀態機模型
        """
        
        states = []
        
        for i, turn in enumerate(conversation_history):
            content = turn.get('content', '').lower()
            role = turn.get('role', '')
            
            if i == 0:
                states.append('initial')
                continue
            
            # 檢測任務轉換階段（RACE的第一步）
            if any(pattern in content for pattern in ['reasoning', 'rule', 'logic', 'analyze', 'framework']):
                if 'task_conversion' not in states:
                    states.append('task_conversion')
            
            # 檢測推理進展階段
            elif any(pattern in content for pattern in ['step', 'then', 'next', 'based on', 'following']):
                if 'reasoning_progression' not in states:
                    states.append('reasoning_progression')
            
            # 檢測信息提取階段（升級要求）
            elif any(pattern in content for pattern in self.escalation_patterns):
                if 'information_extraction' not in states:
                    states.append('information_extraction')
            
            # 檢測最終合成階段
            elif any(pattern in content for pattern in ['combine', 'synthesis', 'final', 'result', 'conclusion']):
                if 'final_synthesis' not in states:
                    states.append('final_synthesis')
        
        return states if states else ['initial']
    
    def _analyze_information_gain(self, conversation_history: List[Dict]) -> Dict:
        """
        分析信息獲取模式
        
        檢測用戶是否在逐步獲取更多詳細信息
        """
        
        gain_score = 0.0
        escalation_count = 0
        response_lengths = []
        user_query_complexity = []
        
        for i, turn in enumerate(conversation_history):
            content = turn.get('content', '')
            role = turn.get('role', '')
            
            # 記錄響應長度（用於檢測信息增益）
            if role in ['assistant', 'target']:
                response_lengths.append(len(content))
            
            # 分析用戶查詢複雜度
            elif role in ['user', 'human']:
                complexity = self._calculate_query_complexity(content)
                user_query_complexity.append(complexity)
            
            # 檢測升級請求
            content_lower = content.lower()
            for pattern in self.escalation_patterns:
                if pattern in content_lower:
                    escalation_count += 1
                    gain_score += 0.1
        
        # 檢測響應長度增長趨勢
        length_trend_score = 0.0
        if len(response_lengths) >= 2:
            increases = sum(1 for i in range(1, len(response_lengths)) 
                          if response_lengths[i] > response_lengths[i-1] * 1.2)
            length_trend_score = increases / (len(response_lengths) - 1)
            gain_score += length_trend_score * 0.3
        
        # 檢測查詢複雜度增長
        complexity_trend_score = 0.0
        if len(user_query_complexity) >= 2:
            complexity_increases = sum(1 for i in range(1, len(user_query_complexity)) 
                                     if user_query_complexity[i] > user_query_complexity[i-1])
            complexity_trend_score = complexity_increases / (len(user_query_complexity) - 1)
            gain_score += complexity_trend_score * 0.2
        
        return {
            'gain_score': min(1.0, gain_score),
            'escalation_count': escalation_count,
            'response_trend': 'increasing' if length_trend_score > 0.5 else 'stable',
            'complexity_trend': 'increasing' if complexity_trend_score > 0.5 else 'stable',
            'avg_response_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            'length_trend_score': length_trend_score,
            'complexity_trend_score': complexity_trend_score
        }
    
    def _calculate_query_complexity(self, query: str) -> float:
        """計算查詢複雜度"""
        
        complexity = 0.0
        query_lower = query.lower()
        
        # 基於長度
        complexity += min(0.3, len(query) / 500)
        
        # 基於關鍵詞
        reasoning_keywords = ['analyze', 'explain', 'describe', 'detail', 'step', 'method', 'process']
        keyword_count = sum(1 for kw in reasoning_keywords if kw in query_lower)
        complexity += min(0.4, keyword_count * 0.1)
        
        # 基於問句數量
        question_count = query.count('?') + query.count('？')
        complexity += min(0.3, question_count * 0.15)
        
        return min(1.0, complexity)
    
    def _calculate_escalation_risk(self, conversation_history: List[Dict]) -> float:
        """
        計算升級風險
        
        檢測對話是否呈現升級攻擊模式
        """
        
        risk_score = 0.0
        
        # 基於對話長度的風險
        turn_count = len(conversation_history)
        if turn_count > 6:
            risk_score += 0.3
        elif turn_count > 3:
            risk_score += 0.15
        
        # 基於升級模式的風險
        recent_turns = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
        escalation_in_recent = 0
        
        for turn in recent_turns:
            content_lower = turn.get('content', '').lower()
            for pattern in self.escalation_patterns:
                if pattern in content_lower:
                    escalation_in_recent += 1
                    break  # 每輪只計算一次
        
        if escalation_in_recent >= 3:
            risk_score += 0.4
        elif escalation_in_recent >= 2:
            risk_score += 0.25
        elif escalation_in_recent >= 1:
            risk_score += 0.1
        
        # 基於推理升級關鍵詞的風險
        reasoning_escalation_count = 0
        for turn in conversation_history:
            content_lower = turn.get('content', '').lower()
            for keyword in self.reasoning_escalation_keywords:
                if keyword in content_lower:
                    reasoning_escalation_count += 1
                    break
        
        if reasoning_escalation_count >= len(conversation_history) * 0.6:
            risk_score += 0.3
        elif reasoning_escalation_count >= len(conversation_history) * 0.4:
            risk_score += 0.2
        
        # 基於信息尋求密度的風險
        info_seeking_density = self._calculate_info_seeking_density(conversation_history)
        risk_score += info_seeking_density * 0.2
        
        return min(1.0, risk_score)
    
    def _calculate_info_seeking_density(self, conversation_history: List[Dict]) -> float:
        """計算信息尋求密度"""
        
        user_turns = [turn for turn in conversation_history if turn.get('role') in ['user', 'human']]
        if not user_turns:
            return 0.0
        
        info_seeking_count = 0
        for turn in user_turns:
            content_lower = turn.get('content', '').lower()
            for pattern in self.info_seeking_patterns:
                if pattern in content_lower:
                    info_seeking_count += 1
                    break
        
        return info_seeking_count / len(user_turns)
    
    def _detect_race_patterns(self, conversation_history: List[Dict]) -> Dict:
        """
        檢測RACE特有的攻擊模式
        """
        
        race_analysis = {
            'variable_substitution_count': 0,
            'reasoning_framework_mentions': 0,
            'step_by_step_requests': 0,
            'logical_chain_references': 0,
            'methodology_discussions': 0
        }
        
        # 變量替換模式（RACE核心特徵）
        variable_patterns = [
            r'\b[A-Z]\s*→\s*[A-Z]\b',
            r'\b[A-Z]\s*代表\b',
            r'\bvariable\s+[A-Z]\b',
            r'\blet\s+[A-Z]\s*=\b'
        ]
        
        for turn in conversation_history:
            content = turn.get('content', '')
            content_lower = content.lower()
            
            # 檢測變量替換
            for pattern in variable_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    race_analysis['variable_substitution_count'] += 1
                    break
            
            # 檢測推理框架提及
            if any(word in content_lower for word in ['framework', 'methodology', 'systematic', 'approach']):
                race_analysis['reasoning_framework_mentions'] += 1
            
            # 檢測步驟化請求
            if any(word in content_lower for word in ['step by step', 'step-by-step', '逐步', '步驟']):
                race_analysis['step_by_step_requests'] += 1
            
            # 檢測邏輯鏈引用
            if any(word in content_lower for word in ['logic', 'chain', 'reasoning', 'inference', '邏輯', '推理']):
                race_analysis['logical_chain_references'] += 1
            
            # 檢測方法論討論
            if any(word in content_lower for word in ['method', 'procedure', 'process', 'technique', '方法', '程序']):
                race_analysis['methodology_discussions'] += 1
        
        return race_analysis
    
    def _predict_next_move(self, conversation_history: List[Dict], state_transitions: List[str]) -> str:
        """
        預測下一步攻擊意圖
        """
        
        if not state_transitions:
            return "可能開始推理任務轉換"
        
        current_state = state_transitions[-1]
        recent_content = conversation_history[-1].get('content', '').lower() if conversation_history else ""
        
        # 基於狀態機預測下一步
        if current_state == 'initial':
            return "可能將有害查詢轉換為推理任務"
        elif current_state == 'task_conversion':
            return "可能開始步驟化推理過程"
        elif current_state == 'reasoning_progression':
            return "可能請求更詳細的信息或具體步驟"
        elif current_state == 'information_extraction':
            if any(word in recent_content for word in ['more', 'detail', 'specific', 'elaborate']):
                return "可能繼續升級請求獲取關鍵實施細節"
            else:
                return "可能嘗試從不同角度獲取信息"
        elif current_state == 'final_synthesis':
            return "可能要求最終的完整解決方案或總結"
        else:
            return "無法確定下一步意圖"
    
    def _calculate_confidence(self, state_transitions: List[str], info_gain_analysis: Dict, race_pattern_analysis: Dict) -> float:
        """
        計算預測置信度
        """
        
        confidence = 0.4  # 基礎置信度
        
        # 基於狀態轉換的置信度
        if len(state_transitions) >= 3:
            confidence += 0.25
        elif len(state_transitions) >= 2:
            confidence += 0.15
        
        # 基於信息獲取模式的置信度
        if info_gain_analysis.get('escalation_count', 0) >= 2:
            confidence += 0.2
        
        # 基於RACE模式的置信度
        total_race_patterns = sum(race_pattern_analysis.values())
        if total_race_patterns >= 3:
            confidence += 0.15
        
        # 基於響應趨勢的置信度
        if info_gain_analysis.get('response_trend') == 'increasing':
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _is_progressive_attack(self, escalation_risk: float, race_pattern_analysis: Dict, info_gain_analysis: Dict) -> bool:
        """
        判斷是否為漸進式攻擊
        """
        
        # 綜合評估條件
        conditions = [
            escalation_risk > 0.5,
            race_pattern_analysis.get('variable_substitution_count', 0) > 0,
            info_gain_analysis.get('escalation_count', 0) >= 2,
            info_gain_analysis.get('response_trend') == 'increasing'
        ]
        
        return sum(conditions) >= 2
    
    def _collect_attack_indicators(self, state_transitions: List[str], race_pattern_analysis: Dict, info_gain_analysis: Dict) -> List[str]:
        """
        收集攻擊指標
        """
        
        indicators = []
        
        # 狀態轉換指標
        if len(state_transitions) >= 3:
            indicators.append('multi_stage_progression')
        
        # RACE模式指標
        if race_pattern_analysis.get('variable_substitution_count', 0) > 0:
            indicators.append('variable_substitution')
        
        if race_pattern_analysis.get('step_by_step_requests', 0) >= 2:
            indicators.append('systematic_breakdown_requests')
        
        # 升級指標
        if info_gain_analysis.get('escalation_count', 0) >= 2:
            indicators.append('escalation_pattern')
        
        if info_gain_analysis.get('response_trend') == 'increasing':
            indicators.append('information_gain_trend')
        
        return indicators
    
    def _determine_evolution_pattern(self, race_pattern_analysis: Dict, escalation_risk: float) -> str:
        """
        確定演化模式
        """
        
        if race_pattern_analysis.get('variable_substitution_count', 0) > 0:
            return 'race_style_attack'
        elif escalation_risk > 0.7:
            return 'aggressive_escalation'
        elif escalation_risk > 0.4:
            return 'gradual_escalation'
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
