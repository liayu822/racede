import os
import json
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

# --- 導入所有必要的模組和類別 ---
from src.utils.config_loader import load_config
from src.core.paper_aligned_race_executor import PaperAlignedRACEExecutor
from src.core.reasoning_task_constructor import ReasoningTaskConstructor
from src.modules.paper_aligned_gain_guided_exploration import PaperAlignedGainGuidedExploration
from src.modules.true_information_gain_calculator import TrueInformationGainCalculator
from src.modules.true_self_play_optimizer import TrueSelfPlayOptimizer
from src.modules.rejection_feedback import RejectionFeedback
from src.modules.rejection_detector import RejectionDetector
from models.closedsource.gpt4_wrapper import GPT4Wrapper
from models.opensource.qwen.qwen_wrapper import QwenWrapper
from src.evaluation.paper_compliant_evaluation import RACESystemEvaluationAdapter
load_dotenv(override=True)
# --- 用於偵錯的臨時程式碼 ---
loaded_key = os.getenv("OPENAI_API_KEY")
if loaded_key:
    print(f"DEBUG: 已成功從 .env 載入 API 金鑰，開頭為: {loaded_key[:6]}...")
else:
    print("DEBUG: 錯誤！無法從 .env 檔案中讀取到 OPENAI_API_KEY。")
# --- 偵錯結束 ---
def create_model(model_key: str, config: dict):
    """
    (*** 核心修正點 ***)
    根據設定檔中的鍵名，創建對應的模型實例。
    這個版本更簡潔且不會錯誤地修改設定。
    """
    # 從 config 中獲取該模型的具體設定字典
    model_config = config['models'][model_key]
    # 從該字典中獲取模型類型，以決定使用哪個 Wrapper
    model_type = model_config.get('model_type', 'closedsource') # 默認為閉源

    logging.info(f"正在創建模型，鍵名: '{model_key}', 類型: '{model_type}'")

    if model_type == 'opensource':
        # 假設所有開源模型 Wrapper 都遵循 QwenWrapper 的模式
        return QwenWrapper(model_config)
    elif model_type == 'closedsource':
        # 假設所有閉源模型 Wrapper 都遵循 GPT4Wrapper 的模式
        return GPT4Wrapper(model_config)
    else:
        raise ValueError(f"未知的模型類型: {model_type}")

def main():
    load_dotenv() 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 1. 初始化階段 ---
    logging.info("實驗開始：初始化設定和資料...")
    config = load_config('/home/server/LiangYu/RACE/configs/config.yaml')

    results_dir = config['paths']['results_dir']
    summary_path = config['paths']['summary_path']
    os.makedirs(results_dir, exist_ok=True)
    
    logging.info("正在載入資料集...")
    try:
        dataset_path = config['dataset']['path']
        n_examples = config['dataset'].get('n_examples', None)
        df = pd.read_csv(dataset_path)
        if n_examples:
            df = df.head(n_examples)
        test_cases = df.to_dict('records')
        logging.info(f"成功載入並處理了 {len(test_cases)} 筆測試案例。")
    except Exception as e:
        logging.error(f"載入或處理資料集時發生錯誤: {e}")
        return

    # --- 2. 模型與模組初始化 ---
    logging.info("="*20 + " 初始化模型與核心模組 " + "="*20)

    # 根據 attacker, target, judge 的 name 欄位找到對應的設定鍵名
    attacker_key = config['models']['attacker']['name']
    target_key = config['models']['target']['name']
    judge_key = config['models']['judge']['name']
    
    attacker_model = create_model(attacker_key, config)
    target_model = create_model(target_key, config)
    
    logging.info("攻擊者和目標模型創建成功。")

    # ... (後續模組初始化不變) ...
    reasoning_constructor = ReasoningTaskConstructor(attacker_model)
    ig_calculator = TrueInformationGainCalculator(attacker_model, num_samples=config['self_play'].get('num_samples', 3))
    rejection_detector = RejectionDetector()
    gge_module = PaperAlignedGainGuidedExploration(
        llm_wrapper=attacker_model,
        information_gain_calculator=ig_calculator,
        rejection_detector=rejection_detector,
        num_candidates=config['gain_guided_exploration'].get('num_candidates', 5)
    )
    self_play_optimizer = TrueSelfPlayOptimizer(shadow_model=attacker_model, num_simulations=config['self_play'].get('num_simulations', 3))
    rejection_feedback = RejectionFeedback(attacker_model)
    logging.info("所有RACE攻擊組件初始化成功。")
    executor = PaperAlignedRACEExecutor(
        base_model=attacker_model,
        reasoning_constructor=reasoning_constructor,
        information_gain_calculator=ig_calculator,
        gge_module=gge_module,
        self_play_optimizer=self_play_optimizer,
        rejection_feedback=rejection_feedback
    )
    logging.info("論文對齊的RACE執行器初始化成功。")

    all_attack_results = []

    # --- 3. 攻擊執行階段 ---
    logging.info("="*20 + " 開始執行RACE攻擊循環 " + "="*20)
    for i, test_case in enumerate(tqdm(test_cases, desc="執行攻擊")):
        logging.info(f"--- 正在處理攻擊案例 {i+1}/{len(test_cases)} ---")
        logging.info(f"目標行為: {test_case['goal']}")
        
        attack_result = executor.execute_paper_aligned_attack(
            target_query=test_case['goal'],
            conversation_limit=config['attack_state_machine'].get('max_turns', 5),
            target_model=target_model,
            enable_defense=config['defense'].get('use_sentry', False) or config['defense'].get('use_guardian', False)
        )
        all_attack_results.append(attack_result)

        detail_path = os.path.join(results_dir, f"attack_case_{i+1:03d}.json")
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(attack_result, f, ensure_ascii=False, indent=4)

    logging.info("所有攻擊案例處理完畢。")

    # --- 4. 最終評估階段 ---
    logging.info("="*20 + " 開始執行最終的論文標準評估 " + "="*20)
    judge_model = create_model(judge_key, config)
    logging.info("評估模型創建成功。")

    adapter = RACESystemEvaluationAdapter(judge_model)
    final_evaluation = adapter.evaluate_race_attack_results(all_attack_results)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_evaluation, f, ensure_ascii=False, indent=4)
    logging.info(f"評估完成！最終評估報告已儲存至: {summary_path}")

    # --- 5. 實驗結束 ---
    logging.info("="*20 + " 實驗流程結束 " + "="*20)
    print("\n實驗摘要:")
    summary = final_evaluation.get('evaluation_summary', {})
    print(f"  總攻擊次數: {summary.get('total_attacks')}")
    print(f"  攻擊成功率 (ASR): {summary.get('asr_percentage', 0.0):.2f}%")
    print(f"  平均危害指數 (HRI): {summary.get('average_hri', 0.0):.2f}")
    print(f"\n詳細攻擊日誌位於: {results_dir}")
    print(f"總體評估報告位於: {summary_path}")

if __name__ == "__main__":
    main()
