import os
import json
import logging
import time
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import yaml

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

def create_timestamped_experiment_session():
    """創建帶時間戳記的實驗會話"""
    
    # 生成時間戳記
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建實驗目錄結構
    base_dir = "experiments"
    session_dir = f"{base_dir}/exp_{timestamp}"
    
    # 創建目錄結構
    dirs_to_create = [
        session_dir,
        f"{session_dir}/attack_details",
        f"{session_dir}/logs", 
        f"{session_dir}/configs",
        f"{session_dir}/results"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    return timestamp, session_dir

def setup_experiment_logging(session_dir: str, timestamp: str):
    """設置實驗專用的日誌記錄"""
    
    # 清除之前的日誌配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 創建日誌文件
    log_file = f"{session_dir}/logs/experiment_{timestamp}.log"
    
    # 配置日誌記錄器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同時輸出到控制台
        ]
    )
    
    return log_file

def save_experiment_metadata(session_dir: str, config: dict, dataset_info: dict):
    """保存實驗元數據"""
    
    metadata = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "session_id": os.path.basename(session_dir),
            "python_version": "3.8+",
            "race_version": "paper_aligned_v1.0"
        },
        "dataset_info": dataset_info,
        "model_config": {
            "attacker": config['models']['attacker']['name'],
            "target": config['models']['target']['name'], 
            "judge": config['models']['judge']['name']
        },
        "attack_config": {
            "max_turns": config['attack_state_machine']['max_turns'],
            "num_candidates": config['gain_guided_exploration']['num_candidates'],
            "num_simulations": config['self_play']['num_simulations']
        }
    }
    
    # 保存元數據
    metadata_file = f"{session_dir}/experiment_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    # 備份配置文件
    config_backup = f"{session_dir}/configs/config_backup.yaml"
    with open(config_backup, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return metadata_file

def generate_experiment_summary(session_dir: str, all_attack_results: list, 
                               final_evaluation: dict, execution_time: float):
    """生成實驗總結報告"""
    
    # 統計信息
    total_attacks = len(all_attack_results)
    successful_attacks = sum(1 for r in all_attack_results 
                           if r.get('attack_outcome', {}).get('attack_successful', False))
    
    # 計算平均輪數
    total_turns = sum(r.get('execution_metadata', {}).get('actual_turns', 0) 
                     for r in all_attack_results)
    avg_turns = total_turns / total_attacks if total_attacks > 0 else 0
    
    # 模組使用統計
    module_stats = {
        'gge_usage': sum(r.get('module_usage_statistics', {}).get('gge_usage', 0) 
                        for r in all_attack_results),
        'self_play_usage': sum(r.get('module_usage_statistics', {}).get('self_play_usage', 0) 
                              for r in all_attack_results),
        'rejection_feedback_usage': sum(r.get('module_usage_statistics', {}).get('rejection_feedback_usage', 0) 
                                       for r in all_attack_results)
    }
    
    # 生成總結
    summary = {
        "experiment_summary": {
            "session_id": os.path.basename(session_dir),
            "completion_time": datetime.now().isoformat(),
            "total_execution_time_minutes": round(execution_time / 60, 2)
        },
        "attack_statistics": {
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "success_rate": round(successful_attacks / total_attacks * 100, 2) if total_attacks > 0 else 0,
            "average_turns_per_attack": round(avg_turns, 2)
        },
        "module_usage_statistics": module_stats,
        "llm_judge_evaluation": final_evaluation.get('evaluation_summary', {}),
        "paper_alignment": {
            "avg_compliance_score": round(
                sum(r.get('paper_alignment_analysis', {}).get('paper_compliance_score', 0) 
                    for r in all_attack_results) / total_attacks, 3
            ) if total_attacks > 0 else 0
        }
    }
    
    # 保存總結
    summary_file = f"{session_dir}/EXPERIMENT_SUMMARY.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    return summary_file

def create_model(model_key: str, config: dict):
    """
    根據設定檔中的鍵名，創建對應的模型實例。
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
    """修改後的主函數，支持時間戳記實驗日誌"""
    
    load_dotenv(override=True)
    
    # --- 用於偵錯的臨時程式碼 ---
    loaded_key = os.getenv("OPENAI_API_KEY")
    if loaded_key:
        print(f"DEBUG: 已成功從 .env 載入 API 金鑰，開頭為: {loaded_key[:6]}...")
    else:
        print("DEBUG: 錯誤！無法從 .env 檔案中讀取到 OPENAI_API_KEY。")
    # --- 偵錯結束 ---
    
    # === 1. 創建時間戳記實驗會話 ===
    timestamp, session_dir = create_timestamped_experiment_session()
    log_file = setup_experiment_logging(session_dir, timestamp)
    
    print(f"🚀 開始新的實驗會話：{timestamp}")
    print(f"📁 實驗目錄：{session_dir}")
    print(f"📝 日誌文件：{log_file}")
    print("-" * 60)
    
    logging.info(f"實驗會話開始：{timestamp}")
    logging.info(f"實驗目錄：{session_dir}")
    
    # === 2. 載入配置和數據 ===
    config = load_config('/home/server/LiangYu/RACE/configs/config.yaml')
    
    # 更新輸出路徑到會話目錄
    results_dir = f"{session_dir}/attack_details"
    summary_path = f"{session_dir}/results/evaluation_summary.json"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{session_dir}/results", exist_ok=True)
    
    logging.info("正在載入資料集...")
    try:
        dataset_path = config['dataset']['path']
        n_examples = config['dataset'].get('n_examples', None)
        
        df = pd.read_csv(dataset_path)
        logging.info(f"數據集載入成功，形狀: {df.shape}")
        logging.info(f"數據集列名: {list(df.columns)}")
        
        # 自動處理列名問題
        if 'goal' not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'goal'})
            logging.info(f"已將列 '{first_col}' 重命名為 'goal'")
        
        if n_examples and n_examples > 0:
            df = df.head(n_examples)
        
        test_cases = df.to_dict('records')
        
        # 記錄數據集信息
        dataset_info = {
            "dataset_path": dataset_path,
            "total_samples": len(test_cases),
            "dataset_columns": list(df.columns),
            "sample_goal": test_cases[0].get('goal', 'N/A')[:100] if test_cases else 'N/A'
        }
        
        logging.info(f"成功載入 {len(test_cases)} 筆測試案例")
        
    except Exception as e:
        logging.error(f"載入資料集失敗: {e}")
        return
    
    # === 3. 保存實驗元數據 ===
    metadata_file = save_experiment_metadata(session_dir, config, dataset_info)
    logging.info(f"實驗元數據已保存：{metadata_file}")
    
    # === 4. 初始化模型與模組 ===
    start_time = time.time()
    
    logging.info("="*20 + " 初始化模型與核心模組 " + "="*20)

    # 根據 attacker, target, judge 的 name 欄位找到對應的設定鍵名
    attacker_key = config['models']['attacker']['name']
    target_key = config['models']['target']['name']
    judge_key = config['models']['judge']['name']
    
    attacker_model = create_model(attacker_key, config)
    target_model = create_model(target_key, config)
    
    logging.info("攻擊者和目標模型創建成功。")

    # 初始化所有RACE組件
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

    # === 5. 攻擊執行階段 ===
    logging.info("="*20 + " 開始執行RACE攻擊循環 " + "="*20)
    for i, test_case in enumerate(tqdm(test_cases, desc="執行攻擊")):
        logging.info(f"--- 正在處理攻擊案例 {i+1}/{len(test_cases)} ---")
        goal_text = test_case.get('goal', 'Unknown')
        logging.info(f"目標行為: {goal_text}")
        
        attack_result = executor.execute_paper_aligned_attack(
            target_query=goal_text,
            conversation_limit=config['attack_state_machine'].get('max_turns', 5),
            target_model=target_model,
            enable_defense=config['defense'].get('use_sentry', False) or config['defense'].get('use_guardian', False)
        )
        all_attack_results.append(attack_result)

        detail_path = os.path.join(results_dir, f"attack_case_{i+1:03d}.json")
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(attack_result, f, ensure_ascii=False, indent=4)

    logging.info("所有攻擊案例處理完畢。")

    # === 6. 最終評估階段 ===
    logging.info("="*20 + " 開始執行最終的論文標準評估 " + "="*20)
    judge_model = create_model(judge_key, config)
    logging.info("評估模型創建成功。")

    adapter = RACESystemEvaluationAdapter(judge_model)
    final_evaluation = adapter.evaluate_race_attack_results(all_attack_results)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_evaluation, f, ensure_ascii=False, indent=4)
    logging.info(f"評估完成！最終評估報告已儲存至: {summary_path}")

    # === 7. 生成實驗總結 ===
    execution_time = time.time() - start_time
    
    summary_file = generate_experiment_summary(
        session_dir, all_attack_results, final_evaluation, execution_time
    )
    
    # === 8. 實驗完成報告 ===
    logging.info("=" * 60)
    logging.info("🎉 實驗完成！")
    logging.info(f"📊 實驗總結：{summary_file}")
    logging.info(f"⏱️  總執行時間：{execution_time/60:.2f} 分鐘")
    logging.info(f"📁 所有結果保存在：{session_dir}")
    
    print("\n" + "=" * 60)
    print("🎉 實驗完成！")
    print(f"📁 實驗結果保存在：{session_dir}")
    print(f"📊 查看總結：{summary_file}")
    print(f"📝 查看日誌：{log_file}")
    print("=" * 60)

    # === 9. 實驗摘要顯示 ===
    logging.info("="*20 + " 實驗流程結束 " + "="*20)
    print("\n實驗摘要:")
    summary = final_evaluation.get('evaluation_summary', {})
    print(f"  總攻擊次數: {summary.get('total_attacks')}")
    print(f"  攻擊成功率 (ASR): {summary.get('asr_percentage', 0.0):.2f}%")
    print(f"  平均危害指數 (HRI): {summary.get('average_hri', 0.0):.2f}")
    print(f"\n詳細攻擊日誌位於: {results_dir}")
    print(f"總體評估報告位於: {summary_path}")

def list_experiment_history():
    """列出歷史實驗記錄"""
    
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        print("還沒有實驗記錄")
        return
    
    experiments = [d for d in os.listdir(experiments_dir) 
                  if d.startswith('exp_') and os.path.isdir(f"{experiments_dir}/{d}")]
    
    experiments.sort(reverse=True)  # 最新的在前
    
    print("📚 實驗歷史記錄：")
    print("-" * 80)
    
    for exp in experiments[:10]:  # 顯示最近10個實驗
        exp_dir = f"{experiments_dir}/{exp}"
        summary_file = f"{exp_dir}/EXPERIMENT_SUMMARY.json"
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            stats = summary.get('attack_statistics', {})
            print(f"🔬 {exp}")
            print(f"   ├─ 總攻擊：{stats.get('total_attacks', 'N/A')}")
            print(f"   ├─ 成功率：{stats.get('success_rate', 'N/A')}%")
            print(f"   ├─ 平均輪數：{stats.get('average_turns_per_attack', 'N/A')}")
            print(f"   └─ 目錄：{exp_dir}")
        else:
            print(f"🔬 {exp} (進行中或異常結束)")
        
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "history":
        list_experiment_history()
    else:
        main()
