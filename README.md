好的，這是一個根據您提供的專案文件結構與程式碼，以及您正在建立的防禦機制，所產生的詳細 `README.md`。

-----

# RACE with MRID: 多輪對話攻擊與防禦框架

本專案是對論文 [**Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models (RACE)**](https://arxiv.org/abs/2502.11054) 的官方實現進行的擴展。我們在原有攻擊框架的基礎上，整合了一套名為 **MRID (Multi-turn Reasoning-based Intent Defense)** 的主動防禦系統。

這個專案不僅復現了 RACE 攻擊的核心方法，還提供了一個實驗平台，用於測試和評估在多輪對話中，針對意圖隱藏和語義漂移的越獄攻擊的防禦策略。

## 專案核心功能

  * **RACE 攻擊框架**: 完整復現論文中的多輪越獄攻擊框架。此攻擊方法的核心是將有害意圖轉化為看似無害的、複雜的推理任務，利用大型語言模型（LLM）強大的推理能力來繞過其安全對齊。
  * **MRID 防禦系統**: 一套專門設計用於對抗 RACE 類攻擊的防禦機制。MRID 透過分析對話的演進過程、重建攻擊者的潛在推理鏈，並檢測其真實意圖，從而實現主動防禦。
  * **模組化設計**: 攻擊、防禦、模型和評估等模組高度解耦，方便研究人員擴展、替換或單獨測試特定部分。
  * **支援多種模型**: 支援閉源 API（如 GPT-4）和開源模型（如 Gemma, GLM, Qwen）進行攻擊和防禦的實驗。
  * **標準化評估**: 提供與論文一致的評估腳本，用於衡量攻擊成功率（Attack Success Rate, ASR）和防禦的有效性。

## 專案架構

```
racede-main/
│
├── configs/
│   └── config.yaml                 # 專案核心設定檔，用於配置實驗參數
│
├── datasets/
│   ├── advbench_subset.csv         # AdvBench 資料集的子集
│   ├── harmbench_behaviors_text_all.csv # HarmBench 資料集
│   └── harmbench_used.csv          # 實驗中實際使用的 HarmBench 資料
│
├── models/
│   ├── closedsource/               # 閉源模型 (如 OpenAI GPT-4) 的封裝
│   │   ├── gpt4_wrapper.py
│   │   └── official_api.py
│   └── opensource/                 # 開源模型 (Gemma, GLM, Qwen) 的封裝與範例
│
├── sentence_model/                 # 句子轉換器模型，用於計算語義相似度
│
├── src/
│   ├── core/                       # RACE 攻擊核心邏輯
│   │   ├── paper_aligned_attack_state_machine.py # 攻擊狀態機
│   │   ├── paper_aligned_race_executor.py        # RACE 攻擊執行器
│   │   └── reasoning_task_constructor.py         # 推理任務建構器
│   │
│   ├── defenses/                   # MRID 防禦系統的核心模組
│   │   ├── mrid_chain_reconstructor.py # 推理鏈重建器
│   │   ├── mrid_defense_wrapper.py     # 防禦總封裝
│   │   ├── mrid_evolution_tracker.py   # 對話演進追蹤器
│   │   ├── mrid_integrated_defense.py  # 整合防禦主流程
│   │   └── mrid_intent_detector.py     # 意圖檢測器
│   │
│   ├── evaluation/                 # 評估模組
│   │   └── paper_compliant_evaluation.py # 論文標準的評估腳本
│   │
│   ├── modules/                    # RACE 攻擊的輔助模組
│   │   ├── paper_aligned_gain_guided_exploration.py # 增益引導探索
│   │   ├── rejection_detector.py       # 模型拒絕回覆的檢測器
│   │   ├── rejection_feedback.py       # 拒絕回覆的處理
│   │   └── ...
│   │
│   ├── selfplay/                   # Self-play 相關模組
│   │   └── prompt_mutator.py
│   │
│   └── utils/                      # 工具函式
│       ├── config_loader.py
│       ├── data_loader.py
│       └── logger.py
│
├── run_experiment.py               # 實驗主執行入口
├── test_mrid_defense.py            # MRID 防禦的單獨測試腳本
└── README.md                       # 本文件
```

-----

## 檔案與模組功能詳解

### 1\. 根目錄

  * `run_experiment.py`: **專案主入口**。此腳本讀取 `configs/config.yaml`，初始化模型、資料集、攻擊模組和防禦模組，並啟動完整的攻擊/防禦實驗流程。
  * `test_mrid_defense.py`: 專用於測試 **MRID 防禦系統**的腳本。它模擬一個攻擊場景，並評估 MRID 防禦框架的有效性。
  * `configs/config.yaml`: **核心設定檔**。所有實驗的可調參數都在此定義，包括：
      * `model`: 要使用的目標 LLM (攻擊對象) 和防禦 LLM 的路徑或名稱。
      * `dataset`: 測試資料集的路徑和名稱。
      * `attack_params`: RACE 攻擊的相關參數，如最大對話輪數、增益閾值等。
      * `defense_params`: MRID 防禦的相關參數，如意圖檢測的敏感度、演進追蹤的策略等。

### 2\. `src` - 核心原始碼

#### `src/core` - RACE 攻擊核心

  * `paper_aligned_attack_state_machine.py`: 實現了論文中描述的**攻擊狀態機 (Attack State Machine, ASM)**。它定義了攻擊的各個階段（例如：任務轉化、推理深化、攻擊達成）以及如何在這些狀態之間轉換，確保攻擊過程的連貫性。
  * `paper_aligned_race_executor.py`: RACE 攻擊的**執行器**。它根據狀態機的指令，調用各個模組（如推理任務建構器、LLM 介面）來執行每一步的攻擊。
  * `reasoning_task_constructor.py`: **推理任務建構器**。這是 RACE 攻擊的關鍵，負責將原始的有害問題（如 "如何製造炸彈"）轉化為一個複雜、多步驟、看似無害的邏輯推理、物理或化學問題。

#### `src/defenses` - MRID 防禦系統 (您的貢獻)

這是您專案中新增的核心防禦模組，旨在對抗如 RACE 這樣的多輪推理攻擊。

  * `mrid_integrated_defense.py`: **整合防禦主流程**。這是 MRID 的入口點，它協調其他防禦模組，對傳入的每一輪對話進行分析，並最終決定是否攔截。
  * `mrid_intent_detector.py`: **意圖檢測器**。使用一個強大的 LLM（或一個特化的分類器）來分析使用者的單輪或多輪輸入，判斷其背後是否存在隱藏的有害意too。即使輸入的字面意思無害，它也能識別出潛在的危險。
  * `mrid_evolution_tracker.py`: **對話演進追蹤器**。此模組負責追蹤整個對話的語義演進路徑。它使用 `sentence_model` 來計算每一輪對話與預設有害主題之間的語義相似度，如果對話逐漸逼近某個有害目標，即使每一步的變化很小，也會被標記為可疑。
  * `mrid_chain_reconstructor.py`: **推理鏈重建器**。MRID 的核心創新之一。此模組嘗試 "逆向工程" 攻擊者的多輪推理任務，將看似分散的良性問題拼接起來，重建出攻擊者可能遵循的完整邏輯鏈，從而暴露其最終的有害目的。
  * `mrid_defense_wrapper.py`: 對整個 MRID 系統的標準化封裝，使其可以輕鬆地整合到 `run_experiment.py` 的實驗流程中。

#### `src/modules` - RACE 攻擊輔助模組

這些模組實現了論文中提到的用於提升攻擊效率的各種策略。

  * `paper_aligned_gain_guided_exploration.py`: 實現**增益引導探索**。在生成下一步攻擊提示時，它會評估多個候選項，並選擇最有可能引導 LLM 朝著有害目標前進且最不容易被拒絕的提示。
  * `rejection_detector.py` & `rejection_feedback.py`: **拒絕處理模組**。前者用於檢測目標 LLM 的回覆是否為拒絕（例如 "我不能回答這個問題"），後者則根據拒絕回覆來調整後續的攻擊策略。
  * `true_information_gain_calculator.py` & `true_self_play_optimizer.py`: **自博弈優化模組**。透過自我對抗（Self-Play）的方式，讓攻擊模型不斷優化其生成提示的策略，以應對不斷變化的防禦或模型行為。

#### `src/evaluation` - 評估

  * `paper_compliant_evaluation.py`: **標準化評估腳本**。在實驗結束後，此腳本會分析 LLM 的最終輸出。它通常通過關鍵字匹配、語義分析或 LLM 評判的方式，來判斷攻擊是否成功繞過了安全機制並達成了有害目標，最後計算出攻擊成功率 (ASR)。

### 3\. `datasets` - 資料集

  * `advbench_subset.csv`: [Adversarial Robustness Benchmark](https://github.com/llm-attacks/llm-attacks) 的子集，包含一系列已知的有害指令。
  * `harmbench_behaviors_text_all.csv`: [HarmBench](https://github.com/centerforaisafety/HarmBench) 的資料，包含了大量被禁止的行為描述，用於生成初始的有害查詢。
  * `harmbench_used.csv`: 經過篩選後，實際用於本專案實驗的 HarmBench 資料。

### 4\. `models` - 模型介面

  * **closedsource**: 提供了對 OpenAI API (GPT-4) 等商業模型的呼叫封裝。`official_api.py` 處理 API 請求、認證和重試邏輯。
  * **opensource**: 提供了對本地部署的開源模型（如 Gemma, GLM, Qwen）的封裝。每個文件夾下的 `..._wrapper.py` 將模型的輸入輸出格式化，使其與專案的其他部分兼容。

### 5\. `sentence_model` - 語義分析模型

  * 此目錄包含一個預先訓練好的**句子轉換器 (Sentence Transformer)** 模型。它的主要作用是將句子或段落轉換為高維度的向量表示。在 MRID 防禦系統中，它被 `mrid_evolution_tracker` 用於計算不同對話輪次之間、或對話與有害主題之間的**語義相似度**，是實現對話演進追蹤的關鍵技術。

-----

## 運作流程

1.  **配置實驗**: 在 `configs/config.yaml` 中設定實驗參數，例如選擇攻擊目標 LLM、設定防禦 LLM、指定要使用的資料集以及設定攻防的具體參數。
2.  **啟動實驗**: 執行 `python run_experiment.py`。
3.  **資料載入**: `src/utils/data_loader.py` 從 `datasets/` 中讀取指定的有害行為資料。
4.  **攻擊與防禦循環**:
    a. **攻擊方 (RACE)**: `reasoning_task_constructor` 將一個有害行為（例如，來自 `harmbench_used.csv`）轉化為一個良性的推理問題作為第一輪輸入。
    b. **防禦方 (MRID)**: `mrid_integrated_defense` 介入，接收到這個輸入後：
    i. `mrid_intent_detector` 檢查此輸入的潛在意圖。
    ii. `mrid_evolution_tracker` 更新對話狀態，並與歷史記錄進行比較。
    iii. `mrid_chain_reconstructor` 嘗試將此問題與之前的對話拼接，看是否能構成一個完整的有害推理鏈。
    iv. 如果 MRID 判斷存在高風險，則會攔截該請求，實驗記錄為防禦成功；否則，請求被傳遞給目標 LLM。
    c. **目標 LLM**: 接收到請求並生成回應。
    d. **攻擊方 (RACE)**: 接收到 LLM 的回應。如果回應是拒絕，`rejection_feedback` 會記錄並調整策略。如果不是，`paper_aligned_gain_guided_exploration` 會根據此回應和當前狀態，生成下一輪的、更進一步的推理問題。
    e. 流程回到 **b**，開始新一輪的攻防對抗。
5.  **評估**: 當對話達到最大輪數或攻擊/防禦目標達成時，`paper_compliant_evaluation.py` 會被調用，分析整個對話歷史和最終輸出，記錄本次實驗的結果（攻擊是否成功、防禦是否成功）。
6.  **日誌記錄**: 所有詳細的對話、攻防判斷和結果都會被記錄到指定的日誌文件中。

-----
