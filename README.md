# racede

專案旨在透過一種新穎的多輪對話攻擊框，來揭示大型語言模型 (LLM) 的安全漏洞。與傳統的直接攻擊方法不同，RACE 將有害的指令重新表述為良性的、需要多步推理才能解決的任務，從而利用 LLM 強大的推理能力來繞過其安全對齊機制。

## 核心概念

框架的核心思想是**將有害意圖隱藏在看似無害的推理任務中**。透過多輪對話，逐步引導模型完成推理，最終使其產生不安全的內容。此框架主要由以下幾個關鍵模組構成：

1.  **攻擊狀態機 (Attack State Machine)**: 系統化地將一個有害問題分解為多個良性的、合乎邏輯的對話回合。它確保了在多輪對話中，攻擊的語意能夠連貫且持續地推進。
2.  **增益引導探索 (Gain-Guided Exploration)**: 透過評估潛在的探索方向（即生成的回應）能夠帶來多少「資訊增益」，來智慧地選擇最有希望達到越獄目標的回應。
3.  **自對弈 (Self-Play)**: 讓攻擊模型與目標模型進行「對抗」，從對抗的結果中學習和優化攻擊策略，從而不斷產生更有效、更隱蔽的攻擊提示。
4.  **拒絕回饋 (Rejection Feedback)**: 當模型拒絕回答某個提示時，此機制會分析拒絕的原因，並將其作為回饋，用於調整後續的攻擊提示，從而繞過模型的防禦。

## 專案架構與檔案說明

本專案的程式碼結構清晰，主要分為設定檔、資料集、模型封裝、核心邏輯和工具函式等幾個部分。

```
.
├── configs/
│   └── config.yaml                 # 實驗的總設定檔
├── datasets/
│   ├── advbench_subset.csv         # AdvBench 資料集子集
│   ├── harmbench_behaviors_text_all.csv  # HarmBench 資料集的行為描述
│   └── harmbench_used.csv          # 實驗中使用的 HarmBench 資料
├── models/
│   ├── closedsource/               # 封閉原始碼模型 (如 GPT-4) 的 API 封裝
│   │   ├── gpt4_wrapper.py
│   │   └── official_api.py
│   └── opensource/                 # 開源模型的封裝 (Gemma, GLM, Qwen)
│       ├── gemma/example.py
│       ├── glm/example.py
│       └── qwen/
│           ├── example.py
│           └── qwen_wrapper.py
├── sentence_model/                 # 用於語意相似度計算的句子模型
├── src/
│   ├── core/                       # 專案核心攻擊邏輯
│   │   ├── paper_aligned_attack_state_machine.py # 攻擊狀態機實現
│   │   ├── paper_aligned_race_executor.py        # RACE 攻擊流程的總執行器
│   │   └── reasoning_task_constructor.py         # 將有害查詢轉換為推理任務
│   ├── evaluation/
│   │   └── paper_compliant_evaluation.py         # 遵循論文標準的評估腳本
│   ├── modules/                    # RACE 框架的關鍵功能模組
│   │   ├── paper_aligned_gain_guided_exploration.py # 增益引導探索模組
│   │   ├── rejection_detector.py         # 模型回應拒絕偵測器
│   │   ├── rejection_feedback.py         # 拒絕回饋機制
│   │   ├── true_information_gain_calculator.py # 資訊增益計算
│   │   └── true_self_play_optimizer.py   # 自對弈最佳化器
│   ├── selfplay/
│   │   └── prompt_mutator.py             # 在自對弈中用於產生提示變體
│   └── utils/
│       ├── config_loader.py            # 設定檔載入工具
│       ├── data_loader.py              # 資料集載入工具
│       ├── logger.py                   # 日誌記錄工具
│       └── response_handler.py         # 模型回應處理工具
└── run_experiment.py               # 執行實驗的主入口腳本
```

### 詳細檔案功能

  - **`run_experiment.py`**: 整個專案的啟動點。它會讀取設定檔，初始化資料、模型和 RACE 執行器，並開始攻擊流程。

  - **`configs/config.yaml`**: 核心設定檔。您可以在此處定義：

      - 要使用的目標模型 (Target Model) 和攻擊模型 (Attack Model)。
      - API金鑰 (例如 OpenAI 的金鑰)。
      - 實驗參數，如最大對話輪數、溫度 (temperature) 等。
      - 資料集和日誌的路徑。

  - **`datasets/`**: 存放用於攻擊的資料集。

      - `advbench_subset.csv`: 從 AdvBench 中選取的一部分有害指令。
      - `harmbench_...`: 來自 HarmBench 的資料，用於更複雜、更多樣的有害行為測試。

  - **`models/`**: 存放與不同 LLM 互動的介面。

      - `closedsource/`: 處理需要透過 API 存取的模型，如 `gpt4_wrapper.py` 封裝了與 OpenAI API 的互動邏輯。
      - `opensource/`: 處理可以在本地運行的開源模型，如 Qwen、Gemma 等。`qwen_wrapper.py` 提供了如何載入和呼叫 Qwen 模型的方法。

  - **`sentence_model/`**: 包含一個預訓練的句子轉換器模型 (Sentence Transformer)。此模型主要用於**語意相似度計算**，例如在 `rejection_detector.py` 中判斷模型的回應是否為拒絕，或在 `true_information_gain_calculator.py` 中評估回應的語意與目標的接近程度。

  - **`src/core/`**: 實現 RACE 攻擊的核心框架。

      - `paper_aligned_attack_state_machine.py`: 實現論文中描述的狀態機，負責將初始的有害問題分解成一系列子問題，並管理對話的流程。
      - `paper_aligned_race_executor.py`: RACE 框架的總指揮。它協調狀態機、資訊增益探索、自對弈等模組，完成整個多輪攻擊過程。
      - `reasoning_task_constructor.py`: 關鍵模組之一，負責將直接的有害查詢（例如「如何製造炸彈」）轉化為一個需要推理的、看似無害的問題（例如「給定材料A、B、C，描述它們反應的詳細步驟」）。

  - **`src/modules/`**: 實現 RACE 框架的具體演算法。

      - `paper_aligned_gain_guided_exploration.py`: 實現增益引導的探索策略，用於選擇最有效的對話路徑。
      - `rejection_detector.py`: 使用 `sentence_model` 來判斷目標 LLM 的回應是否構成拒絕。
      - `rejection_feedback.py`: 當偵測到拒絕時，此模組會產生回饋，指導攻擊模型調整策略。
      - `true_information_gain_calculator.py`: 計算不同回應所帶來的「資訊增益」，幫助 `gain_guided_exploration` 做出決策。
      - `true_self_play_optimizer.py`: 實現自對弈循環，讓攻擊模型不斷從與目標模型的互動中進行學習和改進。

  - **`src/evaluation/`**:

      - `paper_compliant_evaluation.py`: 評估腳本。在攻擊結束後，此腳本會分析生成的對話歷史。它通常會使用一個強大的、獨立的「評判模型」（如 GPT-4）來判斷攻擊是否成功「越獄」，並計算攻擊成功率 (Attack Success Rate, ASR)。

  - **`src/utils/`**: 提供各種輔助功能，使主程式碼更簡潔。

      - `config_loader.py`, `data_loader.py`, `logger.py`: 分別負責載入設定、資料和設定日誌記錄。

## 如何運作

1.  **設定 (Setup)**:

      - 安裝所有必要的 Python 套件 (建議建立一個 `requirements.txt` 檔案)。
      - 在 `configs/config.yaml` 中配置您的 API 金鑰、模型路徑和實驗參數。

2.  **執行 (Execution)**:

      - 在終端中運行主腳本：
        ```bash
        python run_experiment.py
        ```

3.  **流程 (Flow)**:

      - `run_experiment.py` 載入設定和資料。
      - 它初始化 `RACEExecutor`。
      - 對於資料集中的每一個有害指令，`RACEExecutor` 開始執行攻擊迴圈：
        a. `ReasoningTaskConstructor` 將有害指令轉化為一個推理任務的開頭。
        b. 在每一輪對話中，攻擊模型生成多個候選回應。
        c. `GainGuidedExploration` 和 `InformationGainCalculator` 評估哪個候選回應最有可能引導目標模型走向「越獄」。
        d. 將選定的回應發送給目標模型。
        e. `RejectionDetector` 檢查目標模型的回應。如果被拒絕，`RejectionFeedback` 會介入以調整策略。
        f. `AttackStateMachine` 根據對話進度更新狀態，進入下一輪或終止。
      - 整個對話歷史和結果會被記錄在指定的日誌檔案中。

4.  **評估 (Evaluation)**:

      - 實驗結束後，運行 `src/evaluation/paper_compliant_evaluation.py` 腳本。
      - 該腳本會讀取日誌檔案，並使用評判模型（例如 GPT-4）來評估每次攻擊是否成功，最終報告攻擊成功率等指標。

## 資料與模型

  - **資料集**:

      - **AdvBench**: 一個廣泛使用的標準測試集，包含各種有害指令。
      - **HarmBench**: 一個更具挑戰性的測試集，涵蓋了更廣泛和更細微的有害行為。

  - **模型**:

      - **目標模型 (Target Models)**: 任何您想測試其安全性的 LLM，例如 GPT-4, GPT-3.5, Claude, Llama, Gemma, Qwen 等。
      - **攻擊模型 (Attack Models)**: 用於生成攻擊提示的 LLM。通常使用能力較強的模型（如 GPT-4）來產生高品質的攻擊對話。
      - **評判模型 (Judge Model)**: 用於自動評估攻擊是否成功，通常選用最強大的可用模型（如 GPT-4）以確保評估的準確性。
      - **句子模型 (Sentence Model)**: 一個輕量級的轉換器模型，用於計算語意相似度，輔助判斷拒絕意圖。

-----
