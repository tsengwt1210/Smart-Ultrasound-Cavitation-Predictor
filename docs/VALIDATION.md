# 驗證紀錄

日期：2026-10-14。這份紀錄說明本次公開版本整理後實際做過的檢查，與歷史訓練結果分開。

## 執行環境

- Windows，MATLAB R2024a Update 3（24.1.0.2603908）。
- Deep Learning Toolbox 24.1。
- 使用本庫既有 `test.xlsx` 與三份 BPNN 模型，沒有重新訓練。

MATLAB R2025b 的另一套本機安裝沒有 Deep Learning Toolbox，因此模型推論改在已具備該工具箱的 R2024a 完成。MATLAB 版本號本身不能代表工具箱已安裝。

## BPNN 推論介面

執行 `runtests('tests/test_predict_area.m')`：**7 項通過、0 項失敗、0 項未完成**。

| 測試 | 驗證內容 |
| --- | --- |
| `testSavedModelInference` | 三種能量各五組特徵，共 15 組輸入；新介面與原 `net(x) * T_max` 一致，容許絕對差 10⁻¹² |
| `testIndependentOfCurrentFolder` | 切換工作目錄後，仍能從函數所在位置找到模型 |
| `testRejectsUnknownEnergy` | 不支援的能量字串明確報錯 |
| `testRejectsNonfiniteInput` | 拒絕 NaN 與 Inf |
| `testRejectsVectorInput` | 拒絕非純量特徵 |
| `testMissingModel` | 缺少模型時回報對應檔案錯誤 |
| `testDemo` | 範例回傳三列輸入與有限的預測面積 |

`demo_bpnn` 的實際輸出（顯示至約五位小數）：

| 能量 | F1_mag | F2_mag | 預測面積 mm² |
| --- | ---: | ---: | ---: |
| 0.8J | 23931 | 4013 | 0.46693 |
| 1.0J | 34082 | 6162 | 0.65892 |
| 1.2J | 44238 | 8156 | 0.79400 |

這些是既有資料上的推論展示，不能當成新測試集結果或模型準確率。訓練、驗證和測試的解讀見[結果與限制](RESULTS_AND_LIMITATIONS.md)。

## 靜態與檔案檢查

- `predict_area.m`、`demo_bpnn.m` 及測試檔通過 MATLAB Code Analyzer 檢查。
- `rcnn_predict.py` 通過 Python AST 語法解析；檢查過模型建構參數與使用提示。
- 核對三份模型的層數、輸入維度、`T_max`、`mapminmax` 與資料切分設定。
- 檢查 README／文件的本機相對連結及 Git 差異。

## 本次未執行

- U-Net 訓練、五折交叉驗證與影像推論。
- Mask R-CNN 完整推論、Python 套件重裝或 GPU 測試；本庫沒有其權重。
- `.mlapp` 各版本的完整操作、USB 攝影機與 Arduino／LCD 驗收。
- 任何臨床或跨設備評估。
