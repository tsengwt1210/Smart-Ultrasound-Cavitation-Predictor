# 智慧超音波空穴預測系統 (Smart Ultrasound Cavitation Predictor / BioUS-Net)

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

## 專案簡介 (Introduction)
智慧超音波空穴預測系統是一個結合深度學習與醫學超音波技術的創新系統。本專案旨在透過 AI 模型，精準預測超音波在仿生皮膚模擬片上所產生的空穴效應（Cavitation Effect）。透過本系統，能夠有效評估與量化超音波的能量分佈與作用情形，為未來的超音波相關應用提供可靠的預測指標。

## 核心功能與技術 (Features & Technologies)
本系統採用 **MATLAB App Designer** 開發圖形化介面，並具備高度整合的軟硬體架構：
* **多模態 AI 預測模型**:
  * **U-Net**: 用於精準的像素級影像分割，提取超音波影像中的關鍵特徵。
  * **R-CNN**: 負責物件偵測，透過跨語言呼叫 Python 後端腳本進行定位與實例分割。
  * **BPNN (倒傳遞神經網路)**: 用於超音波聲學參數與空穴面積的快速回歸預測。
* **即時影像與 DSP 處理**: 支援 USB 攝影機即時串流與截圖，並結合數位訊號處理技術進行特徵萃取。
* **硬體通訊整合**: 具備 Serial Port 通訊功能，可將分析結果與信心度即時傳送至 Arduino 並顯示於 LCD 螢幕上。

## 獲獎與學術成就 (Achievements & Publications)
本專案在創新性與技術實作上獲得多項肯定：
* **InnoServe 大專校院資訊應用服務創新競賽** - 第三名
* **全國創新競賽** - 獲獎
* **全國 AI 競賽** - 決賽入圍
* **IEEE ICCE-TW 2026** - 論文投稿
* **國科會大專學生研究計畫 (NSTC)** - 計畫申請與執行

## 系統介面與詳細操作步驟 (System Interface & Step-by-Step Guide)

本系統介面設計為五個獨立的功能分頁（Tabs），引導使用者從基礎影像擷取到進階 AI 預測：
<img width="310" height="244" alt="image" src="https://github.com/user-attachments/assets/c5618f75-8df7-4453-ad0a-87db51e6dd56" />

### Tab 1: 影像載入與手動測量 (Image Loading & Manual Measurement)
1. **即時拍攝/導入圖片**：點擊「開啟相機」進行即時影像串流，或點選「載入圖片」匯入測試影像。
2. **比例尺校準 (Scale Calibration)**：點擊「設定比例」，在畫面上畫出 1mm 基準線，系統將自動轉換像素與實體面積 (mm²)。
3. **手動框選**：按 `m` 鍵啟動測量模式，以多邊形描繪空穴反應邊緣，雙擊結束。系統將自動計算面積並可同步回傳至外接 LCD 螢幕。
<img width="299" height="236" alt="image" src="https://github.com/user-attachments/assets/fe781dca-4be2-4161-a6c3-1b6ca19478e8" />

### Tab 2: U-Net 影像分割 (U-Net Image Segmentation)
1. **載入模型**：匯入訓練好的 U-Net `.mat` 模型權重。
2. **執行分割**：點擊分析，模型將進行像素級空穴影像分割，並依據信心分數 (Confidence Score) 排除背景干擾。
3. **檢視結果**：系統提供「遮罩圖」、「信心分數熱圖」、「亮化/暗化效果圖」等多種視覺化模式供切換檢視。
<img width="356" height="250" alt="image" src="https://github.com/user-attachments/assets/8d5a41bb-3705-41e7-b5e5-786165b82627" />

### Tab 3: R-CNN 實例分割 (R-CNN Instance Segmentation)
1. **執行偵測**：匯入圖片後點擊分析，系統會在背景自動呼叫 `rcnn_predict.py` 腳本。
2. **檢視邊界框**：系統會回傳包含邊界框 (Bounding Boxes)、面積與信心分數的結果，標註出獨立的空穴反應區塊。
<img width="356" height="249" alt="image" src="https://github.com/user-attachments/assets/3d372ccf-65e5-4864-ba68-0e0641b6b203" />

### Tab 4: 神經網路訓練 (Neural Network Training)
此分頁提供 BPNN 模型的重新訓練介面：
1. **導入資料**：匯入包含超音波參數的 Excel 數據集。
2. **模型訓練**：點擊「訓練」按鈕，系統會自動切分資料集 (Train/Val/Test = 70/15/15) 並開始迭代訓練。
3. **儲存/載入模型**：訓練完成後可檢視 Performance 圖表，並將權重匯出為 `.mat` 檔。

### Tab 5: 神經網路面積預測 (BPNN Area Prediction)
此為數值快速預測模式，無需完整影像即可預估結果：
1. **輸入參數**：選擇能量等級 (0.8J/1.0J/1.2J)，並輸入對應的 F1_mag 與 F2_mag 數值。
2. **執行預測**：系統載入對應能量參數的 BPNN 模型，極速輸出預期面積，並同步發送至硬體 LCD 顯示。

## 環境需求與執行 (Prerequisites & Usage)

### 環境需求
請確保您的開發環境已安裝以下工具與套件：
* **MATLAB (建議 R2022a 或更新版本)**，並包含以下 Toolboxes：
  * Deep Learning Toolbox
  * Image Processing Toolbox
  * MATLAB Support Package for USB Webcams (用於 Tab 1 相機功能)
* **Python 3.x** (用於 Tab 3 R-CNN 後端)：
  * 需安裝 `torch` (PyTorch) 與 `torchvision`。
* **硬體 (選配)**：
  * USB 攝影機
  * Arduino + LCD 螢幕 (預設通訊埠配置為 `COM7`, 9600 baud rate)

### 安裝與執行
1. 複製此儲存庫到本地端：
   ```bash
   git clone [https://github.com/tsengwt1210/](https://github.com/tsengwt1210/)[你的Repo名稱].git
