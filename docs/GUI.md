# GUI 版本與操作

`newver2fullscreen.mlapp`、`newver3fullscreen.mlapp`、`newver4.mlapp` 是原本已公開的團隊整合版本。介面程式由趙崇凱負責；此處保留供查看各模組如何串接。可先在 MATLAB App Designer 開啟 `newver4.mlapp` 閱讀回呼函數，完整執行前需備妥對應資料及裝置。

## 各分頁需要的內容

| 分頁 | 輸入與額外需求 | 輸出 |
| --- | --- | --- |
| 影像載入與手動量測 | 本機圖片或 USB 顯微鏡攝影機、比例尺 | 圈選面積、截圖 |
| U-Net | 相容的 `.mat` 模型、圖片、比例尺 | 遮罩／疊圖、面積、模型分數 |
| Mask R-CNN | `rcnn_predict.py`、`maskrcnn_model.pth`、可用的 Python 環境 | 輪廓圖、每個區域的像素面積與分數 |
| BPNN 訓練 | 符合[資料格式](DATA_AND_MODELS.md)的 Excel | 訓練圖與保存模型 |
| BPNN 面積預測 | 能量條件、F1_mag、F2_mag、對應模型 | 預測面積 |

U-Net 與 Mask R-CNN 權重不在本庫。只想檢查數值推論時，直接使用 [`demo_bpnn`](../demo_bpnn.m) 可避開攝影機與 Python 環境需求。

## 手動量測

先載入圖片或啟動相機，設定比例尺後再圈選區域。早期 [`new.m`](../new.m) 使用鍵盤操作：`r` 選兩個相距 1 mm 的參考點、`h` 輸入 `mm/pixel`、`c` 截圖、`m` 圈選多邊形、`p` 呼叫 BPNN 預測、`q` 離開。這組快捷鍵屬於早期程式，不應套用到所有 GUI 分頁。

<img width="598" alt="既有手動量測介面" src="https://github.com/user-attachments/assets/fe781dca-4be2-4161-a6c3-1b6ca19478e8" />

## U-Net 分割

載入相容模型與圖片，執行分割後查看遮罩、熱圖或疊圖。面積的物理單位由比例尺決定，信心分數不能取代與人工標註的比對。

<img width="650" alt="既有 U-Net 分割介面" src="https://github.com/user-attachments/assets/8d5a41bb-3705-41e7-b5e5-786165b82627" />

## Mask R-CNN

GUI 執行 `python rcnn_predict.py <model_path> <image_path> <output_path>`。程式保留類別 1 且分數至少 0.8 的預測，將 mask 以 0.5 閾值二值化，再繪製外輪廓並回傳輪廓面積。現有接口繪製的是輪廓，沒有將 bounding box 畫在輸出圖上。

<img width="650" alt="既有 Mask R-CNN 輪廓介面" src="https://github.com/user-attachments/assets/3d372ccf-65e5-4864-ba68-0e0641b6b203" />

## 版本與驗證狀態

三個 `.mlapp` 都保留原檔，這次沒有重建 GUI，也沒有在實體攝影機或 Arduino 上重新驗收。檔名可用來辨識歷史版本，但不代表各版已做過相同的相容性測試。原本 README 的操作畫面亦是歷史展示，不是本次重新執行的截圖。
