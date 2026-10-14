# 變更紀錄

## 2026-10-14 介面圖片更新

- README 與 GUI 說明改用報告原稿的 U-Net、Mask R-CNN、MedSAM 高解析度 PNG，移除舊版低解析度圖片連結。
- 圖片隨專案保存在 `assets/screenshots/`，補上來源、原始解析度，以及較新版截圖與公開程式版本的差異。

## 2026-10-14

- 重寫 README，補上專題流程、個人與團隊分工、資料與模型格式、環境需求、結果限制。
- 新增 `predict_area.m` 與 `demo_bpnn.m`，直接使用本庫既有的三份 BPNN 模型；保留模型前後處理與原始權重。
- 新增 BPNN 推論介面測試，檢查與原 `net(x) * T_max` 的一致性、輸入格式及相對路徑。
- 區分歷史訓練腳本設定與已保存模型，修正 MSE／RMSE 的敘述，說明重複量測及隨機切分限制。
- 補上原本空白的 CPU 依賴檔，修正 GPU 套件索引設定；Python 版本沿用既有環境設定，未重新安裝完整環境。
- 修正 Mask R-CNN 啟動提示檔名，載入本機 checkpoint 時不再嘗試下載預設 backbone。
- 將本庫 U-Net 快照的預設資料路徑改為專案相對路徑，並連結至 BioUS-Net 的版本整理。
- 加入 `.gitignore` 與 `.gitattributes`；三份 GUI、既有模型與數值表沿用原公開檔案。
- 移除沒有對應 LICENSE 檔案的 MIT 標章與未核對的概括成果敘述。

實際檢查方式及尚未驗證項目見[驗證紀錄](docs/VALIDATION.md)。
