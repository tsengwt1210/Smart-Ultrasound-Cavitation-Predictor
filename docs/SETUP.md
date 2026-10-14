# 執行說明

先從 BPNN 範例確認 MATLAB 環境。GUI、影像分割與外接硬體各有額外需求，可分開準備。

## BPNN 面積預測

必要環境是 MATLAB 與 Deep Learning Toolbox。三份模型保存為 MATLAB `network` 物件，只有 MATLAB 主程式仍無法執行推論；安裝了哪些工具箱可用 `ver` 查看。

將 MATLAB Current Folder 切換至專案根目錄後執行：

```matlab
results = demo_bpnn();
```

此範例從 `test.xlsx` 的第 1、9、17 列，各取第一組 F1/F2 特徵，依序載入 `0.8J`、`1.0J`、`1.2J` 模型，回傳含輸入及預測面積的表格。不會訓練或覆寫模型。

也可以直接呼叫：

```matlab
[area_mm2, details] = predict_area(23931, 4013, '0.8J');
disp(details.normalized_area);
disp(details.T_max);
```

`F1_mag`、`F2_mag` 必須是有限的實數純量。能量字串須為 `'0.8J'`、`'1.0J'`、`'1.2J'`；省略時採 `0.8J`。函數預設從自身所在資料夾尋找模型，因此加入 MATLAB path 後，不必保持在專案目錄：

```matlab
addpath('/path/to/Smart-Ultrasound-Cavitation-Predictor');
area_mm2 = predict_area(23931, 4013, '0.8J');
```

第四個參數可指定包含相同模型檔名的資料夾。特徵的量測定義與尺度必須與原實驗一致，任意輸入不具有可解釋的物理意義。

## 驗證推論入口

```matlab
results = runtests('tests/test_predict_area.m');
table(results)
```

測試比較三模型、各五組既有特徵的原始 `net(x) * T_max` 與新入口輸出，並檢查換工作目錄、非法輸入與缺少模型。沒有 Deep Learning Toolbox 時，依賴推論的測試會略過；應查看 Passed、Failed 與 Incomplete，不能把略過當作成功。實際檢查結果見[驗證紀錄](VALIDATION.md)。

## GUI 與影像模組

GUI 使用 MATLAB App Designer。影像操作需 Image Processing Toolbox；U-Net 另需 Deep Learning Toolbox，標註及 `pixelLabel` datastore 仍需 Computer Vision Toolbox。使用相機時需 MATLAB Support Package for USB Webcams，原程式的裝置名是 `USB 2.0 Camera`，應依 `webcamlist` 修改。Arduino／LCD 為選配，需使用與程式相容的序列埠及裝置端程式。

在 App Designer 開啟 `newver4.mlapp` 查看及執行現有整合介面。請先閱讀 [GUI 說明](GUI.md)，確認想使用的分頁是否已具備模型、影像和硬體。

### Mask R-CNN 的 Python 環境

歷史依賴檔以 Python 3.9、PyTorch 2.0.1、torchvision 0.15.2 為基準。CPU 與 CUDA 11.8 兩種環境擇一安裝：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements_cpu.txt
# 有相容 NVIDIA 環境時，改用 requirements_gpu.txt。
```

PyTorch 的版本組合與下載來源可查[官方歷史版本說明](https://pytorch.org/get-started/previous-versions/)。套件版本保留原專案設定，本次未重新安裝完整 Python／GPU 環境。

另行準備與本專案架構相符的 `maskrcnn_model.pth`，再執行：

```bash
python rcnn_predict.py maskrcnn_model.pth input.png output.png
```

本庫未提供該權重。函數顯式設 `weights=None`、`weights_backbone=None`，載入指定的完整 checkpoint，不自動下載預訓練 backbone；參數依 [torchvision 0.15 官方文件](https://docs.pytorch.org/vision/0.15/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html) 核對。只載入來源可信的模型檔。

GUI 透過 PATH 中的 `python` 執行腳本。若命令列可跑、GUI 卻找不到套件，請確認兩者使用相同環境。

### U-Net

本庫 `Unet_V6.m` 是專題快照，預設從 `data/images/` 讀取影像、從 `data/annotations/gTruth.mat` 載入標註，結果寫入 `outputs/unet_masks/`。原始影像、groundTruth 和訓練權重未包含在本庫，MAT 檔內的影像來源路徑也需與本機相符。

完整版本比較與資料準備以 [BioUS-Net](https://github.com/tsengwt1210/BioUS-Net) 為入口。分割用的比例尺、最小區域面積與後處理參數應按影像條件設定；保留腳本中的 FFT 延伸參數不能視為已校準的超音波量測結果。

## 常見問題

| 現象 | 檢查方式 |
| --- | --- |
| 提示需要 Deep Learning Toolbox | 在 MATLAB 執行 `ver`，確認目前版本有安裝工具箱 |
| 找不到 BPNN 模型 | 保留三份 `trained_model_*.mat` 與 `predict_area.m` 的相對位置 |
| 輸出面積不合理 | 檢查能量、F1/F2 順序、量測尺度及是否超出原實驗特徵範圍 |
| GUI 無法啟動相機 | 檢查 USB Webcams 支援套件、`webcamlist` 與程式內裝置名稱 |
| GUI 找不到 R-CNN 權重 | 將相符的 `maskrcnn_model.pth` 放在 GUI 工作目錄；本庫沒有附檔 |
| U-Net 找不到標註中的圖片 | 檢查 groundTruth 物件內的來源路徑；只改外部資料夾名稱仍可能不足 |
