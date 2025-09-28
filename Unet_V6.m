%% ==== U-Net 全流程訓練與圈選腳本 v5.4 (最終功能整合版) ====
% 版本說明：
% v5.4: 根據 v5.2 的優點進行最終整合，並修復錯誤。
%       1. [修復] 將 semanticseg 替換為 predict，解決 "Undefined function" 錯誤，不再依賴 Computer Vision Toolbox。
%       2. [整合] 加入 v5.2 的所有優點：信心分數計算、bwareaopen 預濾波、強化的單張分析、專業版 FFT、進度條等。
% v5.3: 將特徵提取從空間域 (cycles/um) 全面改回時間域 (Hz)。

clear; clc; close all;
%% ===== Step 1: 全域參數設定 =====
gTruthPath_default = 'D:\專題\U-net\scripts\v3table_rebuilt.mat'; % 預設 gTruth 檔案路徑
imageFolderPath = 'D:\專題\U-net\images_2';       % 包含所有原始圖片的資料夾
maskFolderPath = 'D:\專題\U-net\masks\V6';    % 儲存產生出來的 mask 與結果圖
pixelSize_sq_um = 1 * 1; % 【重要】每像素代表的實際"面積" (例如 0.5um * 0.5um = 0.25)。
minAreaThreshold_px = 30000; % 面積計算時，小於此像素數的物件將被過濾
morphologyRadius = 15; % 形態學閉運算的半徑
maskEffectValue = 50;  % 亮度調整圖的效果強度。

% --- 時間頻譜特徵提取參數 ---
% !!請根據您的物理設定修改以下關鍵數值!!
time_per_pixel_s = 2e-6; % 【重要】Y軸方向上，每個像素代表的秒數 (seconds)。
sampling_rate_hz = 1 / time_per_pixel_s; % 取樣率 (單位: Hz)
target_f1_hz = 1.05;  % 目標頻率1 (單位: Hz)
target_f2_hz = 4.08;  % 目標頻率2 (單位: Hz)

fprintf('====================================================\n');
fprintf('    U-Net 全流程腳本 v5.4 - 最終功能整合版\n');
fprintf('====================================================\n');
%% ===== 主流程控制迴圈 =====
while true
    fprintf('\n===== 主選單 =====\n');
    fprintf('[1] 產生標籤遮罩 (Generate Masks from gTruth)\n');
    fprintf('[2] 訓練新模型 (Train a New Model)\n');
    fprintf('[3] 使用現有模型進行批次預測與分析 (Batch Predict with Existing Model)\n');
    fprintf('[4] 從單張圖片提取特徵 (Analyze a Single Image)\n');
    fprintf('[5] 離開程式 (Exit)\n');
    
    choice = input('請輸入您的選擇 [1-5]: ', 's');
    
    switch choice
        case '1'
            %% --- 任務 1: 產生標籤遮罩 ---
            fprintf('\n--- [任務 1: 產生標籤遮罩] ---\n');
            [gTruth, classNames] = loadAndPrepareGTruth(gTruthPath_default);
            if isempty(gTruth), continue; end
            generateMasksFromGTruth(gTruth, classNames, maskFolderPath);
            
        case '2'
            %% --- 任務 2: 訓練新模型 ---
            fprintf('\n--- [任務 2: 訓練新模型] ---\n');
            disp('正在建立與準備訓練資料集 (採用檔名對齊)...');
            [~, classNames] = loadAndPrepareGTruth(gTruthPath_default); 
            
            [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionAlignedSets(imageFolderPath, maskFolderPath, classNames, 0.8);
            if isempty(imdsTrain)
                disp('❌ 訓練集為空，請檢查圖片與遮罩檔名是否匹配 (_mask.png)。返回主選單。');
                continue;
            end
            fprintf('✅ 資料集切分完成: %d 訓練, %d 驗證\n', numel(imdsTrain.Files), numel(imdsVal.Files));
            targetSize = [512 512];
            augmenter = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true, 'RandRotation',[-20, 20]);
            dsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);
            dsVal = pixelLabelImageDatastore(imdsVal, pxdsVal);
            dsTrain = transform(dsTrain, @(data) resizeImageAndLabel(data, targetSize));
            dsVal = transform(dsVal, @(data) resizeImageAndLabel(data, targetSize));
            disp('✅ 資料集準備完成');
            
            inputSize = [targetSize, 3];
            lgraph = unetLayers(inputSize, numel(classNames));
            options = trainingOptions('adam', 'InitialLearnRate', 1e-3, 'MaxEpochs', 30, ...
                'MiniBatchSize', 4, 'Shuffle', 'every-epoch', 'ValidationData', dsVal, 'Plots', 'training-progress');
            
            disp('🚀 開始訓練 U-Net 模型...');
            [net, info] = trainNetwork(dsTrain, lgraph, options);
            disp('✅ U-Net 訓練完成');
            
            if lower(input('是否要儲存模型? (y/n) [y]: ', 's')) ~= 'n'
                dateStr = datestr(now, 'yyyymmdd');
                modelFileName = sprintf('trainedUnet_%s.mat', dateStr);
                save(modelFileName, 'net', 'classNames', 'info', 'targetSize');
                fprintf('✅ 模型已儲存為 %s\n', modelFileName);
            end
            
        case '3'
            %% --- 任務 3: 使用模型進行批次預測 ---
            fprintf('\n--- [任務 3: 使用模型進行批次預測與分析] ---\n');
            
            [file, path] = uigetfile('*.mat', '請選擇要載入的模型 .mat 檔案');
            if isequal(file, 0), disp('⚠️ 已取消選擇模型，返回主選單。'); continue; end
            
            fprintf('正在載入模型: %s\n', fullfile(path, file));
            loadedData = load(fullfile(path, file));
            if isfield(loadedData, 'net') && isfield(loadedData, 'classNames')
                net = loadedData.net;
                loadedClassNames = loadedData.classNames;
            else
                fprintf('❌ .mat 檔案中缺少變數 (net, classNames)。返回主選單。\n'); continue;
            end
            fprintf('✅ 模型載入成功。\n');
            
            nonBackgroundClasses = loadedClassNames(~strcmpi(loadedClassNames, 'background'));
            if isempty(nonBackgroundClasses), error('錯誤：找不到非 "background" 的目標類別。'); end
            [selection, ok] = listdlg('PromptString', {'選擇要分析的目標類別:'}, 'SelectionMode', 'single', 'ListString', nonBackgroundClasses);
            if ~ok, disp('⚠️ 已取消選擇類別，返回主選單。'); continue; end
            targetClassName = nonBackgroundClasses{selection};
            fprintf('🎯 已選擇分析目標: %s\n', targetClassName);
            
            generateImages = (lower(input('是否要產生預測疊圖 (prediction images)? (y/n) [y]: ', 's')) ~= 'n');
            doFeatureExtraction = (lower(input('是否要提取 FFT 頻譜特徵? (y/n) [y]: ', 's')) ~= 'n');
            fprintf('\n請選擇要預測的對象:\n [1] 驗證集 [2] 訓練集 [3] 新的圖片資料夾\n');
            predictChoice = input('請輸入您的選擇 [1-3]: ', 's');
            
            imdsToPredict = []; outputFileName = ''; description = '';
            switch predictChoice
                case {'1', '2'}
                    disp('準備資料集 (採用檔名對齊)...');
                    [imdsAll, ~] = buildAlignedDatastores(imageFolderPath, maskFolderPath, loadedClassNames);
                    rng('default'); 
                    [imdsTrain, imdsVal, ~, ~] = partitionImdsPxds(imdsAll, imdsAll, 0.8);
                    if predictChoice == '1'
                        imdsToPredict = imdsVal;
                        outputFileName = sprintf('prediction_results_validation_%s.xlsx', targetClassName);
                        description = '驗證集 (Validation Set)';
                    else
                        imdsToPredict = imdsTrain;
                        outputFileName = sprintf('prediction_results_training_%s.xlsx', targetClassName);
                        description = '訓練集 (Training Set)';
                    end
                case '3'
                    newImgFolder = uigetdir([], '選擇要圈選的新圖片資料夾');
                    if newImgFolder ~= 0
                        imdsToPredict = imageDatastore(newImgFolder);
                        outputFileName = sprintf('prediction_results_new_images_%s.xlsx', targetClassName);
                        description = ['新資料夾: ' newImgFolder];
                    else
                        disp('⚠️ 已取消選擇資料夾，返回主選單。'); continue;
                    end
                otherwise, disp('無效的選擇，返回主選單。'); continue;
            end
            
            if ~isempty(imdsToPredict.Files)
                calcAreasAndSave(imdsToPredict, description, net, loadedClassNames, targetClassName, pixelSize_sq_um, maskFolderPath, outputFileName, ...
                    minAreaThreshold_px, morphologyRadius, generateImages, maskEffectValue, ...
                    doFeatureExtraction, target_f1_hz, target_f2_hz, sampling_rate_hz);
            else
                fprintf('⚠️ 在指定的路徑中找不到任何圖片，返回主選單。\n');
            end
            
        case '4'
            %% --- 任務 4: 從單張圖片提取特徵 ---
            fprintf('\n--- [任務 4: 從單張圖片提取特徵] ---\n');
            
            [file, path] = uigetfile('*.mat', '請選擇要載入的模型 .mat 檔案');
            if isequal(file, 0), disp('⚠️ 已取消選擇模型，返回主選單。'); continue; end
            
            fprintf('正在載入模型: %s\n', fullfile(path, file));
            loadedData = load(fullfile(path, file));
            if isfield(loadedData, 'net') && isfield(loadedData, 'classNames')
                net = loadedData.net;
                loadedClassNames = loadedData.classNames;
            else
                fprintf('❌ .mat 檔案中缺少變數 (net, classNames)。返回主選單。\n'); continue;
            end
            fprintf('✅ 模型載入成功。\n');
            predictAndAnalyzeSingleImage(net, loadedClassNames, pixelSize_sq_um, maskFolderPath, ...
                    minAreaThreshold_px, morphologyRadius, maskEffectValue, ...
                    target_f1_hz, target_f2_hz, sampling_rate_hz);
        case '5'
            %% --- 任務 5: 離開 ---
            fprintf('程式已結束。\n'); break;
            
        otherwise
            fprintf('無效的選擇，請重新輸入。\n');
    end
end
%% ===== 副函式 (Helper Functions) =====
function [gTruth, classNames] = loadAndPrepareGTruth(defaultPath)
    gTruth = []; classNames = [];
    [filenames, path] = uigetfile('*.mat', '選擇一個或多個 groundTruth .mat 檔案', defaultPath, 'MultiSelect', 'on');
    if isequal(filenames, 0), disp('⚠️ 已取消檔案選擇。'); return; end
    if ~iscell(filenames), filenames = {filenames}; end
    allSources = {}; allLabels = {}; labelDefs = [];
    for i = 1:numel(filenames)
        S = load(fullfile(path, filenames{i}));
        fn = fieldnames(S);
        currentGTruth = [];
        for j = 1:numel(fn)
            if isa(S.(fn{j}), 'groundTruth'), currentGTruth = S.(fn{j}); break; end
        end
        if isempty(currentGTruth), warning('檔案 %s 中沒有找到 groundTruth 物件，已跳過。', filenames{i}); continue; end
        if i == 1 || isempty(labelDefs), labelDefs = currentGTruth.LabelDefinitions;
        elseif ~isequal(labelDefs, currentGTruth.LabelDefinitions), error('❌ 檔案 "%s" 的標籤定義不一致。', filenames{i}); end
        allSources = [allSources; currentGTruth.DataSource.Source];
        allLabels = [allLabels; currentGTruth.LabelData];
    end
    if isempty(allSources), error('❌ 您選擇的檔案中都沒有有效的 groundTruth 物件。'); end
    gTruth = groundTruth(groundTruthDataSource(allSources), labelDefs, allLabels);
    fprintf('✅ 已成功合併 %d 個檔案，總計 %d 筆資料。\n', numel(filenames), height(gTruth.DataSource.Source));
    classNames = labelDefs.Name;
    if ~any(strcmpi(classNames, 'background')), classNames = ['background'; classNames];
    else, bgIdx = strcmpi(classNames, 'background'); classNames = [classNames(bgIdx); classNames(~bgIdx)]; end
    fprintf('✅ 類別設定完成: %s\n', strjoin(classNames, ', '));
end

function generateMasksFromGTruth(gTruth, classNames, maskFolder)
    fprintf('正在產生影像遮罩 (Masks)...\n');
    if ~exist(maskFolder, 'dir'), mkdir(maskFolder); end
    numImages = height(gTruth.DataSource.Source);
    try
        if isempty(gcp('nocreate')), parpool; end
        parfor i = 1:numImages, generate_mask_helper(i, gTruth, classNames, maskFolder); end
        fprintf('✅ 平行運算執行成功。\n');
    catch ME
        fprintf('⚠️ 平行運算失敗，轉為標準迴圈。錯誤: %s\n', ME.message);
        for i = 1:numImages, generate_mask_helper(i, gTruth, classNames, maskFolder); end
    end
    disp('✅ 已轉出 segmentation mask');
end

function generate_mask_helper(idx, gTruth, classNames, maskFolder)
    imgPath = gTruth.DataSource.Source{idx};
    imgInfo = imfinfo(imgPath);
    localMask = zeros(imgInfo.Height, imgInfo.Width, 'uint8');
    for lblIdx = 2:numel(classNames)
        className = classNames{lblIdx};
        if ismember(className, gTruth.LabelData.Properties.VariableNames)
            polygons = gTruth.LabelData{idx, className};
            if ~isempty(polygons) && iscell(polygons)
                for p = 1:numel(polygons)
                    polyXY = polygons{p};
                    if ~isempty(polyXY) && size(polyXY,1) > 2
                        tempBinaryMask = poly2mask(polyXY(:,1), polyXY(:,2), imgInfo.Height, imgInfo.Width);
                        localMask(tempBinaryMask) = (lblIdx - 1);
                    end
                end
            end
        end
    end
    [~, name, ~] = fileparts(imgPath);
    imwrite(localMask * 255, fullfile(maskFolder, [name '_mask.png']));
end

function [imds, pxds] = buildAlignedDatastores(imageFolder, maskFolder, classNames)
    imds = imageDatastore(imageFolder);
    maskFiles = cell(numel(imds.Files), 1);
    validIdx = false(numel(imds.Files), 1);
    for i = 1:numel(imds.Files)
        [~, fname, ~] = fileparts(imds.Files{i});
        expectedMaskPath = fullfile(maskFolder, [fname '_mask.png']);
        if isfile(expectedMaskPath)
            maskFiles{i} = expectedMaskPath;
            validIdx(i) = true;
        end
    end
    imds = subset(imds, validIdx);
    maskFiles = maskFiles(validIdx);
    if isempty(imds.Files), pxds = []; return; end
    if numel(classNames) ~= 2
        warning('偵測到多於1個非背景類別，請手動確認 labelIDs 的像素值設定！');
        labelIDs = 0:(numel(classNames)-1);
    else
        labelIDs = [0; 255]; 
    end
    pxds = pixelLabelDatastore(maskFiles, classNames, labelIDs);
end

function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionAlignedSets(imageFolder, maskFolder, classNames, trainRatio)
    [imds, pxds] = buildAlignedDatastores(imageFolder, maskFolder, classNames);
    if isempty(imds) || isempty(pxds)
        imdsTrain = []; imdsVal = [];
        pxdsTrain = []; pxdsVal = [];
        return;
    end
    idx = randperm(numel(imds.Files));
    numTrain = round(trainRatio * numel(imds.Files));
    trainIdx = idx(1:numTrain);
    valIdx = idx(numTrain+1:end);
    imdsTrain = subset(imds, trainIdx);
    imdsVal = subset(imds, valIdx);
    pxdsTrain = subset(pxds, trainIdx);
    pxdsVal = subset(pxds, valIdx);
end

function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionImdsPxds(imds, pxds, trainRatio)
    idx = randperm(numel(imds.Files));
    numTrain = round(trainRatio * numel(imds.Files));
    imdsTrain = subset(imds, idx(1:numTrain));
    imdsVal = subset(imds, idx(numTrain+1:end));
    pxdsTrain = subset(pxds, idx(1:numTrain));
    pxdsVal = subset(pxds, idx(numTrain+1:end));
end

function dataOut = resizeImageAndLabel(dataIn, targetSize)
    if istable(dataIn), localImage = dataIn{1, 1}{1}; localLabel = dataIn{1, 2}{1};
    elseif iscell(dataIn), localImage = dataIn{1}; localLabel = dataIn{2};
    else, error('Transform function received unexpected data type: %s', class(dataIn)); end
    dataOut = {imresize(localImage, targetSize), imresize(localLabel, targetSize, 'nearest')};
end

function predictAndAnalyzeSingleImage(net, classNames, pixelSize, resultFolder, minArea_px, morphRadius, maskEffectVal, f1_hz, f2_hz, fs_hz)
    [file, path] = uigetfile({'*.png;*.jpg;*.tif;*.bmp', 'Image Files'}, '請選擇一張要分析的圖片');
    if isequal(file, 0), disp('⚠️ 已取消選擇圖片。'); return; end
    
    imgPath = fullfile(path, file);
    fprintf('正在分析圖片: %s\n', imgPath);
    
    nonBackgroundClasses = classNames(~strcmpi(classNames, 'background'));
    if isempty(nonBackgroundClasses), error('錯誤：找不到任何非 "background" 的目標類別。'); end
    [selection, ok] = listdlg('PromptString', {'選擇要分析的目標類別:'}, 'SelectionMode', 'single', 'ListString', nonBackgroundClasses);
    if ~ok, disp('⚠️ 已取消選擇類別，返回。'); return; end
    targetClassName = nonBackgroundClasses{selection};
    fprintf('🎯 已選擇分析目標: %s\n', targetClassName);
    netInputSize = net.Layers(1).InputSize(1:2);
    se = strel('disk', morphRadius);
    originalImg = imread(imgPath);
    
    resizedImg = imresize(originalImg, netInputSize);
    
    % [v5.4 修復] 使用 predict 獲取分數
    scores_resized = predict(net, resizedImg);
    [confidenceMap_resized, predMask_indices_resized] = max(scores_resized, [], 3);
    
    predMask_indices_originalSize = imresize(predMask_indices_resized, [size(originalImg,1) size(originalImg,2)], 'nearest');
    confidenceMap_originalSize = imresize(confidenceMap_resized, [size(originalImg,1) size(originalImg,2)]);
    
    predMask_categorical = categorical(predMask_indices_originalSize, 1:numel(classNames), classNames);
    if size(originalImg, 3) > 1, grayImg = rgb2gray(originalImg); else, grayImg = originalImg; end
    
    binaryMask = predMask_categorical == targetClassName;
    binaryMask = bwareaopen(binaryMask, round(minArea_px / 2)); % [v5.4 整合] bwareaopen
    closedMask = imclose(binaryMask, se);
    
    cc = bwconncomp(closedMask);
    
    fprintf('--------------------------------------------------\n');
    fprintf('               分析結果 (%s)\n', targetClassName);
    fprintf('--------------------------------------------------\n');
    
    if cc.NumObjects == 0
        fprintf('在圖片中未偵測到有效物件。\n');
    else
        stats = regionprops(cc, 'BoundingBox', 'PixelIdxList');
        objCount = 0;
        for objIdx = 1:cc.NumObjects
            pixelArea = numel(stats(objIdx).PixelIdxList);
            if pixelArea < minArea_px, continue; end
            objCount = objCount + 1;
            
            roi_patch = imcrop(grayImg, stats(objIdx).BoundingBox);
            signal_1d = mean(roi_patch, 2);
            
            [f1_mag, f2_mag] = extract_fft_features(signal_1d, f1_hz, f2_hz, fs_hz);
            mean_confidence = mean(confidenceMap_originalSize(stats(objIdx).PixelIdxList));
            
            fprintf('物件 ID: %d\n', objIdx);
            fprintf('  - 像素面積: %.0f px\n', pixelArea);
            fprintf('  - 實際面積: %.2f (um^2)\n', pixelArea * pixelSize);
            fprintf('  - 平均信心分數: %.4f (%.1f%%)\n', mean_confidence, mean_confidence*100);
            fprintf('  - F1 振幅 (%.2f Hz): %.4f\n', f1_hz, f1_mag);
            fprintf('  - F2 振幅 (%.2f Hz): %.4f\n', f2_hz, f2_mag);
            fprintf('\n');
            figure('Name', sprintf('物件 %d 分析 - %s', objIdx, file));
            subplot(2, 1, 1); imshow(roi_patch); title(sprintf('ROI for Object %d', objIdx));
            subplot(2, 1, 2); plot(signal_1d, 'b-'); title('Mean Signal along Y-axis (Time)');
            xlabel('Pixel Row (Y-axis)'); ylabel('Mean Grayscale Value'); grid on;
            legend(sprintf('F1=%.4f, F2=%.4f', f1_mag, f2_mag));
        end
        if objCount == 0, fprintf('偵測到的物件均小於面積閾值 (%d px)，無有效物件。\n', minArea_px); end
    end
    fprintf('--------------------------------------------------\n');
    
    [~, name] = fileparts(file);
    overlayImg = labeloverlay(originalImg, predMask_categorical, 'Colormap', 'jet', 'Transparency', 0.4);
    
    figure('Name', ['總體分析報告: ' file], 'Position', [100 100 1200 500]);
    subplot(1, 3, 1); imshow(originalImg); title('原始圖片');
    subplot(1, 3, 2); imshow(overlayImg); title('預測結果疊圖');
    subplot(1, 3, 3); imagesc(confidenceMap_originalSize); colormap('jet'); colorbar; axis image; title('信心分數熱圖');
    
    if ~exist(resultFolder, 'dir'), mkdir(resultFolder); end
    imwrite(overlayImg, fullfile(resultFolder, [name '_prediction.png']));
    fprintf('  -> 已儲存: %s\n', [name '_prediction.png']);
    effectMagnitude = abs(maskEffectVal);
    if size(originalImg, 3) == 1, originalImg = cat(3, originalImg, originalImg, originalImg); end
    tempImg = int16(originalImg);
    mask3D = repmat(closedMask, [1, 1, 3]);
    brightImg = uint8(min(255, max(0, tempImg + int16(mask3D) * effectMagnitude)));
    darkImg = uint8(min(255, max(0, tempImg - int16(mask3D) * effectMagnitude)));
    imwrite(brightImg, fullfile(resultFolder, [name '_masked_bright.png']));
    fprintf('  -> 已儲存: %s\n', [name '_masked_bright.png']);
    imwrite(darkImg, fullfile(resultFolder, [name '_masked_dark.png']));
    fprintf('  -> 已儲存: %s\n', [name '_masked_dark.png']);
    fprintf('✅ 結果圖已儲存至: %s\n', resultFolder);
end

function calcAreasAndSave(imdsSet, description, net, classNames, targetClassName, pixelSize, resultFolder, outFile, minArea_px, morphRadius, generateImages, maskEffectVal, doFFT, f1_hz, f2_hz, fs_hz)
    safe_description = strrep(description, '\', '/');
    disp(['正在對 ' description ' 進行推論、計算與特徵提取...']);
    netInputSize = net.Layers(1).InputSize(1:2);
    
    % [v5.4 整合] 根據是否提取 FFT 決定表格欄位
    if doFFT
        results = table('Size', [0, 7], 'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'double'}, ...
            'VariableNames', {'ImageName', 'ObjectID', 'PixelArea_px', 'RealArea_um2', 'MeanConfidence', 'F1_Magnitude', 'F2_Magnitude'});
    else
        results = table('Size', [0, 5], 'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
            'VariableNames', {'ImageName', 'ObjectID', 'PixelArea_px', 'RealArea_um2', 'MeanConfidence'});
    end
    se = strel('disk', morphRadius);
    
    h_wait = waitbar(0, ['正在初始化: ' safe_description '...']);
    
    for i = 1:numel(imdsSet.Files)
        waitbar(i/numel(imdsSet.Files), h_wait, sprintf('處理中 %d / %d: %s', i, numel(imdsSet.Files), safe_description));
        originalImg = readimage(imdsSet, i);
        if size(originalImg, 3) > 1, grayImg = rgb2gray(originalImg); else, grayImg = originalImg; end
        
        resizedImg = imresize(originalImg, netInputSize);
        
        % [v5.4 修復] 使用 predict 獲取分數，取代 semanticseg
        scores_resized = predict(net, resizedImg);
        [confidenceMap_resized, predMask_indices_resized] = max(scores_resized, [], 3);
        
        predMask_indices_originalSize = imresize(predMask_indices_resized, [size(originalImg,1) size(originalImg,2)], 'nearest');
        confidenceMap_originalSize = imresize(confidenceMap_resized, [size(originalImg,1) size(originalImg,2)]);
        predMask_categorical = categorical(predMask_indices_originalSize, 1:numel(classNames), classNames);
        
        binaryMask = predMask_categorical == targetClassName;
        binaryMask = bwareaopen(binaryMask, round(minArea_px / 2)); % [v5.4 整合] bwareaopen
        closedMask = imclose(binaryMask, se);
        cc = bwconncomp(closedMask);
        
        if cc.NumObjects > 0
            stats = regionprops(cc, 'BoundingBox', 'PixelIdxList');
            for objIdx = 1:cc.NumObjects
                pixelArea = numel(stats(objIdx).PixelIdxList);
                if pixelArea < minArea_px, continue; end
                
                [~, fname, fext] = fileparts(imdsSet.Files{i});
                mean_confidence = mean(confidenceMap_originalSize(stats(objIdx).PixelIdxList));
                
                if doFFT
                    roi_patch = imcrop(grayImg, stats(objIdx).BoundingBox);
                    signal_1d = mean(roi_patch, 2);
                    [f1_mag, f2_mag] = extract_fft_features(signal_1d, f1_hz, f2_hz, fs_hz);
                    newRow = {string([fname, fext]), objIdx, pixelArea, pixelArea * pixelSize, mean_confidence, f1_mag, f2_mag};
                else
                    newRow = {string([fname, fext]), objIdx, pixelArea, pixelArea * pixelSize, mean_confidence};
                end
                results = [results; newRow];
            end
        end
        
        if generateImages
            [~, name] = fileparts(imdsSet.Files{i});
            overlayImg = labeloverlay(originalImg, predMask_categorical, 'Colormap', 'jet', 'Transparency', 0.4);
            imwrite(overlayImg, fullfile(resultFolder, [name '_prediction.png']));
            fprintf('  -> 已儲存: %s\n', [name '_prediction.png']);
            effectMagnitude = abs(maskEffectVal);
            if size(originalImg, 3) == 1, originalImg = cat(3, originalImg, originalImg, originalImg); end
            tempImg = int16(originalImg);
            mask3D = repmat(closedMask, [1, 1, 3]);
            brightImg = uint8(min(255, max(0, tempImg + int16(mask3D) * effectMagnitude)));
            darkImg = uint8(min(255, max(0, tempImg - int16(mask3D) * effectMagnitude)));
            imwrite(brightImg, fullfile(resultFolder, [name '_masked_bright.png']));
            fprintf('  -> 已儲存: %s\n', [name '_masked_bright.png']);
            imwrite(darkImg, fullfile(resultFolder, [name '_masked_dark.png']));
            fprintf('  -> 已儲存: %s\n', [name '_masked_dark.png']);
        end
    end
    
    close(h_wait);
    
    if ~isempty(results)
        writetable(results, fullfile(resultFolder, outFile));
        fprintf('✅ 分析完成（過濾 <%d px，連接半徑 %d px），結果已存為 %s\n', minArea_px, morphRadius, outFile);
    else
        fprintf('⚠️ 未偵測到有效物件，未產生 Excel 檔案。\n');
    end
end

function [f1_mag, f2_mag] = extract_fft_features(signal, f1_target, f2_target, fs)
    if isempty(signal) || length(signal) < 4, f1_mag = 0; f2_mag = 0; return; end
    
    nyquist_freq = fs / 2;
    if f1_target > nyquist_freq || f2_target > nyquist_freq
        warning('目標頻率 (%.3f 或 %.3f) 超過奈奎斯特頻率 (%.3f)，特徵將回傳 0。', f1_target, f2_target, nyquist_freq);
        f1_mag = 0; f2_mag = 0; return;
    end
    
    L = length(signal);
    n = (0:L-1)';
    win = 0.5 * (1 - cos(2 * pi * n / L));
    
    signal_win = (signal(:) - mean(signal)) .* win;
    
    nfft = 2^nextpow2(L * 4);
    Y = fft(signal_win, nfft);
    P2 = abs(Y / L);
    P1 = P2(1:nfft/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    
    f_axis = fs*(0:(nfft/2))/nfft;
    [~, idx1] = min(abs(f_axis - f1_target));
    [~, idx2] = min(abs(f_axis - f2_target));
    f1_mag = P1(idx1);
    f2_mag = P1(idx2);
end