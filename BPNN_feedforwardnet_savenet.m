clear all;
close all;
clc;

% === 使用者選擇資料組 ===
dataset_id = input('請輸入資料組別 (1 / 2 / 3)：');
if ~ismember(dataset_id, [1 2 3])
    error('輸入錯誤，請輸入 1、2 或 3');
end

% === 是否使用已訓練模型 ===
use_trained = input('是否使用已訓練模型？(1=是, 0=否)：');

% === 對應行數範圍 ===
start_row = (dataset_id - 1) * 8 + 1;
end_row = start_row + 6;

% === 讀取 Excel 資料 ===
data = readmatrix('test.xlsx');
if size(data,1) < end_row
    error('Excel 資料不足，請確認格式');
end

% === 資料格式解析 ===
P_base = data(start_row:start_row+1, 1:5);    % 2×5 輸入
T_matrix = data(start_row+2:start_row+6, 1:5);% 5×5 輸出（面積）

% === 展開輸入與輸出 ===
P_all = repelem(P_base, 1, 5);         % 2×25
T_all = reshape(T_matrix, 1, []);      % 1×25

% === 面積正規化 ===
T_max = max(T_all);
T_all = T_all / T_max;

% === 神經網路結構 ===
hidden_layers = [5,3,2,3]; % 神經元組合
layer_str = strjoin(string(hidden_layers), 'x');
model_file = ['trained_model_dataset' num2str(dataset_id) '_' char(layer_str) '.mat'];

if use_trained && isfile(model_file)
    load(model_file, 'net', 'T_max');
    disp(['已載入模型：' model_file]);
else
    net = feedforwardnet(hidden_layers, 'trainlm');
    net.inputs{1}.size = 2;
    net = configure(net, P_all, T_all);
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 0;

    [net, tr] = train(net, P_all, T_all);

    save_choice = input('是否儲存此訓練後模型？(1=是, 0=否)：');
    if save_choice == 1
        save(model_file, 'net', 'T_max');
        disp(['模型已儲存為：' model_file]);
    else
        disp('模型未儲存。');
    end
end

% === 預測與圖表 ===
out = net(P_all);

figure;
plot(T_all, '-rd', 'LineWidth', 1.5); hold on;
plot(out, '-bo', 'LineWidth', 1.5);
xlabel('Data Index');
ylabel('Normalized Area');
legend('Target','NN Output');
title(['Dataset ' num2str(dataset_id) ' Prediction vs. Target']);
