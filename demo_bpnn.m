function results = demo_bpnn()
%DEMO_BPNN 讀取既有資料，每個能量條件展示一筆已保存模型的推論。
%   此範例是載入與推論檢查，不是獨立測試集評估，也不重新訓練。
    repo_dir = fileparts(mfilename('fullpath'));
    data = readmatrix(fullfile(repo_dir, 'test.xlsx'));
    if size(data,1) < 23 || size(data,2) < 5
        error('Cavitation:InvalidData', 'test.xlsx 至少需要 23 列、5 欄。');
    end
    energies = ["0.8J"; "1.0J"; "1.2J"];
    F1_mag = zeros(3,1);
    F2_mag = zeros(3,1);
    predicted_area_mm2 = zeros(3,1);
    for k = 1:3
        row = (k-1)*8+1;
        F1_mag(k) = data(row,1);
        F2_mag(k) = data(row+1,1);
        predicted_area_mm2(k) = predict_area(F1_mag(k), F2_mag(k), energies(k), repo_dir);
    end
    results = table(energies, F1_mag, F2_mag, predicted_area_mm2, ...
        'VariableNames', {'Energy','F1_mag','F2_mag','PredictedArea_mm2'});
    disp(results);
end
