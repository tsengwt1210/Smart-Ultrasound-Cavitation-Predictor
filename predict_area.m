function [area_mm2, details] = predict_area(F1_mag, F2_mag, energy, model_dir)
%PREDICT_AREA 使用已保存的 BPNN，以兩個頻譜特徵預測測試片空穴面積。
%   AREA = PREDICT_AREA(F1, F2, '0.8J') 回傳 mm^2，不重新訓練模型。
%   [AREA, DETAILS] = ... 回傳模型路徑、正規化輸出及 T_max。
%   第四個參數可指定模型資料夾；預設為本函數所在資料夾。
%   F1/F2 的定義與量測尺度必須與原始實驗一致。

    validateattributes(F1_mag, {'numeric'}, {'real','finite','scalar'}, mfilename, 'F1_mag');
    validateattributes(F2_mag, {'numeric'}, {'real','finite','scalar'}, mfilename, 'F2_mag');
    if nargin < 3
        energy = '0.8J';
    end
    if ~(ischar(energy) && isrow(energy)) && ~(isstring(energy) && isscalar(energy))
        error('Cavitation:InvalidEnergy', 'energy 必須是 0.8J、1.0J 或 1.2J。');
    end
    energy = char(energy);
    if ~ismember(energy, {'0.8J','1.0J','1.2J'})
        error('Cavitation:InvalidEnergy', 'energy 必須是 0.8J、1.0J 或 1.2J。');
    end
    if nargin < 4
        model_dir = fileparts(mfilename('fullpath'));
    end
    if ~(ischar(model_dir) && isrow(model_dir)) && ~(isstring(model_dir) && isscalar(model_dir))
        error('Cavitation:InvalidModelDirectory', 'model_dir 必須是單一路徑。');
    end

    model_file = fullfile(model_dir, ['trained_model_' energy '.mat']);
    if ~isfile(model_file)
        error('Cavitation:ModelNotFound', '找不到模型：%s', model_file);
    end
    if isempty(which('network'))
        error('Cavitation:ToolboxRequired', '讀取已保存的 network 模型需要 Deep Learning Toolbox。');
    end
    % 載入原始 network 物件，保留其中的 mapminmax 等前後處理。
    S = load(model_file, 'net', 'T_max');
    if ~isfield(S, 'net') || ~isfield(S, 'T_max')
        error('Cavitation:InvalidModel', '模型必須包含 net 和 T_max。');
    end
    if ~isa(S.net, 'network') || S.net.numInputs ~= 1 || S.net.inputs{1}.size ~= 2
        error('Cavitation:InvalidModel', 'net 必須是接收兩個特徵的 MATLAB network。');
    end
    validateattributes(S.T_max, {'numeric'}, {'real','finite','scalar','positive'}, mfilename, 'T_max');

    input_features = double([F1_mag; F2_mag]);
    normalized_area = S.net(input_features);
    validateattributes(normalized_area, {'numeric'}, {'real','finite','scalar'}, mfilename, 'normalized_area');
    area_mm2 = normalized_area * S.T_max;
    details = struct('energy', energy, 'model_file', model_file, ...
        'input_features', input_features, 'normalized_area', normalized_area, ...
        'T_max', S.T_max);
end
