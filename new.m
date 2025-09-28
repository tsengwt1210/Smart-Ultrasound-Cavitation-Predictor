function mainProgram
    % 主程式變數
    clear all;
    close all;
    global img isRunning h ratio dynamic_line;
    isRunning = true; % 控制循環的標誌
    ratio = [];       % 初始化比例
    dynamic_line = []; % 用於顯示動態連線

    % 初始化相機
    cam = webcam("USB 2.0 Camera");

    % 創建實時預覽窗口
    h = figure;
    set(h, 'KeyPressFcn', @keyPressCallback); % 設置按鍵回調函數

    disp('實時預覽已啟動。');
    disp('按 "r" 設置比例，按 "h" 手動輸入比例，按 "c" 截圖，按 "m" 測量面積，按 "p" 預測面積，按 "q" 關閉程式。');

    try
        % 實時顯示循環
        while isRunning && isvalid(h) % 檢查窗口是否關閉
            img = snapshot(cam); % 捕捉當前畫面
            imshow(img, 'Parent', gca); % 在當前窗口顯示畫面
            % 在這裡確保清空舊的文字，避免重疊
            delete(findall(gca, 'Type', 'text'));           
            drawnow; % 強制更新圖形窗口
            pause(0.1); % 限制拍攝頻率，避免 timeout
        end
    catch ME
        disp('程式已被手動中止或出現錯誤。');
        disp(ME.message);
    end
    
    % 清理資源
    if exist('cam', 'var') && isa(cam, 'webcam')
        clear cam;
    end
    if isvalid(h)
        close(h);
    end

    % 按鍵回調函數
    function keyPressCallback(~, event)
        persistent screenshotCount;
        if isempty(screenshotCount)
            screenshotCount = 0; % 初始化截圖計數
        end

        switch event.Key
            case 'r' % 兩點測距設置比例
                if ~isempty(img)
                    disp('比例設定模式，請點選兩個參考點。');

                    % 開啟 figure(2) 讓用戶選擇兩點
                    fig2 = figure;
                    imshow(img);
                    title('請選擇比例基準的起點 (左鍵點選)');
                    hold on;

                    % 選擇第一點
                    [x1, y1] = ginput(1);
                    plot(x1, y1, 'ro', 'MarkerSize', 5, 'LineWidth', 2);
                    title('請選擇比例基準的終點 (左鍵點選)');

                    % 選擇第二點
                    [x2, y2] = ginput(1);
                    plot(x2, y2, 'go', 'MarkerSize', 5, 'LineWidth', 2);
                    line([x1 x2], [y1 y2], 'Color', 'r', 'LineWidth', 2);
                    title('比例設定完成');

                    % 計算比例
                    distance_ref = sqrt((x1 - x2)^2 + (y1 - y2)^2);
                    ratio = 1 / distance_ref; % 設定 1 mm/pixel
                    disp(['比例設定完成，比例為: ', num2str(ratio), ' mm/pixel']);

                    hold off;

                    % **自動關閉 figure(2)**
                    close(fig2);
                else
                    disp('無法加載影像，請確保影像存在。');
                end

            
            case 'h' % 手動輸入比例
                ratio = input('請輸入比例 (mm/pixel): ');
                disp(['已手動設定比例為: ', num2str(ratio), ' mm/pixel']);

            case 'c' % 截圖
                checkRatio();
                if ~isempty(img)
                    screenshotCount = screenshotCount + 1;
                    filename = sprintf('screenshot_%d.png', screenshotCount);
                    imwrite(img, filename);
                    disp(['畫面已保存為: ', filename]);
                else
                    disp('未捕捉到畫面，無法保存。');
                end

            case 'm' % 面積測量
                if isempty(ratio)
                    choice = input('尚未設定比例。是否要繼續以像素為單位進行測量？(y/n): ', 's');
                    if ~strcmpi(choice, 'y')
                        disp('請先按 "r" 設置比例後再進行測量。');
                        return;
                    else
                        disp('將以像素平方為單位進行面積計算。');
                    end
                end

                if ~isempty(img)
                    disp('開始面積測量');
                    
                    % 創建 figure(2) 進行選取
                    fig2 = figure;
                    imshow(img);
                    title('請使用滑鼠點選多邊形範圍，按右鍵或雙擊結束。');
                    hold on;
                    x = [];
                    y = [];
                    button = 1;
                    while button == 1
                        [x_new, y_new, button] = ginput(1);
                        if button == 1
                            x = [x; x_new];
                            y = [y; y_new];
                            if length(x) > 1
                                delete(dynamic_line);
                                dynamic_line = plot([x; x(1)], [y; y(1)], 'r-', 'LineWidth', 2);
                            else
                                plot(x_new, y_new, 'ro', 'MarkerSize', 5, 'LineWidth', 2);
                            end
                        end
                    end

                    % 計算並顯示面積
                    if length(x) > 2
                        area_pixel = polyarea(x, y);
                        if isempty(ratio)
                            disp(['選取區域的面積為: ', num2str(area_pixel), ' pixels^2']);
                        else
                            area_real = area_pixel * (ratio^2);
                            disp(['選取區域的實際面積為: ', num2str(area_real), ' mm^2']);
                        end
                    else
                        disp('點選的範圍不足，無法計算面積。');
                    end
                    hold off;
                    
                    % **自動關閉 figure(2)**
                    close(fig2);
                else
                    disp('無法加載影像，請確保影像存在。');
                end

                % 建立二值遮罩
                bw = poly2mask(x, y, size(img, 1), size(img, 2));
                mask = uint8(bw) * 255;

                % 是否要標示單位在檔名（可有可無）
                if isempty(ratio)
                    suffix = 'px';
                else
                    suffix = 'mm';
                end

                % 自動命名儲存檔案（使用 datetime 建議格式）
                t = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
                outputID = char(t);
                filename_img = ['dataset/input/img_' outputID '_' suffix '.png'];
                filename_mask = ['dataset/target/mask_' outputID '_' suffix '.png'];

                % 確保資料夾存在
                if ~exist('dataset/input', 'dir'); mkdir('dataset/input'); end
                if ~exist('dataset/target', 'dir'); mkdir('dataset/target'); end
                
                imwrite(img, filename_img);
                imwrite(mask, filename_mask);
                disp(['已儲存原圖與 mask：', filename_img]);

            case 'p' % 預測面積
                predicted_area = predictCavitationArea();
                
                if ~isempty(predicted_area)
                    disp('在畫面中顯示預測結果...');
                    % 獲取當前圖像的尺寸
                    [h_img, w_img, ~] = size(img);
                    % 計算文字位置（例如在右下角）
                    x_pos = w_img - 10;
                    y_pos = h_img - 10;
                    % 格式化顯示文字
                    text_str = sprintf('預測面積: %.2f mm^2', predicted_area);
                    % 在圖像上顯示文字
                    text(x_pos, y_pos, text_str, ...
                         'Color', 'red', ...
                         'FontSize', 12, ...
                         'FontWeight', 'bold', ...
                         'HorizontalAlignment', 'right', ...
                         'VerticalAlignment', 'bottom');
                else
                    disp('預測失敗，無法在畫面上顯示結果。');
                end

            case 'q' % 關閉程式
                disp('程式已結束。');
                isRunning = false;
                if isvalid(h)
                    close(h);
                end
                if exist('cam', 'var') && isa(cam, 'webcam')
                    clear cam;
                end
        end
    end

    % 檢查比例設定
    function checkRatio()
        if isempty(ratio)
            choice = input('尚未設定比例。是否要繼續以像素為單位執行操作？(y/n): ', 's');
            if ~strcmpi(choice, 'y')
                disp('請先按 "r" 或 "h" 設置比例後再執行操作。');
                return;
            else
                disp('將以像素為單位繼續操作。');
            end
        end
    end
end

function [predicted_area] = predictCavitationArea()
    % 此函數根據使用者輸入的 F1_mag, F2_mag 以及選擇的能量強度，
    % 載入對應的神經網路模型來預測空穴面積。

    predicted_area = []; % 初始化回傳值
    
    disp(' ');
    disp('========================================');
    disp('          神經網路面積預測模式');
    disp('========================================');
    
    try
        % --- 步驟 1: 讓使用者選擇能量強度 ---
        fprintf('請選擇要使用的模型對應的能量強度：\n');
        fprintf('  1: 0.8J\n');
        fprintf('  2: 1.0J\n');
        fprintf('  3: 1.2J\n');
        choice_str = input('請輸入選項 (1/2/3) [若不清楚，請直接按 Enter 使用預設的 0.8J]: ', 's');
        
        % 處理使用者輸入
        if isempty(choice_str)
            disp('未輸入選項，使用預設模型: 0.8J');
            energy_level = '0.8J';
        else
            choice = str2double(choice_str);
            switch choice
                case 1
                    energy_level = '0.8J';
                case 2
                    energy_level = '1.0J';
                case 3
                    energy_level = '1.2J';
                otherwise
                    disp('輸入無效，使用預設模型: 0.8J');
                    energy_level = '0.8J';
            end
        end
        
        % --- 步驟 2: 讓使用者輸入 F1_mag 和 F2_mag ---
        F1_mag = input('請輸入 F1_mag 的能量強度: ');
        F2_mag = input('請輸入 F2_mag 的能量強度: ');
        
        if ~isnumeric(F1_mag) || ~isnumeric(F2_mag) || isempty(F1_mag) || isempty(F2_mag)
            error('輸入值必須為非空的數值。');
        end
        
    catch ME
        disp(['輸入錯誤: ', ME.message]);
        disp('預測已取消。');
        disp('========================================');
        return;
    end
    
    % --- 步驟 3: 載入模型並進行預測 ---
    % 假設您的模型是這樣命名的: 'trained_model_0.8J.mat', 'trained_model_1.0J.mat' 等
    % 您可以根據您訓練程式儲存的檔名格式來修改這裡
    model_filename = ['trained_model_' energy_level '.mat'];
    
    try
        if ~exist(model_filename, 'file')
            error(['找不到模型檔案: ', model_filename, '。請確認模型已訓練並存放在當前路徑。']);
        end
        
        disp(['正在載入模型: ', model_filename]);
        S_model = load(model_filename);
        
        % 檢查模型檔案是否包含必要變數
        if ~isfield(S_model, 'net') || ~isfield(S_model, 'T_max')
            error('載入的模型檔案不完整，缺少 "net" 或 "T_max" 變數。');
        end
        
        net = S_model.net;
        T_max = S_model.T_max;
        
        % 準備輸入資料並預測
        input_data = [F1_mag; F2_mag];
        predicted_area_normalized = net(input_data);
        
        % 使用 T_max 進行反正規化，得到真實預測面積
        predicted_area = predicted_area_normalized * T_max;
        
        disp(' ');
        disp('----------- 預測結果 -----------');
        disp(['使用模型: ', strrep(model_filename, '.mat', '')]);
        disp(['輸入 F1_mag: ', num2str(F1_mag)]);
        disp(['輸入 F2_mag: ', num2str(F2_mag)]);
        fprintf('預測的超音波空穴效應面積為: %.4f mm^2\n', predicted_area);
        disp('----------------------------------');
        
    catch ME
        disp(['預測失敗: ', ME.message]);
        predicted_area = []; % 確保失敗時回傳空值
    end
    disp('========================================');
end