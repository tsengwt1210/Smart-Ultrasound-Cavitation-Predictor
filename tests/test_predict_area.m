function tests = test_predict_area
%TEST_PREDICT_AREA 驗證公開模型的推論介面，不執行訓練或啟動硬體。
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    repo_dir = fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.repo_dir = repo_dir;
    testCase.TestData.old_path = path;
    addpath(repo_dir);
    testCase.TestData.data = readmatrix(fullfile(repo_dir, 'test.xlsx'));
end

function teardownOnce(testCase)
    path(testCase.TestData.old_path);
end

function testSavedModelInference(testCase)
    assumeFalse(testCase, isempty(which('network')), 'Requires Deep Learning Toolbox.');
    energies = {'0.8J','1.0J','1.2J'};
    data = testCase.TestData.data;
    for k = 1:3
        S = load(fullfile(testCase.TestData.repo_dir, ['trained_model_' energies{k} '.mat']));
        row = (k-1)*8+1;
        % 每種能量的五組特徵都需保留 network 物件的原始推論結果。
        for c = 1:5
            x = data(row:row+1,c);
            expected = S.net(x) * S.T_max;
            [actual, details] = predict_area(x(1), x(2), energies{k});
            verifyEqual(testCase, actual, expected, 'AbsTol', 1e-12);
            verifyEqual(testCase, details.T_max, S.T_max);
            verifyTrue(testCase, isfinite(actual));
        end
    end
end

function testIndependentOfCurrentFolder(testCase)
    assumeFalse(testCase, isempty(which('network')), 'Requires Deep Learning Toolbox.');
    original_dir = pwd;
    cleanup = onCleanup(@() cd(original_dir));
    cd(fileparts(testCase.TestData.repo_dir));
    data = testCase.TestData.data;
    actual = predict_area(data(1,1), data(2,1), '0.8J');
    verifyTrue(testCase, isfinite(actual));
end

function testRejectsUnknownEnergy(testCase)
    verifyError(testCase, @() predict_area(1,2,'0.9J'), 'Cavitation:InvalidEnergy');
end

function testRejectsNonfiniteInput(testCase)
    verifyError(testCase, @() predict_area(NaN,2,'0.8J'), 'MATLAB:predict_area:expectedFinite');
    verifyError(testCase, @() predict_area(1,Inf,'0.8J'), 'MATLAB:predict_area:expectedFinite');
end

function testRejectsVectorInput(testCase)
    verifyError(testCase, @() predict_area([1,2],3,'0.8J'), 'MATLAB:predict_area:expectedScalar');
end

function testMissingModel(testCase)
    missing = fullfile(testCase.TestData.repo_dir, 'tests', 'missing-model-directory');
    verifyError(testCase, @() predict_area(1,2,'0.8J',missing), 'Cavitation:ModelNotFound');
end

function testDemo(testCase)
    assumeFalse(testCase, isempty(which('network')), 'Requires Deep Learning Toolbox.');
    results = demo_bpnn();
    verifySize(testCase, results, [3,4]);
    verifyEqual(testCase, results.Energy, ["0.8J";"1.0J";"1.2J"]);
    verifyTrue(testCase, all(isfinite(results.PredictedArea_mm2)));
end
