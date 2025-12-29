function plot_both_row(I1, I2, X, Y, index, CorrectIndex)

% I1, I2: 兩張影像（source 和 target）
% X: source 座標 (n x 2)
% Y: target 座標 (n x 2)
% index: 預測為 inlier 的 index（如 VFC）
% CorrectIndex: ground truth inlier index

n = size(X, 1);
VFCIndex = index;

% ---- 分類四種情況 ----
% 初始化所有點為 False (0)
predicted = false(1, n);
predicted(VFCIndex) = true;

ground_truth = false(1, n);
ground_truth(CorrectIndex) = true;

% 計算 TP, FP, FN, TN
TruePos = find(predicted & ground_truth);     % TP: 預測為 true 且實際為 true
FalsePos = find(predicted & ~ground_truth);   % FP: 預測為 true 但實際為 false
FalseNeg = find(~predicted & ground_truth);   % FN: 預測為 false 但實際為 true
TrueNeg = find(~predicted & ~ground_truth);   % TN: 預測為 false 且實際為 false

% ---- 第一張圖：Matching 線 ----
gap = 20; % 兩張圖之間的間隔
white_gap = ones(size(I1, 1), gap, size(I1, 3)) * 255; % 白色間隔 (255 for uint8)
I_comb = [I1, white_gap, I2];
figure; imshow(I_comb); hold on;
w = size(I1, 2) + gap; % 計算 I2 的偏移寬度（包含間隔）

% 繪製匹配線（需要實作 draw_lines 函數）
draw_lines(X, Y, FalsePos, [1, 0, 0], w);  % FP: red
draw_lines(X, Y, TruePos, [0, 0, 1], w);   % TP: blue
draw_lines(X, Y, FalseNeg, [0, 1, 0], w);  % FN: green

% ---- 評估指標：Precision / Recall / F1 ----
TP = length(TruePos);
FP = length(FalsePos);
FN = length(FalseNeg);
TN = length(TrueNeg);

if (TP + FP) == 0
    precision = 0;
    warning('No positive predictions made, precision set to 0');
else
    precision = TP / (TP + FP);
end

if (TP + FN) == 0
    recall = 0;
    warning('No positive ground truth, recall set to 0');
else
    recall = TP / (TP + FN);
end

if (precision + recall) == 0
    f1 = 0;
    warning('Both precision and recall are 0, F1 set to 0');
else
    f1 = 2 * precision * recall / (precision + recall);
end

fprintf('TP: %d, FP: %d, FN: %d, TN: %d\n', TP, FP, FN, TN);
fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall:    %.2f%%\n', recall * 100);
fprintf('F1 Score:  %.2f%%\n', f1 * 100);

end

% ---- 輔助函數：繪製匹配線 ----
function draw_lines(X, Y, indices, color, offset_w)
    if isempty(indices)
        return;
    end
    
    for i = 1:length(indices)
        idx = indices(i);
        x1 = X(idx, 1);
        y1 = X(idx, 2);
        x2 = Y(idx, 1) + offset_w;  % 加上偏移量
        y2 = Y(idx, 2);
        
        line([x1, x2], [y1, y2], 'Color', color, 'LineWidth', 1);
        plot(x1, y1, '-', 'Color', color, 'MarkerSize', 3, 'MarkerFaceColor', color);
        plot(x2, y2, '-', 'Color', color, 'MarkerSize', 3, 'MarkerFaceColor', color);
    end
end

% ---- 輔助函數：繪製向量 ----
function draw_vectors(X, Y, indices, color)
    if isempty(indices)
        return;
    end
    
    for i = 1:length(indices)
        idx = indices(i);
        x1 = X(idx, 1);
        y1 = X(idx, 2);
        dx = Y(idx, 1) - X(idx, 1);
        dy = Y(idx, 2) - X(idx, 2);
        
        % 使用箭頭繪製向量
        quiver(x1, y1, dx, dy, 0, 'Color', color, 'LineWidth', 1, 'MaxHeadSize', 0.5);
    end
end