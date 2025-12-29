function [result] = histogram_filter(data, bin_width, model, filter_idx)

peak_num = 3 ;

% 計算分段的邊界，與 C++ 相匹配
min_length = round(min(data), 1);
max_length = round(max(data), 1);

% 使用 C++ 方法來決定 bin 的邊界
pitch_list = []; 
pitch_list(end+1) = min_length - bin_width; % 開始比 min 小一個 bin_width
current_pitch = min_length;
while current_pitch < (max_length + bin_width)
    pitch_list(end+1) = current_pitch;
    current_pitch = current_pitch + bin_width;
end

% 取得最大長度來確定分組數量
bins_num = size(pitch_list, 2) - 1;

% 創建存儲索引與數值的 cell 陣列
data_idx = cell(bins_num, 1);
data_bins_num = zeros(bins_num, 1);

if isempty(filter_idx)
    % 遍歷每個 bin，將符合範圍的數據與索引存入
    for i = 1:bins_num
        lower_bound = pitch_list(i);
        upper_bound = pitch_list(i+1);

        % 找到符合該範圍的數據索引
        range = lower_bound < data & data <= upper_bound;

        % 存儲索引與對應數據
        data_idx{i} = find(range); % 存儲原始索引
        data_bins_num(i) = sum(range);
    end
    
else 
    for i = 1:bins_num
        lower_bound = pitch_list(i);
        upper_bound = pitch_list(i+1);

        range = lower_bound < data & data <= upper_bound;

        % 存儲索引與對應數據
        idx = find(range);
        data_idx{i} = filter_idx(idx); % 存儲原始索引
        data_bins_num(i) = sum(range);
    end
end

% 找最高 peak
[~, peak_idx] = max(data_bins_num);

%%%%%%%%%%%% svm抓data %%%%%%%%%%%%
svm_left_idx = max(1, peak_idx-peak_num) : peak_idx-1;
svm_right_idx = peak_idx+1 : min(bins_num, peak_idx+peak_num);

svm_peak = [svm_left_idx, peak_idx, svm_right_idx];

svm_data = data_bins_num(svm_peak)';

if size(svm_left_idx, 2) == 0
    svm_data = [0, 0, 0, svm_data];
elseif size(svm_left_idx, 2) == 1
    svm_data = [0, 0, svm_data];
elseif size(svm_left_idx, 2) == 2
    svm_data = [0, svm_data];
end

if size(svm_right_idx, 2) == 0
    svm_data = [svm_data, 0, 0, 0];
elseif size(svm_right_idx, 2) == 1
    svm_data = [svm_data, 0, 0];
elseif size(svm_right_idx, 2) == 2
    svm_data = [svm_data, 0];
end

svm_data = svm_data / size(data, 1) ;

label = [0];
if model == "ori"
    load('rad_m.mat') ;
    [peak_num, ~, ~] = svmpredict(label, svm_data, rad_m); 
else
    load('scale_m.mat') ;
    [peak_num, ~, ~] = svmpredict(label, svm_data, scale_m); 
end


%%%%%%%%%%%%%%%%%% 最終要抓的 %%%%%%%%%%%%%%%%%%
% 找左右peak
left_idx = max(1, peak_idx-peak_num) : peak_idx-1;
right_idx = peak_idx+1 : min(bins_num, peak_idx+peak_num);

% 合併 peak 與左右 peak 的範圍
final_peak = [left_idx, peak_idx, right_idx];
result = cell2mat(data_idx(final_peak));
