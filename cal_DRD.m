function [result, Yidx_diff]= cal_DRD(Y, i, Kn, diff_X, Xidx_diff)
% Xidx_diff：X 的index (差集的)，也就是在X的排名
% diff_X：差集集合
% [Yidx_X, ~] = setdiff(neighborhoodX, neighborhoodY) ; % 取X、Y的差集

Yidx_diff =[]; % Y 的index (差集的)，也就是非共同鄰居在Y的排名

%if isempty(diff_X) || size(diff_X,2) == Kn
if isempty(diff_X)
    result = [];
else
    diff = Y(i,:) - Y(diff_X,:);
    Y_dist = sum(diff.^2, 2);

    % 根據 dist 排序，並獲得排序後的索引
    [~, sortIdx] = sort(Y_dist);

    % 根據 sortIdx 將 idx 排序
    Yidx_diff = diff_X(sortIdx);
    
    result = [];
    % 遍历 idx 中的每个值
    for j = 1:length(Xidx_diff)
        x_pos = Xidx_diff(j);
        y_pos = find(Yidx_diff == diff_X(j)) + Kn;
        DRD = abs(x_pos - y_pos);
        result = [result, DRD];
    end

end


