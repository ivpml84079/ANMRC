function [index, pred_IR] = BMRC(X, Y, X_info, Y_info, Kn, lambda_set, eta, r, l_ir, h_ir, mult_low, mult_mid)
% aaaaa = 0;2
% bbbbb = 0;
% IR_mid = 3;
% IR = 3;

% mult_low = 1;
% mult_mid = 2; % 1 2 3都行，0會太小 // 要用1 2 3 0可以做參數設定實驗

% L_1=0;
% L_2=0;
% 
% L1_set = [];
% L2_set = [];
% L_final = [];

Xt = X';
Yt = Y';
[N,D] = size(X);

% % predict inlier rate
% X_scale = X_info(:,1);
% Y_scale = Y_info(:,1);
% scale_ratio = max(X_scale, Y_scale) ./ min(X_scale, Y_scale);
% 
% X_deg = X_info(:,2);
% Y_deg = Y_info(:,2);
% rad_diff = abs(wrapToPi(deg2rad(X_deg - Y_deg)));
% 
% pred_IR = predict_IR(rad_diff, scale_ratio) ;

pred_IR = 0.3;

% filter
idx = get_idx(Xt,Yt,Kn,eta);


% iteration 1
kdtreeX = vl_kdtreebuild(Xt(:,idx));
[neighborhoodX,~] = vl_kdtreequery(kdtreeX, Xt(:,idx), Xt, 'NumNeighbors', Kn+1) ;
neighborhoodX = idx(neighborhoodX);
neighborhoodX = neighborhoodX(2:Kn+1,:);

%tic;
kdtreeY = vl_kdtreebuild(Yt(:,idx));
[neighborhoodY,dist_Y] = vl_kdtreequery(kdtreeY, Yt(:,idx), Yt, 'NumNeighbors', Kn+1) ;
neighborhoodY = idx(neighborhoodY);
neighborhoodY = neighborhoodY(2:Kn+1,:);
dist_Y = dist_Y(2:Kn+1,:);

Common_X = {} ;

lambda = 0;

% choose method
% low inlier rate method
if 0 <= pred_IR && pred_IR < l_ir
    lambda = lambda_set(1,1);
    rank_diff_set_C = cell(N,1); % C表示Common
    rank_diff_set_nC = cell(N,1); % nC表示not Common
    neighbor_set_C = cell(N,1);
    neighbor_set_nC = cell(N,1);
    % not_common_rank_diff_set = cell(N,1);
    % common_neighbor_set = cell(N,1);
    % not_common_neighbor_set = cell(N,1);
    % rank_diff_set_ = cell(N,1);
    % not_common_rank_diff_set_ = cell(N,1);
    % common_neighbor_set_ = cell(N,1);
    % not_common_neighbor_set_ = cell(N,1);
    for i = 1:N
        %tic;
        % 取得 membership 與索引
        XX = neighborhoodX(:, i)';
        [maskX, idxInY] = ismember(XX, neighborhoodY(:, i)');
        
        % 交集的值與索引
        intersectVals = XX(maskX);        % 值
        idxX_intersect = find(maskX);    % X 中的索引
        idxY_intersect = idxInY(maskX);  % Y 中的索引
        rank_diff = abs(idxX_intersect-idxY_intersect);

        % X - Y 差集的值與索引
        diffVals_XY = XX(~maskX);         % 值
        idxX_diff = find(~maskX);        % X 中的索引
        not_common_rank_diff = cal_drift(Y, i, Kn, diffVals_XY, idxX_diff);
        %time1 = toc

        % tic;
        % [X_in_Y, X_idx, Y_idx] = intersect(neighborhoodX(:, i)', neighborhoodY(:, i)', 'stable');
        % 
        % % 計算共同鄰居的排名差異
        % rank_diff = abs(X_idx-Y_idx);
        % 
        % [diff_X, Xidx_diff] = setdiff(neighborhoodX(:, i)', neighborhoodY(:, i)', 'stable') ; % 取X、Y的差集
        % not_common_rank_diff = cal_drift(Y, i, Kn, diff_X, Xidx_diff);
        % 
        % time2 = toc

        % 存排名差
        rank_diff_set_C{i} = rank_diff;
        rank_diff_set_nC{i} = not_common_rank_diff;

        % 直接存值
        neighbor_set_C{i} = intersectVals;
        neighbor_set_nC{i} = diffVals_XY;     
    end

    % 把所有欄位的 rank 差攤平成一個向量
    allDiffs_C = horzcat(rank_diff_set_C{:});

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % figure;
    % hold on;
    % % 畫直方圖 (自訂 bin 邊界)
    % edges = 0:max(allDiffs_C);     % 以 0,1,2,... 為邊界
    % histogram(allDiffs_C, edges);
    % xlabel('Distance-ranking differences');
    % ylabel('Frequency');
    % hold off;
    % saveas(gcf, 'D:\\YC\\paper\\Fig\\rank_diff\\177_0.2_C.png');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 計算平均值與標準差 (共同)
    mu_C = mean(allDiffs_C);
    sig_C = std(allDiffs_C);

    % 把所有欄位的 rank 差攤平成一個向量
    allDiffs_nC = horzcat(rank_diff_set_nC{:});
    
    % 計算平均值與標準差 (非共同)
    mu_nC = mean(allDiffs_nC);
    sig_nC = std(allDiffs_nC);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % figure;
    % hold on;
    % % 畫直方圖 (自訂 bin 邊界)
    % edges = 0:max(allDiffs_nC);     % 以 0,1,2,... 為邊界
    % histogram(allDiffs_nC, edges);
    % xlabel('Distance-ranking differences');
    % ylabel('Frequency');
    % hold off;
    % saveas(gcf, 'D:\\YC\\paper\\Fig\\rank_diff\\177_0.2_nC.png');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % [neighborhoodY2, X_minus_Y_set] = createNY2(Y, neighborhoodX,neighborhoodY, N);
    % [rank_diff_set, mu_common, sig_common, mu_noncom, sig_noncom] = get_diff_info_all(N, Kn, neighborhoodX, neighborhoodY2);
    
    IR_mid = mu_C + mult_mid * sig_C;
    IR_low = mu_nC - mult_low * sig_nC;

    for i = 1:N
        common = neighbor_set_C{i};
        Ncommon = neighbor_set_nC{i};
        common_num = length(common);

        % 針對共同鄰居計算排名差異，將排名差異大的移除
        rank_diff = rank_diff_set_C{i};
        common = common(rank_diff<IR_mid);

        % 針對非共同鄰居計算排名差異，納入排名差異小的
        rank_diff = rank_diff_set_nC{i};

        if IR_low > r * (Kn - common_num)
            IR_low = r * (Kn - common_num);
        end

        Ncommon = Ncommon(rank_diff<=IR_low);

        % if ~isempty(common)&~isempty(Ncommon)
        %     i
        % end

        cal_neighbor = [common Ncommon];
        % if L_1 < length(cal_neighbor)
        %     L_1 = length(cal_neighbor);
        % end
        % 
        % L1_set = [L1_set, length(cal_neighbor)];
        % L_final = [L_final, length(cal_neighbor)];
        Common_X{i} = cal_neighbor';
    end

% middle inlier rate method
elseif l_ir <= pred_IR && pred_IR < h_ir
    lambda = lambda_set(1,2);
    rank_diff_set_C = cell(N,1); % C表示Common
    neighbor_set_C = cell(N,1);
    for i = 1:N
        %tic;
        % 取得 membership 與索引
        XX = neighborhoodX(:, i)';
        [maskX, idxInY] = ismember(XX, neighborhoodY(:, i)');
        
        % 交集的值與索引
        intersectVals = XX(maskX);        % 值
        idxX_intersect = find(maskX);    % X 中的索引
        idxY_intersect = idxInY(maskX);  % Y 中的索引
        rank_diff = abs(idxX_intersect-idxY_intersect);
        %time1 = toc

        % tic;
        % [X_in_Y, X_idx, Y_idx] = intersect(neighborhoodX(:, i)', neighborhoodY(:, i)', 'stable');
        % % 計算共同鄰居的排名差異
        % rank_diff_ = abs(X_idx-Y_idx);
        % time2 = toc

        % 存排名差
        rank_diff_set_C{i} = rank_diff;

        % 直接存值
        neighbor_set_C{i} = intersectVals;
    end

    % 把所有欄位的 rank 差攤平成一個向量
    allDiffs_C = horzcat(rank_diff_set_C{:});

    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % figure;
    % hold on;
    % % 畫直方圖 (自訂 bin 邊界)
    % edges = 0:max(allDiffs_C);     % 以 0,1,2,... 為邊界
    % histogram(allDiffs_C, edges);
    % xlabel('Distance-ranking differences');
    % ylabel('Frequency');
    % hold off;
    % saveas(gcf, 'D:\\YC\\paper\\Fig\\rank_diff\\177_0.4_C.png');
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 計算平均值與標準差 (共同)
    mu_C = mean(allDiffs_C);
    sig_C = std(allDiffs_C);

    IR_mid = mu_C + mult_mid * sig_C;
    for i = 1:N
        common = neighbor_set_C{i};

        % 針對共同鄰居計算排名差異，將排名差異大的移除
        rank_diff = rank_diff_set_C{i};
        common = common(rank_diff<IR_mid);
        
        % if L_1 < length(common)
        %     L_1 = length(common);
        % end
        % 
        % L1_set = [L1_set, length(common)];
        % L_final = [L_final, length(common)];

        Common_X{i} = common';
    end

% high inlier rate method
else
    lambda = lambda_set(1,3);
    for i = 1:N
        % 取得 membership 與索引
        XX = neighborhoodX(:, i)';
        [maskX, ~] = ismember(XX, neighborhoodY(:, i)');
        
        % 交集的值與索引
        intersectVals = XX(maskX);  % 值
        % if L_1 < length(intersectVals)
        %     L_1 = length(intersectVals);
        % end
        % 
        % L1_set = [L1_set, length(intersectVals)];
        % L_final = [L_final, length(intersectVals)];
        Common_X{i} = intersectVals';
    end
end
%My1_C_time = toc
%tic;
WX = Construct_coefficent_matrix(Kn,N,D,X,Common_X);
WY = Construct_coefficent_matrix(Kn,N,D,Y,Common_X);
%My1_W_time = toc
idx = [] ; %原為idx
for i = 1:N
    if ~isnan(WX{i})
        dist = sum((WX{i} - WY{i}).^2);
        
        alpha = size(WX{i},1) / Kn ;

        if dist < lambda*alpha
            idx(end+1) = i ; %原為idx
        end
    end
end


%% iteration 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kdtreeX = vl_kdtreebuild(Xt(:,idx));
[neighborhoodX,~] = vl_kdtreequery(kdtreeX, Xt(:,idx), Xt, 'NumNeighbors', Kn+1) ;
neighborhoodX = idx(neighborhoodX);
neighborhoodX = neighborhoodX(2:Kn+1,:);

%tic;
kdtreeY = vl_kdtreebuild(Yt(:,idx));
[neighborhoodY,dist_Y] = vl_kdtreequery(kdtreeY, Yt(:,idx), Yt, 'NumNeighbors', Kn+1) ;
neighborhoodY = idx(neighborhoodY);
neighborhoodY = neighborhoodY(2:Kn+1,:);
dist_Y = dist_Y(2:Kn+1,:);

Common_X = {} ;

% choose method
% low inlier rate method
if 0 <= pred_IR && pred_IR < l_ir
    lambda = lambda_set(1,1);
    rank_diff_set_C = cell(N,1); % C表示Common
    rank_diff_set_nC = cell(N,1); % nC表示not Common
    neighbor_set_C = cell(N,1);
    neighbor_set_nC = cell(N,1);
    for i = 1:N
        % 取得 membership 與索引
        XX = neighborhoodX(:, i)';
        [maskX, idxInY] = ismember(XX, neighborhoodY(:, i)');
        
        % 交集的值與索引
        intersectVals = XX(maskX);        % 值
        idxX_intersect = find(maskX);    % X 中的索引
        idxY_intersect = idxInY(maskX);  % Y 中的索引
        rank_diff = abs(idxX_intersect-idxY_intersect);

        % X - Y 差集的值與索引
        diffVals_XY = XX(~maskX);         % 值
        idxX_diff = find(~maskX);        % X 中的索引
        not_common_rank_diff = cal_drift(Y, i, Kn, diffVals_XY, idxX_diff);

        % 存排名差
        rank_diff_set_C{i} = rank_diff;
        rank_diff_set_nC{i} = not_common_rank_diff;

        % 直接存值
        neighbor_set_C{i} = intersectVals;
        neighbor_set_nC{i} = diffVals_XY;     
    end

    % 把所有欄位的 rank 差攤平成一個向量
    allDiffs_C = horzcat(rank_diff_set_C{:});
    
    % 計算平均值與標準差 (共同)
    mu_C = mean(allDiffs_C);
    sig_C = std(allDiffs_C);

    % 把所有欄位的 rank 差攤平成一個向量
    allDiffs_nC = horzcat(rank_diff_set_nC{:});
    
    % 計算平均值與標準差 (非共同)
    mu_nC = mean(allDiffs_nC);
    sig_nC = std(allDiffs_nC);

    IR_mid = mu_C + mult_mid * sig_C;
    IR_low = mu_nC - mult_low * sig_nC;

    for i = 1:N
        common = neighbor_set_C{i};
        Ncommon = neighbor_set_nC{i};
        common_num = length(common);

        % 針對共同鄰居計算排名差異，將排名差異大的移除
        rank_diff = rank_diff_set_C{i};
        common = common(rank_diff<IR_mid);

        % 針對非共同鄰居計算排名差異，納入排名差異小的
        rank_diff = rank_diff_set_nC{i};

        if IR_low > r * (Kn - common_num)
            IR_low = r * (Kn - common_num);
        end

        Ncommon = Ncommon(rank_diff<=IR_low);

        cal_neighbor = [common Ncommon];
        % if L_2 < length(cal_neighbor)
        %     L_2 = length(cal_neighbor);
        % end
        % 
        % L2_set = [L2_set, length(cal_neighbor)];
        % L_final = [L_final, length(cal_neighbor)];
        Common_X{i} = cal_neighbor';
    end

% middle inlier rate method
elseif l_ir <= pred_IR && pred_IR < h_ir
    lambda = lambda_set(1,2);
    rank_diff_set_C = cell(N,1); % C表示Common
    neighbor_set_C = cell(N,1);
    for i = 1:N
        % 取得 membership 與索引
        XX = neighborhoodX(:, i)';
        [maskX, idxInY] = ismember(XX, neighborhoodY(:, i)');
        
        % 交集的值與索引
        intersectVals = XX(maskX);        % 值
        idxX_intersect = find(maskX);    % X 中的索引
        idxY_intersect = idxInY(maskX);  % Y 中的索引
        rank_diff = abs(idxX_intersect-idxY_intersect);

        % 存排名差
        rank_diff_set_C{i} = rank_diff;

        % 直接存值
        neighbor_set_C{i} = intersectVals;
    end

    % 把所有欄位的 rank 差攤平成一個向量
    allDiffs_C = horzcat(rank_diff_set_C{:});
    
    % 計算平均值與標準差 (共同)
    mu_C = mean(allDiffs_C);
    sig_C = std(allDiffs_C);

    IR_mid = mu_C + mult_mid * sig_C;
    for i = 1:N
        common = neighbor_set_C{i};

        % 針對共同鄰居計算排名差異，將排名差異大的移除
        rank_diff = rank_diff_set_C{i};
        common = common(rank_diff<IR_mid);

        % if L_2 < length(common)
        %     L_2 = length(common);
        % end
        % 
        % L2_set = [L2_set, length(common)];
        % L_final = [L_final, length(common)];

        Common_X{i} = common';
    end

% high inlier rate method
else
    lambda = lambda_set(1,3);
    for i = 1:N
        % 取得 membership 與索引
        XX = neighborhoodX(:, i)';
        [maskX, ~] = ismember(XX, neighborhoodY(:, i)');
        
        % 交集的值與索引
        intersectVals = XX(maskX);  % 值
        % if L_2 < length(intersectVals)
        %     L_2 = length(intersectVals);
        % end
        % 
        % L2_set = [L2_set, length(intersectVals)];
        % L_final = [L_final, length(intersectVals)];
        Common_X{i} = intersectVals';
    end
end
%My2_C_time = toc
%tic;
WX = Construct_coefficent_matrix(Kn,N,D,X,Common_X);
WY = Construct_coefficent_matrix(Kn,N,D,Y,Common_X);
%My2_W_time = toc
index = [] ; %原為index
for i = 1:N
    if ~isnan(WX{i})
        dist = sum((WX{i} - WY{i}).^2);
        
        alpha = size(WX{i},1) / Kn ;

        if dist < lambda*alpha
            index(end+1) = i ; %原為index
        end
    end
end

% L_1
% L_2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% hold on;
% % 畫直方圖 (自訂 bin 邊界)
% edges = 0:1:10;     % 以 0,1,2,... 為邊界
% histogram(L1_set, edges);
% xlabel('$L$','Interpreter','latex');
% ylabel('Frequency');
% xticks(0:1:10);
% hold off;
% saveas(gcf, 'D:\\YC\\paper\\Fig\\ANMR_rule\\tttttt_low.png');


% figure;
% hold on;
% % 畫直方圖 (自訂 bin 邊界)
% edges = 0:1:10;     % 以 0,1,2,... 為邊界
% histogram(L_final, edges);
% xlabel('$L$','Interpreter','latex');
% ylabel('Frequency');
% xticks(0:1:10);
% hold off;
% saveas(gcf, 'D:\\YC\\paper\\Fig\\ANMR_rule\\high_L_hist.png');
