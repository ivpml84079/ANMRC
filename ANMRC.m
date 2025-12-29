function [index, pred_IR] = ANMRC(X, Y, X_info, Y_info, Kn, lambda_set, eta, r, l_ir, h_ir, mult_low, mult_mid)

Xt = X';
Yt = Y';
[N,D] = size(X);

% predict inlier rate
X_scale = X_info(:,1);
Y_scale = Y_info(:,1);
scale_ratio = max(X_scale, Y_scale) ./ min(X_scale, Y_scale);

X_deg = X_info(:,2);
Y_deg = Y_info(:,2);
rad_diff = abs(wrapToPi(deg2rad(X_deg - Y_deg)));

pred_IR = predict_IR(rad_diff, scale_ratio) ;


% IFCNR-based filter
idx = IFCNR(Xt,Yt,Kn,eta);


%% iteration 1
kdtreeX = vl_kdtreebuild(Xt(:,idx));
[neighborhoodX,~] = vl_kdtreequery(kdtreeX, Xt(:,idx), Xt, 'NumNeighbors', Kn+1) ;
neighborhoodX = idx(neighborhoodX);
neighborhoodX = neighborhoodX(2:Kn+1,:);

kdtreeY = vl_kdtreebuild(Yt(:,idx));
[neighborhoodY,~] = vl_kdtreequery(kdtreeY, Yt(:,idx), Yt, 'NumNeighbors', Kn+1) ;
neighborhoodY = idx(neighborhoodY);
neighborhoodY = neighborhoodY(2:Kn+1,:);

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
        not_common_rank_diff = cal_DRD(Y, i, Kn, diffVals_XY, idxX_diff);

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
        Common_X{i} = intersectVals';
    end
end

W_X = construct_W(Kn,N,D,X,Common_X);
W_Y = construct_W(Kn,N,D,Y,Common_X);

idx = [] ;
for i = 1:N
    if ~isnan(W_X{i})
        dist = sum((W_X{i} - W_Y{i}).^2);
        
        alpha = size(W_X{i},1) / Kn ;

        if dist < lambda*alpha
            idx(end+1) = i ; 
        end
    end
end


%% iteration 2 
kdtreeX = vl_kdtreebuild(Xt(:,idx));
[neighborhoodX,~] = vl_kdtreequery(kdtreeX, Xt(:,idx), Xt, 'NumNeighbors', Kn+1) ;
neighborhoodX = idx(neighborhoodX);
neighborhoodX = neighborhoodX(2:Kn+1,:);

kdtreeY = vl_kdtreebuild(Yt(:,idx));
[neighborhoodY,~] = vl_kdtreequery(kdtreeY, Yt(:,idx), Yt, 'NumNeighbors', Kn+1) ;
neighborhoodY = idx(neighborhoodY);
neighborhoodY = neighborhoodY(2:Kn+1,:);

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
        not_common_rank_diff = cal_DRD(Y, i, Kn, diffVals_XY, idxX_diff);

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
        Common_X{i} = intersectVals';
    end
end

W_X = construct_W(Kn,N,D,X,Common_X);
W_Y = construct_W(Kn,N,D,Y,Common_X);

index = [] ; 
for i = 1:N
    if ~isnan(W_X{i})
        dist = sum((W_X{i} - W_Y{i}).^2);
        
        alpha = size(W_X{i},1) / Kn ;

        if dist < lambda*alpha
            index(end+1) = i ; 
        end
    end
end

