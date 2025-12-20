clear;close all;clc;
run('D:\\YC\\vlfeat-0.9.21\\toolbox\\vl_setup')
rng(42);

%% parameter setting
K=10;  %% size of neighborhoods
lambda = [0.36, 0.3, 0.18]; % 目前想要的參數
eta=[0.2, 0.5, 0.5]; %% parameters of the filter
r = 0.4 ;
IR = 3;
l_ir = 0.35;
h_ir = 0.6;

mult_low = 1;
mult_mid = 2;

for fff = 1:9
if fff == 1
    dataset = "Airport";
elseif fff == 2
    dataset = "Small_Village";
elseif fff == 3
    dataset = "University_Campus";
elseif fff == 4
    dataset = "RS";
elseif fff == 5
    dataset = "VGG";
elseif fff == 6
    dataset = "Daisy";
elseif fff == 7
    dataset = "DTU";
elseif fff == 8
    dataset = "adelaidermf";
elseif fff == 9
    dataset = "EVD";
end

if dataset == "Airport"
    data_num = [61, 116, 177, 282, 3479];

elseif dataset == "Small_Village"
    data_num = [924, 970, 1011, 1113, 1204];

elseif dataset == "University_Campus"
    data_num = [60, 98, 172, 333, 403];

elseif dataset == "RS" % 移除：62, 24
    data_num = [3, 9, 11, 12, 14, 16, 17, 18, 22, 25, 27, 28, 33, 41, 43, ...
                49, 50, 64, 65, 73, 82, 84, 86, 88, 90, 101, 102, 103, 105, ...
                108, 110, 120, 123, 124, 125, 126, 128, 129, 130, 132, 133, ...
                134, 136, 137, 139, 147, 149, 151, 155];

elseif dataset == "VGG" % 移除：5, 10
    data_num = [1, 2, 3, 4, 11, 12, 13, 16, 17, 18, 19, 20, ...
                21, 22, 23, 24, 25, 30, 33, 34, 35, 37, 38, 39, 40];

elseif dataset == "Daisy"
    data_num = [1, 3, 5, 7, 10, 14, 18, 21, 22, 24, ...
                25, 26, 29, 30, 31, 32, 33, 34, 35, 36, ...
                37, 38, 39, 40, 41, 42, 43, 50, 51, 52];

elseif dataset == "DTU"
    data_num = [1, 2 ,3, 4, 5, 6, 7, 9, 14, 15, ...
                16, 18, 20, 22, 23, 24, 25, 26, 27, 30, ...
                31, 32, 33, 34, 39, 40, 41, 45, 46, 47, ...
                48, 50, 51, 52, 56, 61, 62, 63, 64, 65, ...
                66, 70, 71, 72, 79, 80, 81, 84, 85, 86, ...
                87, 88, 89, 95, 96, 97, 98, 100, 103, 108, ...
                111, 112, 113, 114, 115, 116, 117, 119, 120, 121, ...
                122, 123, 124, 125, 126, 127, 128, 129, 130, 131];

elseif dataset == "adelaidermf" % 移除：12 13
    data_num = [2, 3, 4, 5, 8, 9, 10, 11, 14, ...
                15, 16, 17, 18, 19, 20, 22, 23, 29, 30, 36];

elseif dataset == "SUIRD" % 移除：25, 31, 82
    data_num = [1, 7, 11, 18, 19, 22, 29, 30, ...
                33, 34, 39, 40, 42, 43, 45, 46, 48, 49, ...
                50, 51, 55, 57, 66, 67, 68, 70, 71, 75, ...
                77, 78, 79, 80, 81, 84, 86];

elseif dataset == "EVD" % 移除：25, 31, 82
    data_num = [1, 2, 3, 4, 5, 6, 7, 8, ...
                9, 10, 11, 12, 13, 14, 15];

else
    disp("NO dataset!!!");
end

adjust_num = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"];

for num = 1:size(data_num, 2)
    src_num = num2str(data_num(num));

    src_num = string(sprintf('%04d', str2double(src_num)));
    dst_num = "1" + extractAfter(src_num, 1);

    if dataset == "Airport" || dataset == "Small_Village" || dataset == "University_Campus"
        dst_num = num2str(data_num(num)+1);
        dst_num = string(sprintf('%04d', str2double(dst_num)));
    end

    d_num = size(data_num, 2);
    pred_result_tmp = [];
    for i = 1:9

        %%%%%%%%%%%%%%%%%
        X_ = readmatrix("D:\\YC\\YC_dataset\\" + dataset + "\\" + src_num + "_pts.csv");
        Y_ = readmatrix("D:\\YC\\YC_dataset\\" + dataset + "\\" + dst_num + "_pts.csv");
        gt = readmatrix("D:\\YC\\YC_dataset\\" + dataset + "\\" + src_num + "_" + dst_num + ".csv", 'OutputType', 'string');
        adjust = readmatrix("D:\\YC\\YC_dataset\\" + dataset + "\\adjust\\" + src_num + "_" + dst_num + "_" + adjust_num(1, i) + ".csv");
        adjust = adjust+1;
        info_X = readmatrix("D:\\YC\\YC_dataset\\" + dataset + "\\" + src_num + "_info.csv");
        info_Y = readmatrix("D:\\YC\\YC_dataset\\" + dataset + "\\" + dst_num + "_info.csv");
        info_X(adjust, :) = []; % 刪除 X_ 中指定行
        info_Y(adjust, :) = []; % 刪除 Y_ 中指定行
        info_X = [];
        info_Y = [];
        %%%%%%%%%%%%%%%%%

        % 去除 adjust 指定的索引
        X_(adjust, :) = []; % 刪除 X_ 中指定行
        Y_(adjust, :) = []; % 刪除 Y_ 中指定行
        gt(adjust, :) = []; % 刪除 gt 中指定行
        gt = strip(replace(gt, '"', '')); % 去引號
        true_indices = find(strcmpi(gt, "true"));  % 忽略大小寫

        tic;
        [index, pred_IR] = BMRC(X_, Y_, info_X, info_Y, K, lambda, eta, r, l_ir, h_ir, mult_low, mult_mid);
        runtime = toc ;

        [precise, recall, corrRate] = evaluate(true_indices, index, size(X_,1));
        Fscore=2*precise*recall/(precise+recall) ;

        result = [precise, recall, Fscore, runtime];
        all_result = cat(1, all_result, result) ;
    end
end
end
