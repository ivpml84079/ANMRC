clear;close all;clc;
run('.\\vlfeat-0.9.21\\toolbox\\vl_setup')
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

%% input
src_num = "0282";
dst_num = "0283";
adjust_num = "0.70"; % 0.10 ~ 0.90

X = readmatrix(".\\data\\" + src_num + "_pts.csv");
Y = readmatrix(".\\data\\" + dst_num + "_pts.csv");
info_X = readmatrix(".\\data\\" + src_num + "_info.csv");
info_Y = readmatrix(".\\data\\" + dst_num + "_info.csv");
gt = readmatrix(".\\data\\" + src_num + "_" + dst_num + ".csv", 'OutputType', 'string');
adjust = readmatrix(".\\data\\adjust\\" + src_num + "_" + dst_num + "_" + adjust_num + ".csv");
adjust = adjust+1;


% 去除 adjust 指定的索引
X(adjust, :) = []; 
Y(adjust, :) = []; 
info_X(adjust, :) = []; 
info_Y(adjust, :) = []; 
gt(adjust, :) = []; 
gt = strip(replace(gt, '"', '')); % 去引號
true_indices = find(strcmpi(gt, "true"));  % 忽略大小寫


%% ANMRC
tic;
[index, pred_IR] = ANMRC(X, Y, info_X, info_Y, K, lambda, eta, r, l_ir, h_ir, mult_low, mult_mid);
runtime = toc 

%% plot and accuracy
I1 = imread(".\\data\\IMG_" + src_num + ".JPG");
I2 = imread(".\\data\\IMG_" + dst_num + ".JPG");
I1 = imresize(I1, [720, 1080]);
I2 = imresize(I2, [720, 1080]);
plot_both_row(I1, I2, X, Y, index, true_indices);
