clear;
addpath(genpath(pwd));

% load data
load('E:\Y\Single view\data\warpAR10P.mat');

rand('twister', 5489);
c = length(unique(gnd));  
islocal = 1;

%% 参数设置
repeatTimes = 10;    % 每组参数重复实验次数
r1 = 3;
r2 = 5;
% r3=4;
t = zeros(1,8);
acc_array = zeros(1,repeatTimes);
nmi_array = zeros(1,repeatTimes);
AC_s = zeros(1,8);
NMI_s = zeros(1,8);
AC = zeros(1,8);
NMI = zeros(1,8);

lambda_list = [1e-3, 1e-2, 1e-1,1e1,1e2,1e3];
alpha_list = [1e-3,2e-3,3e-3,4e-3,5e-3,6e-3];
gamma_list = [1e-2, 1e-1,1e1,1e2,1e3];

results = [];

%% 聚类循环
for kk = 4
    k = 5 * kk;
    % R = [5, r1, r2,r3];
    R = [2, r1, r2];
    fea_size = size(fea);
    
    for lambda = lambda_list
        for alpha = alpha_list
            for gamma = gamma_list
            
            fprintf('\n== Running: lambda = %.1e, alpha = %.1e,gamma = %.1e ==\n', lambda, alpha,gamma);
            acc_array = zeros(1, repeatTimes);
            nmi_array = zeros(1, repeatTimes);

            for i = 1:repeatTimes
               
                tic;
                [node,~]=AGRNTR(fea, R, k,c,islocal,lambda, alpha,gamma);
                TimeCost(i)=toc;

                % ========= 聚类 =========
                A = node{end};
                A = permute(A, [1,3,2]);
                A = reshape(A, [R(end)*R(1), fea_size(end)]);

                clusterResults = evalResults(A, gnd);

              
                acc_array(i) = clusterResults(1);
                nmi_array(i) = clusterResults(2);
            end

            % ========= 记录平均结果 =========
            av_acc = mean(acc_array);
            AChat_s = std(acc_array) * 100;
            av_nmi = mean(nmi_array);
            MIhat_s = std(nmi_array) * 100;
            AvgTime=mean(TimeCost);
            StdTime=std(TimeCost);
           
            results = [results;
                lambda, alpha, gamma, av_acc*100, AChat_s, av_nmi*100, MIhat_s, AvgTime, StdTime];
            
            fprintf('lambda = %10.2e | alpha = %10.2e | gamma = %10.2e| ACC = %6.2f%% ± %5.2f | NMI = %6.2f%% ± %5.2f\n', ...
                lambda, alpha, gamma, av_acc*100, AChat_s, av_nmi*100, MIhat_s);
            fprintf('Average running time: %.4f seconds (Std: %.4f)\n', AvgTime, StdTime);

            end
        end
    end
end

%% 输出结果表格
T = array2table(results, ...
    'VariableNames', {'Lambda','Alpha','Gamma','ACC_mean','ACC_std','NMI_mean','NMI_std','AvgTime','StdTime'});

disp('===== 参数搜索结果 =====');
disp(T);




