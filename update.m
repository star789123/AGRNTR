function [F, A, L,evs] = update(X, c, k, r,lambda, beta,islocal)
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% k: number of neighbors
% r: regularization parameter
% islocal: 1=仅更新k近邻; 0=更新所有点
% y: 聚类标签
% A: 学到的相似度矩阵
% evs: 图拉普拉斯矩阵的特征值（每次迭代）

addpath(genpath('\funs'))

NITER = 30;
num = size(X,2);

if nargin < 6
    islocal = 1;
end
if nargin < 4
    r = -1;
end
if nargin < 3
    k = 15;
end

%% Step 1: 计算原始空间距离矩阵
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);

%% Step 2: 计算 r 的默认值
A = zeros(num);  
for i = 1:num
    di = distX1(i,2:k+2); 
    id = idx(i,2:k+2); 
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end

A = (A + A') / 2;   % 确保 A 对称

%% Step 5: 初始化 F（特征嵌入）
D0 = diag(sum(A));
L0 = D0 - A;
[F, ~, evs] = eig1(L0, c, 0);



%% Step 6: 迭代更新 A 和 F
for iter = 1:NITER
    distf = L2_distance_1(F',F');
    A = zeros(num);

    for i = 1:num
        if islocal == 1
            idxa0 = idx(i,2:k+1); % 只更新初始k近邻
        else
            idxa0 = 1:num;        % 全部点
        end

        dfi = distf(i,idxa0);
        dxi = distX(i,idxa0);
        ad = -(0.25*beta*dxi + lambda * dfi) / (2 * r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end

    % 对称化
    A = (A + A') / 2;
    D = diag(sum(A));
    L = D - A;

    % 更新特征嵌入
    [F, ~, ev] = eig1(L, c, 0);
    evs(:,iter+1) = ev;
end

end
