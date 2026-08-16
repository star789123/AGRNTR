function [F, A, L,evs] = update(X, c, k, r,lambda, beta,islocal)
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% k: number of neighbors
% r: regularization parameter


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

%% 
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);

%% 
A = zeros(num);  
for i = 1:num
    di = distX1(i,2:k+2); 
    id = idx(i,2:k+2); 
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end

A = (A + A') / 2;  

%% 
D0 = diag(sum(A));
L0 = D0 - A;
[F, ~, evs] = eig1(L0, c, 0);



%% 
for iter = 1:NITER
    distf = L2_distance_1(F',F');
    A = zeros(num);

    for i = 1:num
        if islocal == 1
            idxa0 = idx(i,2:k+1); % 只更新初始k近邻
        else
            idxa0 = 1:num;        
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
