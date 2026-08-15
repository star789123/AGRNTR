function [H, W] = constructHW1_revised(fea, p, method)
% 稳定版超图构造：无NaN、无Inf、标准超图定义  超边 = 一个节点 + 它的 p 个邻居
% H: 节点数 × 超边数 (n×n，每个节点一个超边)
% W: 对角超边权重矩阵

[nSmp, d] = size(fea);  % nSmp=样本数，d=特征维度

%% 1. KNN 搜索（找每个点的最近邻）
% 找每个样本自己 + 最近的 p 个邻居
[idxNN, distNN] = knnsearch(fea, fea, 'K', p+1);

idxNN = idxNN(:, 2:end);   % 去掉自己，只保留 p 个邻居的索引
distNN = distNN(:, 2:end); % 去掉自己，只保留 p 个邻居的距离

%% 2. 根据不同 method 计算节点之间的相似度 vals
% vals 大小：nSmp × p，每一行是当前点与 p 个邻居的相似度
switch lower(method)
    case '01'        % 0-1 相似度：相连就是1，不连就是0
        vals = ones(nSmp, p);

    case 'heat'      % 热核相似度：距离越近，相似度越接近1
        sigma = median(distNN(:)) + eps;  % 用所有距离的中位数做高斯核宽度
        vals = exp(-distNN.^2 / sigma^2); % 高斯公式计算相似度

    case 'dot'       % 归一化点积相似度（余弦相似度）
        vals = zeros(nSmp, p);
        for i = 1:nSmp
            vec = fea(i,:); vec = vec / (norm(vec)+eps); % 自身单位化
            nbrs = fea(idxNN(i,:), :);                   % 邻居特征
            nbrs = nbrs ./ (vecnorm(nbrs,2,2)+eps);      % 邻居单位化
            vals(i,:) = vec * nbrs';                    % 点积 = 余弦相似度
        end
        vals = (vals + 1) / 2;  % 把 [-1,1] 映射到 [0,1]

    case 'hik'       % 直方图相交核（适合非负特征，如图像）
        vals = zeros(nSmp, p);
        for i = 1:nSmp
            % 逐维度取最小值，然后求和 = 相交相似度
            vals(i,:) = sum(min(fea(i,:), fea(idxNN(i,:), :)), 2)';
        end
        vals = vals / (max(vals(:)) + eps); % 归一化到 [0,1]

    otherwise
        error('Unknown method'); % 不认识的方法直接报错
end

%% 3. 构造超图关联矩阵 H（最关键！）H 的列 = 超边,H 的行 = 节点,H (i,j) = 节点 i 属于超边 j 的强度,总超边数 = 总样本数 n
% 给每个节点创建一条超边：包含自己 + p 个邻居
rowIdx = repmat((1:nSmp)', 1, p+1);  % 行号：每个节点重复 p+1 次  哪些节点属于这条超边
colIdx = [ (1:nSmp)', idxNN ];       % 列号：自己 + 邻居编号  .这条超边包含谁
val    = [ones(nSmp,1), vals];       % 值：自己=1，邻居=计算出的相似度  邻居的关联强度 = 刚才算的相似度
%一条超边 = 以某个节点为中心，包含它自己和它最近的 p 个邻居
% 用上面的行、列、值，构造稀疏矩阵 H（nSmp × nSmp）
H = sparse(rowIdx(:), colIdx(:), val(:), nSmp, nSmp);

%% 4. 构造超边权重对角矩阵 W
sumH = sum(H, 1);                % 每条超边（每列）的数值总和
cntH = sum(H > 0, 1) + eps;      % 每条超边里有多少个非零元素（+eps防除0）
Wvals = sumH ./ cntH;            % 超边权重 = 总和 / 元素个数（平均权重）

Wvals = full(Wvals)';            % 转成列向量
Wvals(Wvals < 0) = 0;            % 保证权重非负

% 构造稀疏对角矩阵 W
W = spdiags(Wvals, 0, nSmp, nSmp);


