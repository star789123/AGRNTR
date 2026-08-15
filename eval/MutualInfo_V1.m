% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% vvvvvvvvvv 【修正后的】第一份代码中的 NMI 计算方法 vvvvvvvvvvvvvvvv
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function MIhat = MutualInfo_V1(L1, L2)
% MutualInfo: 计算两组标签的归一化互信息 (版本1：基于算术平均)
% MIhat = MI / mean(H1, H2) 或者 MI / max(H1, H2)
% 这是在很多早期论文中常见的版本。

% --- 预处理标签 ---
L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end
% 将标签转换为从1开始的连续整数
L1 = L1 - min(L1) + 1;
L2 = L2 - min(L2) + 1;

% --- 构建混淆矩阵 (Contingency Table) ---
nClass = max(max(L1), max(L2));
G = zeros(nClass);
for i = 1:nClass
    for j = 1:nClass
        G(i,j) = length(find(L1 == i & L2 == j));
    end
end
sumG = sum(G(:));

% --- 计算熵和互信息 ---
P12 = G / sumG;                  % 联合概率分布 (nClass x nClass)
P1 = sum(P12, 2);                % L1 的边缘概率分布 (nClass x 1)
P2 = sum(P12, 1);                % L2 的边缘概率分布 (1 x nClass)

H1 = -sum(P1 .* log2(P1 + eps));  % L1 的熵
H2 = -sum(P2 .* log2(P2 + eps));  % L2 的熵

% --- 【核心修正点】 ---
% 之前错误的行:
% MI = sum(P12(:) .* log2(P12(:) ./ (P1(:) * P2) + eps));

% 修正后的行:
% 我们需要计算 P(i,j) / (P(i)*P(j))。P(i)*P(j) 是一个外积。
Denominator = P1 * P2; % (nClass x 1) * (1 x nClass) -> (nClass x nClass)
Ratio = P12 ./ (Denominator + eps);
MI = sum(P12 .* log2(Ratio + eps), 'all'); % 使用 'all' 直接对矩阵所有元素求和

% --- 归一化 (使用算术平均) ---
MIhat = MI / ((H1+H2)/2);

% 确保值在 [0,1] 区间
MIhat = max(0, min(1, real(MIhat)));
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% ^^^^^^^^^^^ 【修正后的】第一份代码中的 NMI 计算方法 ^^^^^^^^^^^^^^^^^
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %