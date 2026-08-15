function Z_neq_out = Z_neq(Z, n)
    % Z: 核张量cell，Z{k} = [r_k × I_k × r_k+1]
    % n: 当前更新mode
    % 返回: Z_neq_out = [r_n × prod(I_neq) × r_{n+1}]

    N = size(Z,1);  % 模式数
    Z = circshift(Z, -n);  % 把第n个mode放最后

    r_cur  = size(Z{1},1);       % r_n
    r_next = size(Z{N-1},3);     % r_{n+1}

    % 初始化 P
    P = Z{1};  % [r_cur × I_1 × r_2]

    for i = 2:N-1
        % 左侧乘积的大小：[r_cur × prod(I_1:I_{i-1}) × r_i]
        szP = size(P);
        zl = reshape(P, szP(1)*szP(2), szP(3));  % [r_cur * prod(I_prev) × r_i]

        % 右侧乘积的大小：[r_i × I_i × r_{i+1}]
        szZ = size(Z{i});
        zr = reshape(Z{i}, szZ(1), szZ(2)*szZ(3));  % [r_i × (I_i * r_{i+1})]

        % 矩阵乘积
        P = zl * zr;  % [r_cur * prod(I_prev) × (I_i * r_{i+1})]

        % 更新 P 尺寸
        new_dim2 = szP(2) * szZ(2);  % 更新 prod(I_prev) × I_i
        P = reshape(P, szP(1), new_dim2, szZ(3));  % [r_cur × new_dim2 × r_{i+1}]
    end

    % 最后P大小：[r_cur × prod(I_neq) × r_next]
    Z_neq_out = P;
end
