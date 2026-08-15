function Z_neq_out = Z_neq(Z, n)
   
    N = size(Z,1);  
    Z = circshift(Z, -n);  

    r_cur  = size(Z{1},1);       % r_n
    r_next = size(Z{N-1},3);     % r_{n+1}

    
    P = Z{1};  

    for i = 2:N-1
        
        szP = size(P);
        zl = reshape(P, szP(1)*szP(2), szP(3));  

        
        szZ = size(Z{i});
        zr = reshape(Z{i}, szZ(1), szZ(2)*szZ(3)); 

        
        P = zl * zr;  % [r_cur * prod(I_prev) × (I_i * r_{i+1})]

        
        new_dim2 = szP(2) * szZ(2);  
        P = reshape(P, szP(1), new_dim2, szZ(3));  % [r_cur × new_dim2 × r_{i+1}]
    end

   
    Z_neq_out = P;
end
