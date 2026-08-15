function MIhat = MutualInfo_V1(L1, L2)

L2 = L2(:);
if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end

L1 = L1 - min(L1) + 1;
L2 = L2 - min(L2) + 1;


nClass = max(max(L1), max(L2));
G = zeros(nClass);
for i = 1:nClass
    for j = 1:nClass
        G(i,j) = length(find(L1 == i & L2 == j));
    end
end
sumG = sum(G(:));


P12 = G / sumG;                  
P1 = sum(P12, 2);               
P2 = sum(P12, 1);               

H1 = -sum(P1 .* log2(P1 + eps)); 
H2 = -sum(P2 .* log2(P2 + eps));  


Denominator = P1 * P2; 
Ratio = P12 ./ (Denominator + eps);
MI = sum(P12 .* log2(Ratio + eps), 'all'); 


MIhat = MI / ((H1+H2)/2);


MIhat = max(0, min(1, real(MIhat)));
end
