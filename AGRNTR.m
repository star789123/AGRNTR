function [node,S]=AGRNTR(Y,r,k,c,islocal,lambda, alpha,gamma)
MaxIter = 100;
n = size(Y);
n = n(:);%size or original tensor
d = numel(n);%order
num = size(Y,d);
node=cell(1,d);
r = r(:);
Tol = 1e-4;
islocal=1;


for i=1:d-1
    node{i}=rand(r(i),n(i),r(i+1));
end
node{d}=rand(r(d),n(d),r(1));

od=[1:d]';
err=1;
node = node';
sum_err = zeros(1, MaxIter);
trace_loss    = zeros(1, MaxIter);
total_loss    = zeros(1, MaxIter);
S_reg_loss    = zeros(1, MaxIter);

  X = reshape(Y, [numel(Y)/n(end), n(end)])';
  distX = L2_distance_1(X',X'); 
  [distX1, idx] = sort(distX,2); 
  S = zeros(num);
  for i = 1:num
      di = distX1(i,2:k+2); 
      ranking = idx(i,2:k+2);
      S(i,ranking) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps); 
  infind=find(S>10); 
  S(infind)=1;
  
%%
 S = (S+S')/2; 
 D = diag(full(sum(S, 2))); 
 L = D - S; 

  

for i = 1:MaxIter*d                                                                                                                                                                                                                                                                                  
    err0=err;
    if i>1
        Y=shiftdim(Y,1);
        od=circshift(od,-1);
    end
    Y = reshape(Y,[n(od(1)),numel(Y)/n(od(1))]); %mode-n matricization
    A = node{od(1)}; %rk ik rk+8
    A = permute(A,[2,3,1]); %ik rk rk+1
    A = reshape(A,[n(od(1)),r(od(1))*r(od(2))]);  % mode-2 matricization of Z(k), Ik* RkRk+1
    B = Z_neq(node,od(1)); %rk i1...ik-1*ik+1...in rk+1
    B = permute(B,[1,3,2]);
    B = reshape(B,[r(od(2))*r(od(1)),prod(n(od(2:end)))]); %mode-2 Matricization of Z(~=k)', RkRk+1* I1...In-1In+1...IN
     t_inner = 20;
            YBT = Y*B';
            BBT = B*B';
            if mod(i, d) ~= 0
                for jj = 1:t_inner
                    A = A.*((YBT) ./ max(eps, (A*BBT)));
                end
            else
                for jj = 1:t_inner
                    A = A.*((YBT+alpha*S*A) ./ max(eps, (A*BBT+alpha*D*A)));
                end
            end
           
      
    A_updated = reshape(A,[n(od(1)),r(od(2)),r(od(1))]);
    A_updated = permute(A_updated,[3,1,2]);
    node{od(1)} = A_updated;
    
    % %
    if mod(i, d) == 0
        complete_iter = i/d;
        
        
        err1 = norm(Y - A * B, 'fro')^2;
        sum_err(complete_iter) = err1;
        
        
        G = node{d};
        G_mode2 = reshape(permute(G, [2,3,1]), [n(d), r(1) * r(d)]);
        [F, S, L,~] = update(G_mode2', c, k, gamma, lambda, alpha, islocal);
         
        
        S_reg_loss(complete_iter) = gamma * norm(S, 'fro')^2;
        trace_loss(complete_iter) = 2 * lambda * trace(F' * L * F);
        total_loss(complete_iter) = 0.5*sum_err(complete_iter) + trace_loss(complete_iter) + S_reg_loss(complete_iter);
        
        if complete_iter > 1 && abs(sum_err(complete_iter-1)-sum_err(complete_iter)) <= Tol
            break;
        end
    end
    
    Y = reshape(Y,n(od)');
end

return
end

