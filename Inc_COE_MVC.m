function [W, A, Z,ns,Z_old, loss] = Inc_COE_MVC(X, W_old, A_old,Z_old,n_old, param)


lambda = param.lambda;
mu = param.mu;
gamma = param.gamma;
k = param.k;
d = 20;
loss = 0;


ns = size(X,2);  % 当前视图的样本数

   X = NormalizeFea(X, 0);




   % if iv > nv_o
        [U, ~, V] = svd(rand(d, size(X,1)), 'econ');
        W = U*V';
        clear U V

clear F

Z = zeros(k, ns);

MAX_Iter = 20;

A = zeros(d, k);
for iter = 1:MAX_Iter
    
    % update Z
    if ns>=n_old
        temp = A'*W*X;
    awx1 = temp(:, 1:n_old);      
    awx2 = temp(:, n_old+1:end);   

    Z1 = (awx1+mu*Z_old)/(1+lambda+mu);
    Z2 = (awx2)/(1+lambda);
    Z = cat(2, Z1, Z2);
    else
     awx1 = Z_old(:, 1:ns);
     awx2 = Z_old(:, ns+1:end);
      
    Z = (A'*W*X+mu*awx1)/(1+lambda+mu);
    Z_old = cat(2, Z, awx2);
    end
   
 % update W
 
        [U, ~, V] = svd(A*Z*X', 'econ');
        W = U*V';
        clear U V


    % update A

     tempA = W*X*Z'+gamma*A_old;
    [U, ~, V] = svd(tempA, 'econ');
    A = U*V';
    clear U V


    %  Calculate Loss 
        
        loss_recon = norm(X - W'*A*Z, 'fro')^2; 
        loss_regZ = lambda * norm(Z, 'fro')^2;
       
        if ns >= n_old
            loss_mu = mu * norm(Z(:, 1:n_old) - Z_old, 'fro')^2;
        else
           
            loss_mu = mu * norm(Z - awx1, 'fro')^2; 
        end
        
       
        loss_gamma = gamma * norm(A - A_old, 'fro')^2;
        
        
        loss(iter) = loss_recon + loss_regZ + loss_mu + loss_gamma; 













end

end



