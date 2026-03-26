function [W, A, Z,ns, loss] = Init_COE_MVC(X, param)



nv = length(X);
ns = size(X,2);  %当前视图的样本数
k = param.k;
lambda = param.lambda;
d = 20;

    X = NormalizeFea(X, 0);




    [U, ~, V] = svd(rand(d, size(X,1)), 'econ');
    W = U*V';
    clear U V


A = zeros(d, k);
Z = zeros(k, ns);
MAX_Iter =  20;
loss = 0;


for iter = 1:MAX_Iter

   

    % update Z
   Z = (A'*W*X)/(1+lambda);
    

        % update A
 
    tempA = W*X*Z';
    [U, ~, V] = svd(tempA, 'econ');
    A = U*V';
    clear U V
    
    % update W
 
        [U, ~, V] = svd(A*Z*X', 'econ');
        W = U*V';
        clear U V

    % Calculate Loss 
        
       loss_recon = norm(X - W'*A*Z, 'fro')^2;
        loss_regZ = lambda * norm(Z, 'fro')^2;
        
        loss(iter) = loss_recon + loss_regZ;

 
   



end




end



