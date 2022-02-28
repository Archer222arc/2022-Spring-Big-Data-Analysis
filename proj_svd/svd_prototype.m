function [U, V, d]=svd_prototype(A,k,q,opts)
% Compute partial SVD decomposition for a given matrix A of its k-th largest 
% eigenvalues with a random algorithm Prototype randomized SVD
% 
%          A = U_k * Sigma_k * V_k'
% 
% Input:
%        --- A : mxn matrix input
%        --- k : Number of singular values to be computed
%        --- q : Power of preprocessing
%        --- opts : Options structure with field(s)
%               --- p: additional sampling, k (default)
% Output:
%       --- U,V : approximate left and right singular vectors of A
%       --- d : approximate singular values of A

%% initialize
if(~isfield(opts,'p'))            opts.p = k; end
p = opts.p;   
[~,n]  = size(A);
omega = randn(n,k+p);
Y = A*omega;

%% main

for i = 1:q
    if i == 1;    AA = (A*A');  end
    Y = AA*Y;
end

[Q,~] = qr(Y);
Q = Q(:,1:k+p); 

B = Q'*A;  
[Ub,S,V] = svds(B,k); 
U = Q*Ub;
d = diag(S);


end
