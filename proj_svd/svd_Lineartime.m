function [Uout, Vout, d]=svd_Lineartime(A,k,c,opts)
% Compute partial SVD decomposition for a given matrix A of its k-th largest 
% eigenvalues with a random algorithm Linear Time SVD
% 
%          A = U_k * Sigma_k * V_k'
% 
% Input:
%        --- A : mxn matrix input
%        --- k : Number of singular values to be computed
%        --- c : Number of samples
%        --- opts : Options 
%              --- p: importantce sampling probability, if empty, use 2-norm
%              --- post: 1 for postprocessing, 0 for not
% 
% Output:
%       --- U,V : approximate left and right singular vectors of A
%       --- d : approximate singular values of A
%% initialize
[m,n] = size(A);
flag = 0;
% change the matrix A to a high matrix
if m < n
    A = A';
    flag = 1;
    [m,n] = size(A);
end
if k > n 
    k = n;
end

if ~isfield(opts,'post');         opts.post = 1; end
if(~isfield(opts,'p'))
    opts.p = zeros(n,1);
    for i=1:n
        opts.p(i) = norm(A(:,i))^2;
    end
    opts.p = opts.p/sum(opts.p)';
end

p = opts.p;    post = opts.post;
C = zeros(m,c);
%% main

index = randsample(1:n,c,true,p');
C = A(:,index)./repmat(sqrt(p(index)*c)',m,1);

if(post)
    [Uc,~,~] = svds(C,k);
    Y = (Uc'*A)';
    [V,R] = qr(Y);
    [Ur,Sr,UrVr] = svd(R');
    U = Uc*Ur;
    V = V*UrVr;
    d = diag(Sr);
    V = V(:,1:k);
else
    [U,Sc,V] = svds(C,k);
    d = diag(Sc);
end

if flag == 1
    Uout = V;
    Vout = U;
else
    Uout = U;
    Vout = V;
end

end
