function [A, U, V, d]= svd_gen(m, n, p, num)
% Generate the datasets for directly test
%
% Input:
%     --- [m,n] : the size of A to be decomposed
%     --- p : the rank of matrix A for dataset 1
%             t = 1~5 specifies the distribution of singular values of A
%             for dataset 2
%
% Output:
%     --- A : mxn matrix
%     --- U,V : left and right singular vectors of A
%     --- d : true singular values of matrix A

if num == 1
    A = randn(m,p)*randn(p,n);
    [U,S,V] = svd(A);
    d = diag(S);
else
    t = p;
    p = min(m,n);
    L = randn(m, m);
    [U, ~] = qr(L);
    L = randn(n, n);
    [V, ~] = qr(L);
    d = zeros(p,1);
    switch t
        case 1
            d(1:20)= 10.^(-4/19*(0:19)');
            d(21:p)= 1e-4./((1:(p-20))'.^0.1);
        case 2
            d= (1:p)'.^(-2);
        case 3
            d= (1:p)'.^(-3);
        case 4
            d= exp(-(1:p)'/7);
        case 5
            d= 10.^(-0.1*(1:p)');
        case 6
            d(1:3)= 1;
            d(4:6)= 0.67;
            d(7:9)= 0.34;
            d(10:12)= 0.01;
            d(13:p)= 1e-2*(p-13:-1:0)'./(p-13);
        otherwise
            disp('invalid distribution p');
    end
    S= spdiags(d, 0, m, n);
    A = U * S * V';
end
end
