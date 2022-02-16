function [A,u,b] = l1_gen(m,n,sparsity)

A = randn(m,n);
u = sprandn(n,1,sparsity);
b = A*u;

end
