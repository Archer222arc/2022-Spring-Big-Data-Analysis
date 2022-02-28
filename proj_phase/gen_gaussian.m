function [A,AT,y,z] = gen_gaussian(n,m)
% Generating data for Phase Retrieval problem of the form
%
%                       \|A^*z\| = y
%
% Input:
%        --- n : vector length
%        --- m : sample size
%
% Output:
%        --- A : A=[a1,a2,...,am] 
%        --- AT : AT(z) = A^* z
%        --- y : |A^* z|^2
%        --- z : true data 
%


z = randn(n,1)+1i*randn(n,1);                   
Am = 1/sqrt(2)*randn(n,m)+1i/sqrt(2)*randn(n,m);
y = abs(Am'*z).^2 ;
A =  Am;
AT = @(z) Am'*z;

end
