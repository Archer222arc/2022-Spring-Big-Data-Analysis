function [A,AT,y,x] = gen_CDP(n,L)
% Generating data for Phase Retrieval problem of the form
%
%                       \|A^*z\| = y
%
% Input:
%        --- n : vector length
%        --- L : number of masks
%
% Output:
%        --- A : A=[a1,a2,...,am] 
%        --- AT : AT(z) = A^* z
%        --- y : |A^* z|^2
%        --- z : true data 

x = randn(n,1)+1i*randn(n,1);                
mask = randsrc(n,L,[1i -1i 1 -1]);
tmp = rand(size(mask));
mask = mask.*((tmp <= 0.2)*sqrt(3)+(tmp > 0.2)/sqrt(2));
AT = @(x) fft(conj(mask).*repmat(x,[1 L]));  
A =  @(x) mean(mask.*ifft(x),2);         

y = abs(AT(x)).^2;  
end
