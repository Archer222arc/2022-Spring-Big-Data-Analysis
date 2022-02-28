function [z,out] = Wirtinger_flow(A,AT,y,x,opts)
% Apply the gradient descent with initialization to phase retrieval problem
%                       \|A^* x\| = y
% Input:
%        --- A : A(x) = Ax
%        --- AT : AT(x) = A^Tx
%        --- y : |A^* x|^2
%        --- x : True data
%        --- opts : Options structure with field(s)
%              --- maxiter: maximal iteration
%              --- preiter: iteration of oreprocessing procedure
%              --- regu: a regularization term
%              --- tau0: a parameter controling the stepsize
%              --- maxstep: maximal stepsize
%
% Output:
%        --- z : Approximate solution of x
%        --- out : Other output
%
%% initialization
if ~isfield(opts,'maxiter');         opts.maxiter = 3e3; end
if ~isfield(opts,'preiter');         opts.preiter = 200; end
if size(A) ~= 1
if ~isfield(opts,'lambda');          opts.lambda = min(sqrt(sum(y)/numel(y)),sqrt(length(x)*sum(y)/norm(A,'fro'))); end
else
if ~isfield(opts,'lambda');          opts.lambda = sqrt(sum(y(:))/numel(y)); end 
end
if ~isfield(opts,'tau0');            opts.tau0 = 330; end
if ~isfield(opts,'maxstep');         opts.maxstep = 0.3; end

maxiter = opts.maxiter;    preiter = opts.preiter;     lambda = opts.lambda;
tau0 = opts.tau0;          maxstep = opts.maxstep;

tic;
[n,~] = size(x);
[m,k] = size(y);
if(k~=1)
    m = 1;
end

if size(A) == 1
%% preprocess z
z = randn(n,1);
z = z/norm(z,2);
for i=1:preiter
     z = A(y.*AT(z));
     z = z/norm(z,2);
end 
z = lambda*z;               


%% main loop

err_list=  zeros(maxiter,1);
for i=1:maxiter
    err_list(i) = DIST(x,z);
    Az = AT(z);
    g  = A((abs(Az).^2-y).*Az)/m; 
    stept = min(maxstep,1-exp(-i/tau0));
    z = z - stept/lambda^2 * g;        
%     max(abs(z))
end


out.time = toc;
out.err = err_list;
else
%% preprocess z
z = randn(n,1);
z = z/norm(z,2);
for i=1:preiter
     z = A*(y.*AT(z));
     z = z/norm(z,2);
end 
z = lambda*z;               


%% main loop

err_list=  zeros(maxiter,1);
for i=1:maxiter
    err_list(i) = DIST(x,z);
    Az = AT(z);
    g  = A*((abs(Az).^2-y).*Az)/m; 
    stept = min(maxstep,1-exp(-i/tau0));
    z = z - stept/lambda^2 * g;        
end


out.time = toc;
out.err = err_list;

end

function d = DIST(x,z)
nx = norm(x,2);
d = norm(x-exp(-1i*angle(x'*z))*z,2)/nx;
end
end

