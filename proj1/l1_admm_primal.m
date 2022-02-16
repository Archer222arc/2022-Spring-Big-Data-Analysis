function [x, out] = l1_admm_primal(x0,A,b,mu,opts) 
%--------------------------------------------------------------------------
% Solving L1 minimization problem directly via ADMM.
%
% The program aims to solve the L1 minimization problem of the form
%
%         min_x \mu ||x||_1 + ||Ax-b||_1
%
% An ADMM method is performed on the primal problem
%
%         min_x \mu ||x||_1 + ||y||_1
%         s.t. Ax-b=y
%
% Input:
%        x0 --- The initial point
%         A --- The matrix apperaed in optimization problem
%         b --- The vector appeared in optimization problem
%        mu --- L1 regularization term
%      opts --- Options structure with field(s)
%               tol: stop criterion
%               t: scale of Lagrange multiplier
%               step: gradient step in linearized procedure
%               maxiter: maximal iteration
%               gamma: relaxation parameter
%               print: whether print information
%               itprint: print frequency
%
% Output:
%         x --- The optimal point founded by algorithm
%       out --- Miscellaneous information during the computation
%
%--------------------------------------------------------------------------
% setup options for the solver

if ~isfield(opts,'tol');              opts.tol = 1e-12; end
if ~isfield(opts,'t');                opts.t = 1e-3; end
if ~isfield(opts,'gamma');            opts.gamma = 1.618; end
if ~isfield(opts,'step');             opts.step = 1/opts.t/svds(A,1)^2; end
if ~isfield(opts,'maxiter');          opts.maxiter = 5e4; end
if ~isfield(opts,'print');            opts.print = 1; end
if ~isfield(opts,'itprint');          opts.itprint = 50; end
if ~isfield(opts,'verbose');          opts.verbose = 1; end
%--------------------------------------------------------------------------
% copy value to parameters

tol = opts.tol;            t = opts.t;                gamma = opts.gamma;
maxiter = opts.maxiter;    itprint = opts.itprint;    pri = opts.print;
alpha = opts.step;
verbose = opts.verbose;

%--------------------------------------------------------------------------
%  setup print format

stra1 = ['%5s','%14s','%15s','%15s','\n'];
str_head = sprintf(stra1, ...
    'iter','obj','violation','|x-x_k|_2');
str_num = '%4d    %+5.4e    %+5.4e    %+5.4e\n';

%--------------------------------------------------------------------------
% initial preparation

[m,n] = size(A);
iter = 0;
xone = ones(n,1);
lambda = zeros(m,1);
x = x0;
x_old = x;
Ax = A*x;
y = Ax-b;
cons = Ax-b-y;
y_old = y;
res = inf;
tic;

%--------------------------------------------------------------------------
% start computation

if(pri)
    fprintf('ADMM solver started \n');
end
while(res>tol&&iter<maxiter)
    iter = iter+1;
    
    %----------------------------------------------------------------------
    % solve subproblem and update multiplier
    
    xtmp = x-alpha*(A'*(lambda+t*cons));
    x = max(xtmp-alpha*mu*xone,0)+min(xtmp+alpha*mu*xone,0);
    Ax = A*x;
    ytmp = y+(lambda/t+(Ax-b-y));
    y = ytmp - l1_proj(ytmp,1/t);
    cons = Ax-b-y;
    lambda = lambda+gamma*t*cons;
    res = norm(x-x_old)+norm(y-y_old);
    x_old = x;
    y_old = y;
    
    %----------------------------------------------------------------------
    % print information
    
    if(pri&&mod(iter,itprint)==0) && verbose == 2
        if(iter==1||mod(iter-itprint,itprint*10)==0)
            fprintf("%s",str_head);
        end
        obj = mu*norm(x,1)+norm(Ax-b,inf);
        fprintf(str_num,iter,obj,norm(cons),res);
    end
end

%--------------------------------------------------------------------------
% finish and output

out.time = toc;
out.iter = iter;
out.lambda = lambda;
out.y = y;
out.cons = norm(A*x-b-y);
out.gap = abs(mu*norm(x,1)+norm(y,inf)+b'*lambda)/(abs(mu*norm(x,1)+norm(y,inf))+abs(b'*lambda));
out.value = mu*norm(x,1)+norm(A*x-b,inf);
if verbose >= 1
    fprintf("the iteration terminated with time: %3.2e, opt value: %3.2e, constraint violation: %3.2e, opt gap: %3.2e, L1 norm: %3.2e\n",...
        out.time, out.value, out.cons, out.gap, norm(x,1));
end

end
