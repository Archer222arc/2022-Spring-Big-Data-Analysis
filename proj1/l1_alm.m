function [x, out] = l1_alm(x0,A,b,mu,opts)
% Solving L1 minimization problem directly via Augmented Lagrangian method,
% which is equivalent to Bregman method on this problem
%
% The program aims to solve the L1 minimization problem of the form
%
%        min_x \mu ||x||_1  +  ||Ax-b||_1
%
% Augmented Lagrangian method solves the following subproblem by ISTA/FISTA
% in each iteration
%
% min_{x,y} \mu ||x||_1  +  ||y||_1  +  \lambda^T (Ax-b-y)  +  t/2||Ax-b-y||_2^2,
%
% and updates Lagrange multiplier according to the soultion of subproblem.
% We also apply a continuum method for faster convergence.
%
% Input:
%        x0 --- The initial point
%         A --- The matrix apperaed in optimization problem
%         b --- The vector appeared in optimization problem
%        mu --- L1 regularization term
%      opts --- Options structure with field(s)
%               tol: stop criterion
%               subtol: subproblem stop criterion
%               FISTA: whether use FISTA
%               warm_step: whether calculate step size by Lip constant
%               step: default step size
%               t: update parameter of Lagrange multiplier
%               maxiter: maximal iteration
%               maxsubiter: maximal iteration of subproblem
%               ctm: whether use continuation method
%               tol_ctm: stop criteria of ctm
%               mu_ctm: initial mu of ctm
%               converge_ctm: when to stop usage of ctm
%               print: whether print information
%               itprint: print frequency
%
% Output:
%         x --- The optimal point founded by algorithm
%       out --- Miscellaneous information during the computation
%
%% initialization

if ~isfield(opts,'tol');              opts.tol = 1e-10; end
if ~isfield(opts,'subtol');           opts.subtol = 1e-8; end
if ~isfield(opts,'FISTA');            opts.FISTA = 1; end
if ~isfield(opts,'warm_step');         opts.warm_step = 1; end
if ~isfield(opts,'step');             opts.step = 1e-3; end
if ~isfield(opts,'t');                opts.t = 1e-3; end
if ~isfield(opts,'maxiter');          opts.maxiter = 5e2; end
if ~isfield(opts,'maxsubiter');         opts.maxsubiter = 4e3; end
if ~isfield(opts,'ctm');              opts.ctm = 0; end
if ~isfield(opts,'tol_ctm');           opts.tol_ctm = opts.tol*10; end
if ~isfield(opts,'mu_ctm');           opts.mu_ctm = mu*1e2; end
if ~isfield(opts,'converge_ctm');           opts.converge_ctm = 1e-3; end
if ~isfield(opts,'print');            opts.print = 1; end
if ~isfield(opts,'itprint');          opts.itprint = 1; end
if ~isfield(opts,'verbose');          opts.verbose = 1; end

tol = opts.tol;            subtol = opts.subtol;      FISTA = opts.FISTA;
warm_step = opts.warm_step;  step = opts.step;          t = opts.t;
maxiter = opts.maxiter;    itprint = opts.itprint;    pri = opts.print;
maxsubiter = opts.maxsubiter;  tol_ctm = opts.tol_ctm;      ctm = opts.ctm;
mu_ctm = opts.mu_ctm;      converge_ctm = opts.converge_ctm;
verbose = opts.verbose;

stra1 = ['%4s','%13s','%12s','%14s','%9s','\n'];
str_head = sprintf(stra1, ...
    'iter','obj','violation','|x-x_k|_2','subiter');
str_num = '%4d  %+5.4e  %+5.4e  %+5.4e   %6d\n';

[m,n] = size(A);
iter = 0;
x = x0;
xone = ones(n,1);
y = A*x-b;
lambda = zeros(m,1);
res = inf;

if(warm_step)
    cond_A = svds(A,1);
    step = 1/(t*cond_A^2);
end
tic;

%% main loop

if(pri)
    fprintf('Augmented Lagrangian solver started \n');
end
while(res > tol && iter < maxiter)
    iter = iter+1;
    subtotiter = 0;
    
    % solve subproblem by (accelerated) proximal gradient method
    
    if(ctm&&res>converge_ctm)
        mu_now = mu_ctm;
    else
        mu_now = mu;
    end
    while 1
        subiter = 0;
        subres = inf;
        xsubold = x;
        ysubold = y;
        x_F = x;
        y_F = y;
        tF = 1;
        while(subiter<maxsubiter)
            if(mu_now~=mu&&subres<tol_ctm)
                break
            elseif(subres<subtol)
                break;
            end
            subiter = subiter+1;
            subtotiter = subtotiter+1;
            if(FISTA==0)
                z = A * x - b - y;
                xtmp = x - step * (A' * (lambda + t*z));
                x = max(xtmp-mu_now*step*xone,0)+min(xtmp+mu_now*step*xone,0);
               
                ytmp = y+step * (lambda-t*z);
                y = sign(ytmp).* max(abs(ytmp)-step,0);
%                 y = ytmp-l1_proj(ytmp,step);
            else
                z = A*x_F-b-y_F;
                xtmp = x_F-step * (A' * (lambda+t*z));
                x = max(xtmp - mu_now * step * xone,0) + min(xtmp + mu_now * step * xone,0);
                ytmp = y_F+step * (lambda - t * z);
                y = sign(ytmp).* max(abs(ytmp)-step,0);
%                 y = ytmp-l1_proj(ytmp,step);
                tFn = (1 + (1 + 4 * tF^2)^0.5)/2;
                x_F = x + (tF - 1)/tFn * (x - xsubold);
                y_F = y + (tF - 1)/tFn * (y - ysubold);
                tF = tFn;
            end
            subres = norm(xsubold-x)+norm(ysubold-y);
            ysubold = y;
            xsubold = x;
        end
        if(mu_now == mu)
            break;
        else
            mu_now = max(mu,mu_now/10);
        end
    end
    
    % update Lagrange multiplier
    
    lambda = lambda+t*(A*x-b-y);
    gap = abs(mu*norm(x,1)+norm(y,inf)+b'*lambda)/(abs(mu*norm(x,1)+norm(y,inf))+abs(b'*lambda));
    res = gap+norm(z,2)/norm(b,2);
    
    % print information
    
    if(pri&&mod(iter,itprint)==0) && verbose == 2
        if(iter==1||mod(iter-itprint,itprint*10)==0)
            fprintf("%s",str_head);
        end
        obj = mu*norm(x,1)+norm(A*x-b);
        fprintf(str_num,iter,obj,norm(z),res,subtotiter);
    end
end

%% output


out.time = toc;
out.iter = iter;
out.lambda = lambda;
out.y = y;
out.z = norm(A*x-b-y)/norm(b,2);
out.gap = abs(mu*norm(x,1)+norm(y,inf)+b'*lambda)/(abs(mu*norm(x,1)+norm(y,1))+abs(b'*lambda));
out.value = mu*norm(x,1)+norm(A*x-b,inf);
out.cons = norm(A*x-b-y)/norm(b,2);

if verbose >= 1
    fprintf("the iteration terminated with time: %3.2e, opt value: %3.2e, constraint violation: %3.2e, opt gap: %3.2e, L1 norm: %3.2e\n",...
        out.time, out.value, out.cons, out.gap, norm(x,1));
end
