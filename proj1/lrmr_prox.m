function [x,out] = lrmr_prox(x0,M,Omega,mu,opts)
% Solving low-rank recovery problem with proximal gradient method applying
% continuum strategy.
%
% This program solve problem of the form
%                   min_X  mu*||X||_* + ||P(X)-P(M)||_F^2
%
% We implement both ISTA/FISTA in each iteration
%
% Input:
%        x0 --- The initial point
%        M  --- The given observation matrix
%        P  --- The projection metrix
%        mu --- Nuclear norm regularization term
%      opts --- Options structure with field(s)
%               tol: stop criterion
%               subtol: subproblem stop criterion
%               FISTA: whether use FISTA
%               warm_step: whether calculate step size by Lip constant
%               step: default step size
%               maxiter: maximal iteration
%               maxsubiter: maximal iteration of subproblem
%               ctm: whether use continuation method
%               tol_ctm: stop criteria of ctm
%               mu_ctm: initial mu of ctm
%               ctmsto: when to stop usage of ctm
%               print: whether print information
%               itprint: print frequency
%
% Output:
%         x --- The optimal point founded by algorithm
%       out --- Miscellaneous information during the computation
%% initialization
if ~isfield(opts,'tol');              opts.tol = 1e-8; end
if ~isfield(opts,'FISTA');            opts.FISTA = 1; end
if ~isfield(opts,'warm_step');         opts.warm_step = 0; end
if ~isfield(opts,'step');             opts.step = 1e-1; end
if ~isfield(opts,'maxiter');          opts.maxiter = 3000; end
if ~isfield(opts,'maxsubiter');         opts.maxsubiter = 5e2; end
if ~isfield(opts,'ctm');              opts.ctm = 1; end
if ~isfield(opts,'tol_ctm');           opts.tol_ctm = 1/opts.tol; end
if ~isfield(opts,'mu_ctm');           opts.mu_ctm = 10/mu; end
if ~isfield(opts,'print');            opts.print = 1; end
if ~isfield(opts,'itprint');          opts.itprint = 1; end
if ~isfield(opts,'verbose');          opts.verbose = 1; end

tol = opts.tol;            FISTA = opts.FISTA;
warm_step = opts.warm_step;  step0 = opts.step;          
maxiter = opts.maxiter;    itprint = opts.itprint;    pri = opts.print;
maxsubiter = opts.maxsubiter;  tol_ctm = opts.tol_ctm;      ctm = opts.ctm;
mu_ctm = opts.mu_ctm;      
verbose = opts.verbose; tol_decay = 1e-2;

stra1 = ['%4s','%13s','%12s','%14s','%9s','\n'];
str_head = sprintf(stra1, ...
    'iter','obj','violation','|x-x_k|_2','subiter');
str_num = '%4d  %+5.4e  %+5.4e  %+5.4e   %6d\n';

[m,n] = size(M);
iter = 0;
x = x0;
z = x;
res = inf;
tF = 1;
x_F = x;
% norm_m = norm(M.*P,'F');
% norm_m = norm(M(Omega),'F');
norm_m = norm(M,'F');
if(warm_step)
    xtmp = x;
    xtmp(Omega) = xtmp(Omega)-M;
    cond_x = svds(xtmp,1);
    step = 1/(cond_x^2);
end
if(ctm&&res>tol_ctm)
    mu_now = mu_ctm;
else
    mu_now = mu;
end
%% main loop
tic;
if(pri)
    fprintf('Proximal gradient \n');
end

while(res > tol && iter < maxiter)
    iter = iter+1;
    step = step0/sqrt(max(iter-1000,1));
    step = step0;
    subiter = 0;
    % continuum
    while (mu_now > mu && subiter < maxsubiter && res > tol_ctm)
%         step = step0/max(sqrt(subiter-1000),1);
        xold = x;
        if (FISTA == 0)
%             xtmp = x - step * P.*(x-M);
            z = 2*(x(Omega)-M);
            xtmp = x;
            xtmp(Omega) = xtmp(Omega) - step*z;

%             xtmp = x - step * z.* P;
            [U,S,V] = svd(xtmp,'eco');
            S = diag(max(diag(S)-step*mu_now,0));
            x = U*S*V';
        else
%             xtmp = x - step * P.*(x-M);
            z = 2* (x_F(Omega)-M);
%             xtmp = x_F - step * z(Omega);
%             xtmp = x_F - step * z .* P;
            xtmp = x_F;
            xtmp(Omega) = xtmp(Omega)- step*z;
            [U,S,V] = svd(xtmp,'eco');
            S = diag(max(diag(S)-step*mu_now,0));
            x = U*S*V';
            tFn = (1 + (1 + 4 * tF^2)^0.5)/2;
            x_F = x + (tF - 1)/tFn * (x - xold);
            tF = tFn;
        end
        subiter = subiter+1;
        res = norm(xold-x,'F')/norm_m;
    end
    
    xold = x;
    if (FISTA == 0)
            z = 2*(x(Omega)-M);
        xtmp = x;
        xtmp(Omega) = xtmp(Omega) - step*z;

        [U,S,V] = svd(xtmp,'eco');
        S = diag(max(diag(S)-step*mu_now,0));
        x = U*S*V';
    else
        z = 2*(x_F(Omega)-M);
        xtmp = x_F;
        xtmp(Omega) = xtmp(Omega)- step*z;
        [U,S,V] = svd(xtmp,'eco');
        S = diag(max(diag(S)-step*mu_now,0));
        x = U*S*V';
        tFn = (1 + (1 + 4 * tF^2)^0.5)/2;
        x_F = x + (tF - 1)/tFn * (x - xold);
        tF = tFn;
    end
    iter = iter+1;
    
    res = norm(z,'F')/norm_m;
    if mu_now > mu
        mu_now = mu_now/10;
        iter = iter+subiter;
        tol_ctm = max(tol_ctm*tol_decay,tol);
    end
    
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
out.z = norm(x(Omega)-M,2)/norm_m;
out.sigma = svd(x);
out.normn = norm(out.sigma,1);
out.value = mu*out.normn + norm(z,'F')^2;

if verbose >= 1
    fprintf("the iteration terminated with time: %3.2e, opt value: %3.2e, relative error_M: %3.2e, Nuclear norm: %3.2e\n",...
        out.time, out.value, out.z, out.normn);
end
% if verbose >= 1
%     fprintf("the iteration terminated with time: %3.2e, iteration %d, opt value: %3.2e, relative error_M: %3.2e, Nuclear norm: %3.2e\n",...
%         out.time, out.iter, out.value, out.z, out.normn);
% end

end
