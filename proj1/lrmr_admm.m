function [x,out] = lrmr_admm(x0,M,Omega,mu,opts)
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
%% initialization
if ~isfield(opts,'tol');              opts.tol = 1e-8; end
if ~isfield(opts,'t');                opts.t = 0.1; end
if ~isfield(opts,'gamma');            opts.gamma = 1.618; end
if ~isfield(opts,'step');             opts.step = 1; end
if ~isfield(opts,'maxiter');          opts.maxiter = 1000; end
if ~isfield(opts,'print');            opts.print = 1; end
if ~isfield(opts,'itprint');          opts.itprint = 50; end
if ~isfield(opts,'verbose');          opts.verbose = 1; end

tol = opts.tol;            t = opts.t;                gamma = opts.gamma;
maxiter = opts.maxiter;    itprint = opts.itprint;    pri = opts.print;
alpha = opts.step;        
verbose = opts.verbose;
stra1 = ['%5s','%14s','%15s','%15s','\n'];
str_head = sprintf(stra1, ...
    'iter','obj','violation','|x-x_k|_2');
str_num = '%4d    %+5.4e    %+5.4e    %+5.4e\n';

[m,n] = size(x0);
iter = 0;
% lambda = zeros(m,n);
x = x0;
x_old = x;
y = x(Omega)-M;
cons = zeros(size(y));
lambda = zeros(size(y));
k = 10;
y_old = y;
res = inf;
subres = inf;
tic;
mu_now = mu*1000;

if(pri)
    fprintf('ADMM solver started \n');
end
while(res>tol && iter<maxiter)
%     iter = iter+1;
    subiter = 0;
    % solve subproblem and update multiplier
    
    while (subiter < 100 && subres > tol)
        xtmp = x;
        xtmp(Omega) = xtmp(Omega) - alpha * (lambda+t*cons);
        [U,S,V] = svd(xtmp);
        S = diag(max(diag(S)-alpha*mu_now,0));
        x = U*S*V';    
        y = lambda+t*(x(Omega)-M);
        y = y/(2+t);
        cons = x(Omega)-M-y;
        lambda = lambda+gamma*t*cons;
        subres = (norm(x-x_old,'F')+norm(y-y_old))/norm(x,'F');
        x_old = x;
        y_old = y;
        subiter = subiter+1;
    end
    iter = iter+subiter;
    if mu_now > mu
        mu_now = mu_now/10;
    else
        iter = iter+1;
        xtmp = x;
        xtmp(Omega) = xtmp(Omega) - alpha * (lambda+t*cons);
        [U,S,V] = svd(xtmp,'eco');
%         [U,S,V] = svds(xtmp,k);
        S = diag(max(diag(S)-alpha*mu,0));
        x = U*S*V';    
        y = lambda+t*(x(Omega)-M);
        y = y/(2+t);
        cons = x(Omega)-M-y;
        lambda = lambda+gamma*t*cons;
        res = (norm(x-x_old,'F')+norm(y-y_old))/norm(x,'F');
        x_old = x;
        y_old = y;
    % print information
    end
    
    if(pri&&mod(iter,itprint)==0) && verbose == 2
        if(iter==1||mod(iter-itprint,itprint*10)==0)
            fprintf("%s",str_head);
        end
        obj = mu*norm_nuc(x)+norm(x(Omega)-M,'F')^2;
        fprintf(str_num,iter,obj,norm(cons),res);
    end
end
out.time = toc;
out.iter = iter;
out.z = norm(x(Omega)-M,2)/norm(M);
out.sigma = svd(x);
out.normn = norm(out.sigma,1);
out.value = mu*out.normn + norm(x(Omega)-M)^2;
out.cons = norm(x(Omega)-M-y)/norm(M);

if verbose >= 1
%     fprintf("the iteration terminated with time: %3.2e, opt value: %3.2e, relative error_M: %3.2e, Nuclear norm: %3.2e, Constraint violation: %3.2e\n",...
%         out.time, out.value, out.z, out.cons, out.normn);
    fprintf("the iteration terminated with time: %3.2e, opt value: %3.2e, relative error_M: %3.2e, Nuclear norm: %3.2e",...
        out.time, out.value, out.z, out.normn);
end
end
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
%% initialization
% if ~isfield(opts,'tol');              opts.tol = 1e-8; end
% if ~isfield(opts,'t');                opts.t = 0.1; end
% if ~isfield(opts,'gamma');            opts.gamma = 1.618; end
% if ~isfield(opts,'step');             opts.step = 1; end
% if ~isfield(opts,'maxiter');          opts.maxiter = 1000; end
% if ~isfield(opts,'print');            opts.print = 1; end
% if ~isfield(opts,'itprint');          opts.itprint = 50; end
% if ~isfield(opts,'verbose');          opts.verbose = 1; end
% 
% tol = opts.tol;            t = opts.t;                gamma = opts.gamma;
% maxiter = opts.maxiter;    itprint = opts.itprint;    pri = opts.print;
% alpha = opts.step;        
% verbose = opts.verbose;
% stra1 = ['%5s','%14s','%15s','%15s','\n'];
% str_head = sprintf(stra1, ...
%     'iter','obj','violation','|x-x_k|_2');
% str_num = '%4d    %+5.4e    %+5.4e    %+5.4e\n';
% 
% [m,n] = size(x0);
% iter = 0;
% % lambda = zeros(m,n);
% x = x0;
% x_old = x;
% y = x;
% y(Omega) = x(Omega)-M;
% cons = zeros(m,n);
% lambda = zeros(size(x));
% k = 10;
% y_old = y;
% res = inf;
% subres = inf;
% tic;
% mu_now = mu*1000;
% 
% if(pri)
%     fprintf('ADMM solver started \n');
% end
% while(res>tol && iter<maxiter)
% %     iter = iter+1;
%     subiter = 0;
%     % solve subproblem and update multiplier
%     
%     while (subiter < 50 && subres > tol && mu_now > mu)
%         xtmp = x - alpha*(lambda+t*cons);
% %         xtmp(Omega) = xtmp(Omega) - alpha * (lambda+t*cons);
%         [U,S,V] = svd(xtmp);
%         S = diag(max(diag(S)-alpha*mu_now,0));
%         x = U*S*V';    
% %         y = lambda+t*(x(Omega)-M);
%         y = lambda;
%         y(Omega) = y(Omega)+t*(x(Omega)-M);
%         y = y/(2+t);
%         cons = x-y;
%         cons(Omega) = cons(Omega)-M;
%         lambda = lambda+gamma*t*cons;
%         subres = (norm(x-x_old,'F')+norm(y-y_old,'F'))/norm(x,'F');
%         x_old = x;
%         y_old = y;
%         subiter = subiter+1;
%     end
%     iter = iter+subiter;
%     if mu_now > mu
%         mu_now = mu_now/10;
%     else
%         iter = iter+1;
%         xtmp = x - alpha*(lambda+t*cons);
% %         xtmp(Omega) = xtmp(Omega) - alpha * (lambda+t*cons);
%         [U,S,V] = svd(xtmp);
%         S = diag(max(diag(S)-alpha*mu,0));
%         x = U*S*V';    
% %         y = lambda+t*(x(Omega)-M);
%         y = lambda;
%         y(Omega) = y(Omega)+t*(x(Omega)-M);
%         y = y/(2+t);
%         cons = x-y;
%         cons(Omega) = cons(Omega)-M;
%         lambda = lambda+gamma*t*cons;
%         subres = (norm(x-x_old,'F')+norm(y-y_old,'F'))/norm(x,'F');
%         x_old = x;
%         y_old = y;
%     % print information
%     end
%     
%     if(pri&&mod(iter,itprint)==0) && verbose == 2
%         if(iter==1||mod(iter-itprint,itprint*10)==0)
%             fprintf("%s",str_head);
%         end
%         obj = mu*norm_nuc(x)+norm(x(Omega)-M,'F')^2;
%         fprintf(str_num,iter,obj,norm(cons),res);
%     end
% end
% out.time = toc;
% out.iter = iter;
% out.z = norm(x(Omega)-M,2)/norm(M);
% out.sigma = svd(x);
% out.normn = norm(out.sigma,1);
% out.value = mu*out.normn + norm(x(Omega)-M)^2;
% % out.cons = norm(x(Omega)-M-y)/norm(M);
% 
% if verbose >= 1
% %     fprintf("the iteration terminated with time: %3.2e, opt value: %3.2e, relative error_M: %3.2e, Nuclear norm: %3.2e, Constraint violation: %3.2e\n",...
% %         out.time, out.value, out.z, out.cons, out.normn);
%     fprintf("the iteration terminated with time: %3.2e, opt value: %3.2e, relative error_M: %3.2e, Nuclear norm: %3.2e",...
%         out.time, out.value, out.z, out.normn);
% end