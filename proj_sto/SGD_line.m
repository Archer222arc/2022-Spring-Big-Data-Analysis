function [x,out] = SGD_line(x0,func,n,opts)
%--
% Solving a general stochastic optimization problem via Stochastic line search.
%
% The program aims to solve the L1 minimization problem of the form
%
%         min_x \sum_i f_i(x)
%
% Input:
%     --- x0: The initial point
%     --- func: Objective function (structure)
%             --- f : Objective function
%             --- g : Stochastic gradient with batchsize t
%             --- testfunc : Loss in trainset 
%             --- fsub : Objective function in batch
%     --- n : Size
%     --- opts :  Options 
%             --- gamma: accept parameter in linesearch
%             --- submaxiter: maximal search times
%             --- step0: initial learning rate
%             --- bsize: batch size
%             --- maxiter: number of maximal iteration
%             --- print: 1 for print ; 0 for not
%             --- verbose: print frequency
%
% Output:
%     --- x : The optimal point founded by algorithm
%     --- out : Other information
%
%% initialize
if ~isfield(opts,'gamma');            opts.gamma = 0.1; end
if ~isfield(opts,'submaxiter');             opts.submaxiter = 5; end
if ~isfield(opts,'step0');            opts.step0 = 1e-1; end
if ~isfield(opts,'maxiter');          opts.maxiter = 3e6; end
if ~isfield(opts,'bsize');            opts.bsize = 1e2; end
if ~isfield(opts,'print');            opts.print = 1; end
if ~isfield(opts,'verbose');          opts.verbose = 1e4; end


gamma = opts.gamma;        maxiter = opts.maxiter;    bsize = opts.bsize;
verbose = opts.verbose;    pri = opts.print;          lr = opts.step0;
submaxiter = opts.submaxiter;
f = func.f;     g = func.g;     testfunc = func.testfunc;   fsub = func.fsub;

stra1 = ['%9s','%14s','%15s','%15s','\n'];
str_head = sprintf(stra1, ...
    'iter','f','error','g norm');
str_num = '%8d    %+5.4e    %+5.4e    %+5.4e\n';

m = length(x0);
iter = 0;
mu = 0.95;
x = x0;
t = randperm(m,bsize);

% t = 1:bsize;
fval = f(x);
err_out = testfunc(x);
nu = g(x,1:n);
gval = norm(g(x,1:n));
tic;

%% main loop
if(pri)
    fprintf('Stochastic line search solver started \n');
end

while(iter<maxiter)
    iter = iter+1;
    grad = g(x,t);
    step = lr;
    subiter = 0;
    obj_p = fsub(x,t);
    nu = grad;
%     nu = mu*nu + (1-mu)*grad;
%     n_g = norm(grad);
%     n_g = norm(g(x,1:n)-func.lambda*sign(x));
    n_g = norm(grad-func.lambda*sign(x));
%     x_new = x-step*grad;
    x_new = x-step*nu;

    while(subiter<submaxiter)
        subiter = subiter+1;
        if(fsub(x_new,t)<obj_p-gamma*step*n_g^2)
            break;
        end
%         x_new = x-step*grad;
        x_new = x-step*nu;
        step = 0.5*step;
    end
    t = t+bsize;
    x = x_new;
    
    
    if(pri&&mod(iter,verbose)==0)
        if(iter==1||mod(iter-verbose,verbose*10)==0)
            fprintf("%s",str_head);
        end
        fobj = f(x);
        err = testfunc(x);
        fgrad = norm(g(x,1:n)-func.lambda*sign(x));
%         step = step * 0.5;
        fval = [fval,fobj]; 
        err_out = [err_out,err];
        gval = [gval,fgrad];
        fprintf(str_num,iter,fobj,err,fgrad);
    end
end

out.time = toc;
out.iter = iter;
out.f = fval;
out.err = err_out;
out.g = gval;

end
