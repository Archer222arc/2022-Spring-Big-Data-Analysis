function [x,out] = RMSprop(x0,func,n,opts)
% implementation of RMSprop for problem of the form
%               min 1/n * sum f(x_i)
%
% Input:
%     --- x0: The initial point
%     --- func: Objective function (structure)
%             --- f : Objective function
%             --- g : Stochastic gradient with batchsize t
%             --- testfunc : Loss in trainset 
%     --- n : Size
%     --- opts :  Options 
%             --- mom: momemtum coefficient
%             --- step: learning rate
%             --- bsize: batch size
%             --- maxiter: number of maximal iteration
%             --- eps: regularization parameter
%             --- print: 1 for print ; 0 for not
%             --- verbose: print frequency
%
% Output:
%     --- x : The optimal point founded by algorithm
%     --- out : Other information
%% initialize
if ~isfield(opts,'mom');              opts.mom = 0.95; end
if ~isfield(opts,'step');             opts.step = 1e-3; end
if ~isfield(opts,'maxiter');          opts.maxiter = 3e6; end
if ~isfield(opts,'bsize');            opts.bsize = 1; end
if ~isfield(opts,'print');            opts.print = 1; end
if ~isfield(opts,'verbose');          opts.verbose = 1e4; end
if ~isfield(opts,'decay_rate');       opts.decay_rate=0.8; end
if ~isfield(opts,'eps');              opts.eps = 1e-6; end

n = func.n;
f = func.f;                g = func.g;                testfunc = func.testfunc;
mom = opts.mom;            maxiter = opts.maxiter;    bsize = opts.bsize;
verbose = opts.verbose;    pri = opts.print;          lr = opts.step;
decay_rate = opts.decay_rate;   eps = opts.eps;       

stra1 = ['%9s','%14s','%15s','%15s','\n'];
str_head = sprintf(stra1, ...
    'iter','train loss','test loss','g norm');
str_num = '%8d    %+5.4e    %+5.4e    %+5.4e\n';

m = length(x0);
iter = 0;
x = x0;
r = zeros(m,1);
t = randperm(n,bsize);
% t = 1:bsize;
err_train = f(x);
err_test = testfunc(x);
gval = norm(g(x,1:n));
tic;

%% main loop

if(pri)
    fprintf('RMSprop solver started \n');
end
while(iter<maxiter)
    iter = iter+1;
    g_now = g(x,t);
    t = t+bsize;
    r = mom*r+(1-mom)*g_now.*g_now;
    d = -1./(eps+sqrt(r)) .* g_now;
    x = x+lr*d;
       
    if(pri&&mod(iter,verbose)==0)
        if(iter==1||mod(iter-verbose,verbose*10)==0)
            fprintf("%s",str_head);
        end
        lr = lr * decay_rate;
        fobj = f(x);
        e_test = testfunc(x);
        gnorm = norm(g(x,1:n)-func.lambda*sign(x));
        err_train = [err_train,fobj]; 
        err_test = [err_test,e_test];
        gval = [gval,gnorm];
        fprintf(str_num,iter,fobj,e_test,gnorm);
    end
end

%% output

out.time = toc;
out.iter = iter;
out.err_test = err_test;
out.err_train = err_train;
out.g = gval;

end
