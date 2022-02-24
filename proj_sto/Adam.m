function [x,out] = Adam(x0,func,n,opts)
% implementation of Adam for problem of the form
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
%             --- beta1:  decay rate 1
%             --- beta2: decay rate 2
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
%
%% initialize

if ~isfield(opts,'beta1');               opts.beta1 = 0.999; end
if ~isfield(opts,'beta2');              opts.beta2 = 0.999; end
if ~isfield(opts,'eps');              opts.eps = 1e-5; end
if ~isfield(opts,'step');             opts.step = 1e-3; end
if ~isfield(opts,'maxiter');          opts.maxiter = 3e6; end
if ~isfield(opts,'bsize');            opts.bsize = 1; end
if ~isfield(opts,'print');            opts.print = 1; end
if ~isfield(opts,'verbose');          opts.verbose = 1e4; end
if ~isfield(opts,'decay_rate');       opts.decay_rate = 0.5; end

f = func.f;                g = func.g;                testfunc = func.testfunc;
beta1 = opts.beta1;              beta2 = opts.beta2;            bsize = opts.bsize;
eps = opts.eps;            verbose = opts.print;          lr = opts.step;
verbose = opts.verbose;    maxiter = opts.maxiter;    
decay_rate = opts.decay_rate;

stra1 = ['%9s','%14s','%15s','%15s','\n'];
str_head = sprintf(stra1, ...
    'iter','f','error','g norm');
str_num = '%8d    %+5.4e    %+5.4e    %+5.4e\n';

m = length(x0);
iter = 0;
x = x0;
Eg2 = zeros(m,1);
Eg = zeros(m,1);
exp_beta1 = 1;
exp_beta2 = 1;
t = randperm(m,bsize);
% t = 1:bsize;
fval = f(x);
err_out = testfunc(x);
gval = norm(g(x,1:n));
tic;

%% main loop
if(verbose)
    fprintf('ADAM solver started \n');
end
while(iter<maxiter)
    iter = iter+1;
    exp_beta1 = exp_beta1*beta1;
    exp_beta2 = exp_beta2*beta2;
    
    g_now = g(x,t);
    t = t+bsize;
    Eg = beta1*Eg+(1-beta1)*g_now;
    Eg2 = beta2*Eg2+(1-beta2)*(g_now.^2);
    Eghat = Eg/(1-exp_beta1);
    Eg2hat = Eg2/(1-exp_beta2);
    x = x-lr./(sqrt(Eg2hat+eps)).*Eghat;
    
    
    if(verbose&&mod(iter,verbose)==0)
        if(iter==1||mod(iter-verbose,verbose*10)==0)
            fprintf("%s",str_head);
        end
        lr = lr*decay_rate;
        fobj = f(x);
        err = testfunc(x);
        gnorm = norm(g(x,1:n)-func.lambda*sign(x));
        fval = [fval,fobj]; 
        err_out = [err_out,err];
        gval = [gval,gnorm];
        fprintf(str_num,iter,fobj,err,gnorm);
    end
end

%% output 

out.time = toc;
out.iter = iter;
out.f = fval;
out.err = err_out;
out.g = gval;

end
