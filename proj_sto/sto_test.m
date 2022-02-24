% Compare the performance of different algorithms for logistic regression 
% with l1 regularization
%           min 1/n * sum log(1+exp(-y_i w^t x))+ lambda * ||w||_1
% Implementation of 
%           Adam
%           Momentum
%           SGD with linesearch (extra-credit)

%% Initialization and preprocess the datasets
% clc;
% clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);

global A;
global lambda;

dir = "./dataset";
[A_m,m1,n1] = sto_load("mnist",dir);
[A_c,m2,n2] = sto_load("covtype",dir);

lambda_list = [10,1,0.1,0.01];
lengn = length(lambda_list);

%% test dataset mnist
% f_adam_m = cell(lengn,1); g_adam_m = cell(lengn,1); e_adam_m = cell(lengn,1);
% f_mom_m = cell(lengn,1); g_mom_m = cell(lengn,1); e_mom_m = cell(lengn,1);
% f_sgd_m = cell(lengn,1); g_sgd_m = cell(lengn,1); e_sgd_m = cell(lengn,1);


for i=1:lengn
    A = A_m;
    [m,n] = size(A);
    lambda = lambda_list(i);
    x0 = zeros(n,1);
    opt.bsize = 16;
    opt.maxiter = 30*m/opt.bsize+1;
    opt.verbose = 70000/opt.bsize;
    opts.decay_rate = 1;
    opt.step = 1e-3;
    opt.step0 = 1e-2;
    func.f = @f;
    func.g = @g;
    func.testfunc = @testfunc;
    func.fsub = @fsub;
    func.lambda = lambda;
    
    [x1,out1] = Momentum(x0,func,m,opt);
%     [x2,out2] = Adam(x0,func,m,opt);
%     [x3,out3] = SGD_line(x0,func,m,opt);
    
    f_mom_m{i} = out1.f; g_mom_m{i} = out1.g; e_mom_m{i} = out1.err;
%     f_adam_m{i} = out2.f; g_adam_m{i} = out2.g; e_adam_m{i} = out2.err;

%     f_sgd_m{i} = out3.f; g_sgd_m{i} = out3.g; e_sgd_m{i} = out3.err;
end

%% test dataset covtype

% f_adam_c = cell(lengn,1); g_adam_c = cell(lengn,1); e_adam_c = cell(lengn,1);
% f_mom_c = cell(lengn,1); g_mom_c = cell(lengn,1); e_mom_c = cell(lengn,1);
% f_sgd_c = cell(lengn,1); g_sgd_c = cell(lengn,1); e_sgd_c = cell(lengn,1);

% for i=1:lengn
for i = 3
    A = A_c;
    [m,n] = size(A);
    lambda = lambda_list(i);
    x0 = zeros(n,1);
    opt.bsize = 16;
    opt.maxiter = floor(30*m/opt.bsize)+1;
    opt.verbose = floor(m/opt.bsize);
    opt.step = 1e-3;
    if i > 3
        opt.step0 = 1e-3;
    else
        opt.step0 = 1e-2;
    end
    func.f = @f;
    func.g = @g;
    func.testfunc = @testfunc;
    func.fsub = @fsub;
    func.lambda = lambda;

%     [x1,out1] = Momentum(x0,func,m,opt);
%     [x2,out2] = Adam(x0,func,m,opt);
    [x3,out3] = SGD_line(x0,func,m,opt);
  
%     f_mom_c{i} = out1.f; g_mom_c{i} = out1.g; e_mom_c{i} = out1.err;
%     f_adam_c{i} = out2.f; g_adam_c{i} = out2.g; e_adam_c{i} = out2.err;
    f_sgd_c{i} = out3.f; g_sgd_c{i} = out3.g; e_sgd_c{i} = out3.err;
end

function fv = f(x)
    global A;
    global lambda;
    [m,~] = size(A);
    Ax = A*x;
    Amax = Ax>0;
    fv = sum(log(1+exp(-abs(Ax)))+Amax.*Ax)/m+lambda/m*norm(x,1);
end

function fv = fsub(x,t)
    global A;
    global lambda;
    [m,~] = size(A);
    t = mod(t,m)+1;
    subA = A(t,:);
    Ax = subA*x;
    Amax = Ax>0;
    fv = sum(log(1+exp(-abs(Ax)))+Amax.*Ax)/length(t)+length(t)*lambda/m*norm(x,1);
end

function gv = g(x,t)
    global A;
    global lambda;
    [m,~] = size(A);
    t = mod(t,m)+1;
    subA = A(t,:);
    subx = subA*x;
    submax = subx>0;
    submin = subx<=0;
    esubx = exp(-abs(subx));
    gv = sum(subA'.*((submax+submin.*esubx)./(1+esubx)/length(t))',2)+length(t)*lambda/m*sign(x);
end

function err = testfunc(x)
    global A;
    [m,~] = size(A);
    err = sum(sign(A*x)+1)/2/m;
end
