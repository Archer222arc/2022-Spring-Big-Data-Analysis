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

global A_train A_test;
global lambda;

dir = "./dataset";
% [A_m,m1,n1] = sto_load("mnist",dir);
[A_c_train, A_c_test, m_c,n_c] = sto_load("covtype",dir);
[A_g_train, A_g_test, m_g,n_g] = sto_load("gisette",dir);

lambda_list = [10,1,0.1,0.01];
lengn = length(lambda_list);



% %% test dataset covtype
% 
% % f_adam_c = cell(lengn,1); g_adam_c = cell(lengn,1); e_adam_c = cell(lengn,1);
% % f_mom_c = cell(lengn,1); g_mom_c = cell(lengn,1); e_mom_c = cell(lengn,1);
% % f_sgd_c = cell(lengn,1); g_sgd_c = cell(lengn,1); e_sgd_c = cell(lengn,1);
% 
% % for i=1:lengn
% for i = 1:4
% %     A = A_c;
%     A_train = A_c_train;
%     A_test = A_c_test;
%     [m,n] = size(A_train);
%     lambda = lambda_list(i)/m;
%     x0 = zeros(n,1);
%     opt.bsize = 16;
%     opt.maxiter = floor(30*m/opt.bsize)+1;
%     opt.verbose = floor(m/opt.bsize);
%     opt.step = 1e-3;
%     if i > 3
%         opt.step0 = 1e-3;
%     else
%         opt.step0 = 1e-2;
%     end
%     func.n = m;
%     func.f = @f;
%     func.g = @g;
%     func.testfunc = @testfunc;
%     func.trainfunc = @trainfunc;
%     func.fsub = @fsub;
%     func.lambda = lambda;
% 
%     opt.step = 1e-1;
%     [x1,out1] = AdaGrad(x0,func,m,opt);
%     if i > 3
%         opt.step = 1e-2;
%     else
%         opt.step = 1e-3;
%     end
%     [x2,out2] = Adam(x0,func,m,opt);
%     if i > 3
%         opt.step = 1e-1;
%     else
%         opt.step = 1e-2;
%     end
%     [x3,out3] = Momentum(x0,func,m,opt);
%     [x4,out4] = SGD_line(x0,func,m,opt);
%     [x5,out5] = RMSprop(x0,func,m,opt);
%   
% %     f_mom_c{i} = out1.f; g_mom_c{i} = out1.g; e_mom_c{i} = out1.err;
%     f_ada_c{i} = out1.err_train;  g_ada_c{i} = out2.g;  e_ada_c{i} = out1.err_test;
%     f_adam_c{i} = out2.err_train; g_adam_c{i} = out2.g; e_adam_c{i} = out2.err_test;
%     f_mom_c{i} = out3.err_train; g_mom_c{i} = out3.g; e_mom_c{i} = out3.err_test;
%     f_sgd_c{i} = out4.err_train; g_sgd_c{i} = out4.g; e_sgd_c{i} = out4.err_test;
%     f_rms_c{i} = out5.err_train; g_rms_c{i} = out5.g; e_rms_c{i} = out5.err_test;
% 
% end
%% test dataset gisette

% f_adam_c = cell(lengn,1); g_adam_c = cell(lengn,1); e_adam_c = cell(lengn,1);
% f_mom_c = cell(lengn,1); g_mom_c = cell(lengn,1); e_mom_c = cell(lengn,1);
% f_sgd_c = cell(lengn,1); g_sgd_c = cell(lengn,1); e_sgd_c = cell(lengn,1);

% for i=1:lengn
for i = 1:1
%     A = A_c;
    A_train = A_g_train;
    A_test = A_g_test;
    [m,n] = size(A_train);
    lambda = lambda_list(i)/m;
    x0 = zeros(n,1);
    opt.bsize = 16;
    opt.maxiter = floor(30*m/opt.bsize)+1;
    opt.verbose = floor(m/opt.bsize);
    opt.step = 1e-3;
    if i > 3
        opt.step0 = 1e-1;
    else
        opt.step0 = 1e-2;
    end
    func.n = m;
    func.f = @f;
    func.g = @g;
    func.testfunc = @testfunc;
    func.trainfunc = @trainfunc;
    func.fsub = @fsub;
    func.lambda = lambda;

    if i > 3
        opt.step = 1e-2;
    else
        opt.step = 1e-2;
    end
    [x1,out1] = AdaGrad(x0,func,m,opt);
    if i > 3
        opt.step = 1e-3;
    elseif i == 1
        opt.step = 1e-3;
    else
        opt.step = 1e-2;
    end
    [x2,out2] = Adam(x0,func,m,opt);
    if i > 3
        opt.step = 1e-2;
    elseif i == 1
        opt.step = 1e-1;
    else
        opt.step = 1e-2;
    end
    [x3,out3] = Momentum(x0,func,m,opt);
    opt.step = 1e-2;
    [x4,out4] = SGD_line(x0,func,m,opt);
    [x5,out5] = RMSprop(x0,func,m,opt);
  
    f_ada_g{i} = out1.err_train;  g_ada_g{i} = out2.g;  e_ada_g{i} = out1.err_test;
    f_adam_g{i} = out2.err_train; g_adam_g{i} = out2.g; e_adam_g{i} = out2.err_test;
    f_mom_g{i} = out3.err_train; g_mom_g{i} = out3.g; e_mom_g{i} = out3.err_test;
    f_sgd_g{i} = out4.err_train; g_sgd_g{i} = out4.g; e_sgd_g{i} = out4.err_test;
    f_rms_g{i} = out5.err_train; g_rms_g{i} = out5.g; e_rms_g{i} = out5.err_test;
end
% %% test dataset mnist
% % f_adam_m = cell(lengn,1); g_adam_m = cell(lengn,1); e_adam_m = cell(lengn,1);
% % f_mom_m = cell(lengn,1); g_mom_m = cell(lengn,1); e_mom_m = cell(lengn,1);
% % f_sgd_m = cell(lengn,1); g_sgd_m = cell(lengn,1); e_sgd_m = cell(lengn,1);
% 
% 
% for i=1:lengn
%     A = A_m;
%     [m,n] = size(A);
%     lambda = lambda_list(i);
%     x0 = zeros(n,1);
%     opt.bsize = 16;
%     opt.maxiter = 30*m/opt.bsize+1;
%     opt.verbose = 70000/opt.bsize;
%     opts.decay_rate = 1;
%     opt.step = 1e-3;
%     opt.step0 = 1e-2;
%     func.f = @f;
%     func.g = @g;
%     func.testfunc = @testfunc;
%     func.fsub = @fsub;
%     func.lambda = lambda;
%     
%     [x1,out1] = AdaGrad(x0,func,m,opt);
%     [x2,out2] = Adam(x0,func,m,opt);
%     [x3,out3] = SGD_line(x0,func,m,opt);
%     
% %     f_mom_m{i} = out1.f; g_mom_m{i} = out1.g; e_mom_m{i} = out1.err;
%     f_ada_m{i} = out1.f; g_ada_m{i} = out1.g; e_ada_m{i} = out1.err;
%     f_adam_m{i} = out2.f; g_adam_m{i} = out2.g; e_adam_m{i} = out2.err;
% 
%     f_sgd_m{i} = out3.f; g_sgd_m{i} = out3.g; e_sgd_m{i} = out3.err;
% end
function fv = f(x)
    global A_train;
    global lambda;
    [m,~] = size(A_train);
    Ax = A_train*x;
%     Amax = Ax>0;
    fv = sum(1-tanh(Ax))/m+lambda * norm(x)^2;
%     fv = sum(1-tanh(Ax)+Amax.*Ax)/m+lambda * norm(x)^2;
%     fv = sum(log(1+exp(-abs(Ax)))+Amax.*Ax)/m+lambda*norm(x)^2;
end

function fv = fsub(x,t)
    global A_train;
    global lambda;
    [m,~] = size(A_train);
    t = mod(t,m)+1;
    subA = A_train(t,:);
    Ax = subA*x;
%     Amax = Ax>0;
%     fv = sum(1-tanh(Ax)+Ax.*Amax)/length(t)+lambda * norm(x)^2;
    fv = sum(1-tanh(Ax))/length(t)+lambda * norm(x)^2;
end

function gv = g(x,t)
    global A_train;
    global lambda;
    [m,~] = size(A_train);
    t = mod(t,m)+1;
    subA = A_train(t,:);
    subx = subA*x;
    submax = subx>0;
%     submin = subx<=0;
%     esubx = tanh(-abs(subx));
    gv = subA'*(-(1-tanh(subx).^2))/length(t)+lambda*x;
%     gv = subA'*(-(1-tanh(subx).^2)+submax)/length(t)+lambda*x;
%     gv = subA'*((1-tanh(subx).^2+submax+submin.*esubx)./(1+esubx))/length(t)/2+lambda*x;
%     gv = sum(subA'.*((submax+submin.*esubx)./(1+esubx)/length(t))',2)+length(t)*lambda/m*x;
end
function err = trainfunc(x)
    global A_train;
    global lambda;
    [m,~] = size(A_train);
    err = sum(1-tanh(A_train*x))/m + lambda * norm(x)^2 ;
end

function err = testfunc(x)
    global A_test;
    [m,~] = size(A_test);
    err = sum(1-tanh(A_test*x))/2/m;
%     err = sum(sign(A_test*x)+1)/m;
end
