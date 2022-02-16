% clc;
% clear;
seed = 97006855;
% seed = 1;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));
resfun = @(x) norm(A*x-b,inf);
nrm1fun = @(x) norm(x,1);

n_list = [256,512,1024,2048];
m_list = [128,256,512,1024];
mu = 1e-2;
sparsity = 0.1;

num = length(n_list);
norm_mosek = zeros(num,1); time_mosek = zeros(num,1); vopt_mosek = zeros(num,1);
norm_ALM = zeros(num,1); time_ALM = zeros(num,1); vopt_ALM = zeros(num,1);
norm_gurobi = zeros(num,1); time_gurobi = zeros(num,1); vopt_gurobi = zeros(num,1);
norm_ADMM = zeros(num,1); time_ADMM = zeros(num,1); vopt_ADMM = zeros(num,1);
consvio_ALM = zeros(num,1); consvio_ADMM = zeros(num,1); 
optgap_ALM = zeros(num,1); optgap_ADMM = zeros(num,1);

for i=1:num
    n = n_list(i); m = m_list(i);
    [A,u,b] = l1_gen(m,n,0.1);
    x0 = rand(n,1);
    
    %----------------------------------------------------------------------
    % CVX
    
    opts1.solver = 0;   % mosek
    [x_11, out_11] = l1_cvx(A, b, mu, opts1);
    norm_mosek(i) = norm(x_11,1); time_mosek(i) = out_11.time; 
    vopt_mosek(i) = abs(out_11.value-mu*norm(u,1))/abs(mu*norm(u,1));
    
    opts1.solver = 1;   % gubori
    [x_12, out_12] = l1_cvx(A, b, mu, opts1);
    norm_gurobi(i) = norm(x_12,1); time_gurobi(i) = out_12.time; 
    vopt_gurobi(i) = abs(out_12.value-mu*norm(u,1))/abs(mu*norm(u,1));
    merr_gurobi(i) = errfun(x_12, x_11);

%     
    %----------------------------------------------------------------------
    % Augmented Lagrangian method
    
    opts2.FISTA = 0; % ISTA
    opts2.ctm = 0;
    [x_21,out_21] =  l1_alm(x0,A,b,mu,opts2);
    norm_ALM1(i) = norm(x_21,1); time_ALM1(i) = out_21.time; 
    vopt_ALM1(i) = abs(out_21.value-mu*norm(u,1))/abs(mu*norm(u,1));
    consvio_ALM1(i) = out_21.cons;  optgap_ALM1(i) = out_21.gap;
    merr_ALM1(i) = errfun(x_21, x_11);

%     opts2.FISTA = 0; % ISTA + continuation
%     opts2.ctm = 1;
%     [x_22,out_22] =  l1_alm(x0,A,b,mu,opts2);
%     norm_ALM2(i) = norm(x_22,1); time_ALM2(i) = out_22.time; 
%     vopt_ALM2(i) = abs(out_22.value-mu*norm(u,1))/abs(mu*norm(u,1));
%     consvio_ALM2(i) = out_22.cons;  optgap_ALM2(i) = out_22.gap;    

    opts2.FISTA = 1; % FISTA
    opts2.ctm = 0;
    [x_23,out_23] =  l1_alm(x0,A,b,mu,opts2);
    norm_ALM(i) = norm(x_23,1); time_ALM(i) = out_23.time; 
    vopt_ALM(i) = abs(out_23.value-mu*norm(u,1))/abs(mu*norm(u,1));
    consvio_ALM(i) = out_23.cons;  optgap_ALM(i) = out_23.gap;
    merr_ALM(i) = errfun(x_23, x_11);

%     
%     %----------------------------------------------------------------------
%     % ADMM
% 
    %[x_31,~,~,out_31] =  ADMMdual_L1solver(A,b,mu,[]); % dual
    [x_32,out_32] =  l1_admm_primal(x0,A,b,mu,[]); % primal
    norm_ADMM(i) = norm(x_32,1); time_ADMM(i) = out_32.time; 
    vopt_ADMM(i) = abs(out_32.value-mu*norm(u,1))/abs(mu*norm(u,1));
    consvio_ADMM(i) = out_32.cons;  optgap_ADMM(i) = out_32.gap;
    merr_ADMM(i) = errfun(x_32, x_11);

    
    fprintf('The comparison results: (opt value represents the gap between current solution and ground truth)\n')
    fprintf("\nThe scale of matrix A is (n, m) = (%4d, %4d)\n",n_list(i),m_list(i));
    fprintf('CVX --- mosek:     time: %3.2e, opt value: %3.2e, L1 norm: %3.2e\n', ...
        time_mosek(i),vopt_mosek(i),norm_mosek(i));
    fprintf('    --- gurobi:    time: %3.2e, opt value: %3.2e, err-to-cvx-mosek: %3.2e,, L1 norm: %3.2e\n', ...
        time_gurobi(i),vopt_gurobi(i),merr_gurobi(i),norm_gurobi(i));
    fprintf('ALM+ISTA:         time: %3.2e, opt value: %3.2e, constraint violation: %3.2e, opt gap: %3.2e, err-to-cvx-mosek: %3.2e, L1 norm: %3.2e\n', ...
        time_ALM1(i),vopt_ALM1(i),consvio_ALM1(i),optgap_ALM1(i),merr_ALM1(i),norm_ALM1(i));
%     fprintf('ALM+ISTA+Continuum:         time: %3.2e, opt value: %3.2e, constraint violation: %3.2e, opt gap: %3.2e, L1 norm: %3.2e\n', ...
%         time_ALM2(i),vopt_ALM2(i),consvio_ALM2(i),optgap_ALM2(i),norm_ALM2(i));
    fprintf('ALM+FISTA:         time: %3.2e, opt value: %3.2e, constraint violation: %3.2e, opt gap: %3.2e, err-to-cvx-mosek: %3.2e, L1 norm: %3.2e\n', ...
        time_ALM(i),vopt_ALM(i),consvio_ALM(i),optgap_ALM(i),merr_ALM(i),norm_ALM(i));
    fprintf('Linearized ADMM:   time: %3.2e, opt value: %3.2e, constraint violation: %3.2e, opt gap: %3.2e, err-to-cvx-mosek: %3.2e, L1 norm: %3.2e \n', ...
        time_ADMM(i),vopt_ADMM(i),consvio_ADMM(i),optgap_ADMM(i),merr_ADMM(i),norm_ADMM(i));
end

%--------------------------------------------------------------------------
% print comparison result

