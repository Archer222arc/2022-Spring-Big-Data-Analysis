

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
    
    % cvx
    
    opts1.solver = 0;   % mosek
    [x_mosek, out_mosek] = l1_cvx(A, b, mu, opts1);
    norm_mosek(i) = norm(x_mosek,1); time_mosek(i) = out_mosek.time; 
    vopt_mosek(i) = abs(out_mosek.value-mu*norm(u,1))/abs(mu*norm(u,1));
    
    opts1.solver = 1;   % gurobi
    [x_gurobi, out_gurobi] = l1_cvx(A, b, mu, opts1);
    norm_gurobi(i) = norm(x_gurobi,1); time_gurobi(i) = out_gurobi.time; 
    vopt_gurobi(i) = abs(out_gurobi.value-mu*norm(u,1))/abs(mu*norm(u,1));
    merr_gurobi(i) = errfun(x_gurobi, x_mosek);

    % Augmented Lagrangian method
    
    opts2.FISTA = 0; % ISTA
    opts2.ctm = 0;
    [x_ALM_ISTA,out_21] =  l1_alm(x0,A,b,mu,opts2);
    norm_ALM1(i) = norm(x_ALM_ISTA,1); time_ALM1(i) = out_21.time; 
    vopt_ALM1(i) = abs(out_21.value-mu*norm(u,1))/abs(mu*norm(u,1));
    consvio_ALM1(i) = out_21.cons;  optgap_ALM1(i) = out_21.gap;
    merr_ALM1(i) = errfun(x_ALM_ISTA, x_mosek);

%     opts2.FISTA = 0; % ISTA + continuation
%     opts2.ctm = 1;
%     [x_22,out_22] =  l1_alm(x0,A,b,mu,opts2);
%     norm_ALM2(i) = norm(x_22,1); time_ALM2(i) = out_22.time; 
%     vopt_ALM2(i) = abs(out_22.value-mu*norm(u,1))/abs(mu*norm(u,1));
%     consvio_ALM2(i) = out_22.cons;  optgap_ALM2(i) = out_22.gap;    

    opts2.FISTA = 1; % FISTA
    opts2.ctm = 0;
    [x_ALM_FISTA,out_ALM_FISTA] =  l1_alm(x0,A,b,mu,opts2);
    norm_ALM(i) = norm(x_ALM_FISTA,1); time_ALM(i) = out_ALM_FISTA.time; 
    vopt_ALM(i) = abs(out_ALM_FISTA.value-mu*norm(u,1))/abs(mu*norm(u,1));
    consvio_ALM(i) = out_ALM_FISTA.cons;  optgap_ALM(i) = out_ALM_FISTA.gap;
    merr_ALM(i) = errfun(x_ALM_FISTA, x_mosek);

    
    % ADMM

    [x_ADMM,out_ADMM] =  l1_admm_primal(x0,A,b,mu,[]); 
    norm_ADMM(i) = norm(x_ADMM,1); time_ADMM(i) = out_ADMM.time; 
    vopt_ADMM(i) = abs(out_ADMM.value-mu*norm(u,1))/abs(mu*norm(u,1));
    consvio_ADMM(i) = out_ADMM.cons;  optgap_ADMM(i) = out_ADMM.gap;
    merr_ADMM(i) = errfun(x_ADMM, x_mosek);

    
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


