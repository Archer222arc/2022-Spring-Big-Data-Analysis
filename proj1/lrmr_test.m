% clc;
% clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));
resfun = @(x) norm(A*x-b,1);
nrm1fun = @(x) norm(x,1);

m_list = [40,100,200,500,150];
n_list = [40,100,200,500,300];
r_list = [3,2,10,10,10];
Omega_list = cell(5);
% for i = 1 : 5
%     Omega_list{i} = lrmc_gen(i);
% end

mu_list = [0.1,0.01,0.001];
out_list = cell(3);

for i = 1 : 5
    m = m_list(i);
    n = n_list(i);
    [M,Omega,A] = lrmr_gen(i);
    x0 = rand(m,n);
    data = zeros(4,5);
    for j = 1:3
        mu = mu_list(j);
        opts_mosek.solver = 0;
        [x_mosek,out_mosek] = lrmr_cvx(x0,M,Omega,mu,opts_mosek);
        
%         opts_gurobi.solver = 1;
%         [x_gurobi,out_gurobi] = lrmr_cvx(x0,M,Omega,mu,opts_gurobi);
        
        opts_prox.FISTA = 1;
        [x_prox,out_prox] = lrmr_prox(x0,M,Omega,mu,opts_prox);
        
        opts_ista.FISTA = 0;
        [x_ista,out_ista] = lrmr_prox(x0,M,Omega,mu,opts_ista);
        
        opts_admm = {};
        [x_admm,out_admm] = lrmr_admm(x0,M,Omega,mu,opts_admm);

        
        fprintf("The comparison results:\n");
        fprintf("The scale of matrix A is (m,n) = (%4d, %4d)\n",m_list(i),n_list(i));
        fprintf("CVX mosek:    time:%3.2e, opt value: %3.2e, relative error M: %3.2e, relative error A: %3.2e, Nuclear norm: %3.2e\n",...
            out_mosek.time, out_mosek.value, out_mosek.z, norm(x_mosek-A,'fro')/norm(A,'F'), out_mosek.normn);
        fprintf("Proximal (FISTA) :    time:%3.2e, opt value: %3.2e, relative error M: %3.2e, relative error A: %3.2e, Nuclear norm: %3.2e\n",...
            out_prox.time, out_prox.value, out_prox.z, norm(x_prox-A,'fro')/norm(A,'F'), out_prox.normn);
        fprintf("Proximal (ISTA) :    time:%3.2e, opt value: %3.2e, relative error M: %3.2e, relative error A: %3.2e, Nuclear norm: %3.2e\n",...
            out_ista.time, out_ista.value, out_ista.z, norm(x_ista-A,'fro')/norm(A,'F'), out_ista.normn);  
%         fprintf("ADMM :    time:%3.2e, opt value: %3.2e, relative error M: %3.2e, relative error A: %3.2e, Nuclear norm: %3.2e\n",...
%             out_admm.time, out_admm.value, out_admm.z, norm(x_admm-A,'fro')/norm(A,'F'), out_admm.normn);   
        fprintf("ADMM :    time:%3.2e, opt value: %3.2e, relative error M: %3.2e, relative error A: %3.2e, constraint violation: %3.2e, Nuclear norm: %3.2e\n",...
            out_admm.time, out_admm.value, out_admm.z, norm(x_admm-A,'fro')/norm(A,'F'), out_admm.cons, out_admm.normn);   
        data(1,:) = [out_mosek.time, out_mosek.value, out_mosek.z, norm(x_mosek-A,'fro')/norm(A,'F'), out_mosek.normn];
        data(3,:) = [out_prox.time, out_prox.value, out_prox.z, norm(x_prox-A,'fro')/norm(A,'F'), out_prox.normn];
        data(2,:) = [out_ista.time, out_ista.value, out_ista.z, norm(x_ista-A,'fro')/norm(A,'F'), out_ista.normn];  
        data(4,:) = [out_admm.time, out_admm.value, out_admm.z, norm(x_admm-A,'fro')/norm(A,'F'), out_admm.normn];
        
        % make table
        label.title1 = ["Algorithm","$\\mu=$"+string(mu)];
        label.title2 = ["CPU time","Opt value","Error with M in norm $\\|\\cdot\\|_\\F$","Error with in norm $\\|\\cdot\\|_\\F$","$\\|x\\|_\\F$"];
        label.length1 = [5];
%         label.title2 = ["-","$c=2k$","$c=10k$","$c=50k$","$q=0$","$q=1$","$q=2$"];
        label.col = ["Mosek","Proximal (ISTA)","Proximal (FISTA)","ADMM"];
        opt.H = "None";

        % time
        opt.label = "example"+string(i);
        opt.caption = "Numerical result on example "+string(i)+" with different $\\mu$";
           
        opt.filename = "./table/example"+string(i)+string(j);
        maketable(data,label,opt);

    end
end
