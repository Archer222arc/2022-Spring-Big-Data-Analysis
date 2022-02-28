function [x, out] = lrmr_cvx(x0,M,Omega,mu,opts) 
% Solving L1 minimization problem directly via CVX toolbox.
%
% The program aims to solve the L1 minimization problem of the form
% 
%        min_x \mu mu*||X||_* + ||P(X)-P(M)||_F^2
%
% Input:
%         A --- The matrix apperaed in optimization problem 
%         b --- The vector appeared in optimization problem
%        mu --- L1 regularization term
%      opts --- Options structure with field(s)
%               solver: CVX internal solver, 0 -- Mosek
%
% Output:
%         x --- The optimal point founded by algorithm
%       out --- Miscellaneous information during the computation
%
%% initialization

if ~isfield(opts,'solver');              opts.solver = 1; end

solver = opts.solver;
[m,n]=size(x0);

if(solver==0)
    fprintf("cvx_mosek begin \n");
    cvx_begin quiet
        cvx_solver mosek
        variable x(m,n)
        minimize(mu*norm_nuc(x)+square_pos(norm(x(Omega)-M,2)))
    cvx_end
else
    fprintf("cvx_gurobi begin \n");
    cvx_begin quiet
        cvx_solver gurobi
        variable x(m,n)
        minimize(mu*norm_nuc(x)+square_pos(norm(x(Omega)-M,2)))
    cvx_end
end

out.time = cvx_cputime;
out.value = cvx_optval;
out.tol = cvx_slvtol;
out.status = cvx_status;
out.normn = norm_nuc(x);
out.z = norm(x(Omega)-M,2)/norm(M,2);
fprintf("the iteration terminated with time: %3.2e, opt value: %3.2e, relative error M: %3.2e, Nuclear norm: %3.2e\n",...
        out.time,out.value, out.z, out.normn);
end
