function [x, out] = l1_cvx(A,b,mu,opts) %#ok<*STOUT>
%--------------------------------------------------------------------------
% Solving L1 minimization problem directly via CVX toolbox.
%
% The program aims to solve the L1 minimization problem of the form
% 
%        min_x \mu ||x||_1 + ||Ax-b||_inf
%
% Input:
%         A --- The matrix apperaed in optimization problem 
%         b --- The vector appeared in optimization problem
%        mu --- L1 regularization term
%      opts --- Options structure with field(s)
%               solver: CVX internal solver, 0 -- Mosek, 1 -- Gurobi
%
% Output:
%         x --- The optimal point founded by algorithm
%       out --- Miscellaneous information during the computation
%
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% setup options for the solver

if ~isfield(opts,'solver');              opts.solver = 1; end

%--------------------------------------------------------------------------
% copy value to parameters

solver = opts.solver;

%--------------------------------------------------------------------------
% initial preparation

[m,n] = size(A);

%--------------------------------------------------------------------------
% start computation

if(solver==0)
    cvx_begin quiet
        cvx_solver mosek
        variable x(n)
        minimize(mu*norm(x,1)+norm(A*x-b,inf))
    cvx_end
else
    cvx_begin quiet
        cvx_solver gurobi
        variable x(n)
        minimize(mu*norm(x,1)+norm(A*x-b,inf))
    cvx_end
end

%--------------------------------------------------------------------------
% finish and output

out.time = cvx_cputime;
out.value = cvx_optval;
out.tol = cvx_slvtol;
out.status = cvx_status;

end
