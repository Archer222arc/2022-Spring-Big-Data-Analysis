for i = 1:num
fprintf("Mosek & %3.2e & %3.2e &- & - & - & %3.2e\\\\\\hline\n",time_mosek(i),vopt_mosek(i),norm_mosek(i));
fprintf("Gurobi & %3.2e & %3.2e & - & - & %3.2e & %3.2e\\\\\\hline\n",time_gurobi(i),vopt_gurobi(i),merr_gurobi(i),norm_gurobi(i));
fprintf("ALM with ISTA & %3.2e & %3.2e & %3.2e & %3.2e &%3.2e & %3.2e \\\\\\hline\n",time_ALM1(i),vopt_ALM1(i),consvio_ALM1(i),optgap_ALM1(i),merr_ALM1(i),norm_ALM1(i));
fprintf("ALM with FISTA & %3.2e & %3.2e & %3.2e & %3.2e &%3.2e & %3.2e \\\\\\hline\n",time_ALM(i),vopt_ALM(i),consvio_ALM(i),optgap_ALM(i),merr_ALM(i),norm_ALM(i));
fprintf("ADMM & %3.2e & %3.2e & %3.2e & %3.2e&%3.2e & %3.2e  \\\\\\hline\n",time_ADMM(i),vopt_ADMM(i),consvio_ADMM(i),optgap_ADMM(i),merr_ADMM(i),norm_ADMM(i));
end
