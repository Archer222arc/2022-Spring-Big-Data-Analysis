%--------------------------------------------------------------------------
% Compare prformance of algorithms on randomized SVD
%
% The following algorithms are available:
%
%      1. Linear S_value SVD
%      2. prototype SVD
% 
% We test two algorithms with different parameters and compare their
% performance
%
%--------------------------------------------------------------------------
%
% Author: Xiang Meng
% Version 1.0
% Last revision 2020.5
%
%--------------------------------------------------------------------------
% data prepatation

% clc;
% clear;

m = 4096;
n = 4096;
t = 1;
k_list = [5,10,15,20];
k_len = length(k_list);
[A, Ut, Vt, St] = svd_gen(m,n,t,2);
% S_value = zeros(k_len,6);
% err_Sigma = zeros(k_len,6);
% err_U = zeros(k_len,6);
% err_V = zeros(k_len,6);
tic;
[~,~,~] = svds(A,20);
S_value_matlab = toc;
S_value = cell(k_len,6);

%--------------------------------------------------------------------------
% compare performance

for i=1:k_len
    opts = [];
    k = k_list(i);
    grid = 1:k;
    Vnorm = norm(Vt(:,1:k),'fro');
    Unorm = norm(Ut(:,1:k),'fro');
    Snorm = norm(St(1:k),'fro');
    
    %linear
    tic;
    [U, V, d] = svd_Lineartime(A,k,2*k,opts);
    time(i,1) = toc;
    err_U(i,1) = compare_dist(U,Ut(:,1:k))/Unorm;
    err_V(i,1) = compare_dist(V,Vt(:,1:k))/Vnorm;
    err_sigma(i,1) = norm(d-St(1:k),'fro')/Snorm;
    S_value{i,1} = d;
    
    tic;
    [U, V, d] = svd_Lineartime(A,k,10*k,opts);
    time(i,2) = toc;
    err_U(i,2) = compare_dist(U,Ut(:,1:k))/Unorm;
    err_V(i,2) = compare_dist(V,Vt(:,1:k))/Vnorm;
    err_sigma(i,2) = norm(d-St(1:k),'fro')/Snorm;
    S_value{i,2} = d;
    
    tic;
    [U, V, d] = svd_Lineartime(A,k,50*k,opts);
    time(i,3) = toc;
    err_U(i,3) = compare_dist(U,Ut(:,1:k))/Unorm;
    err_V(i,3) = compare_dist(V,Vt(:,1:k))/Vnorm;
    err_sigma(i,3) = norm(d-St(1:k),'fro')/Snorm;
    S_value{i,3} = d;
    
    %prototype
    tic;
    [U, V, d] = svd_prototype(A,k,0,opts);
    time(i,4) = toc;
    err_U(i,4) = compare_dist(U,Ut(:,1:k))/Unorm;
    err_V(i,4) = compare_dist(V,Vt(:,1:k))/Vnorm;
    err_sigma(i,4) = norm(d-St(1:k),'fro')/Snorm;
    S_value{i,4} = d;
    
    tic;
    [U, V, d] = svd_prototype(A,k,1,opts);
    time(i,5) = toc;
    err_U(i,5) = compare_dist(U,Ut(:,1:k))/Unorm;
    err_V(i,5) = compare_dist(V,Vt(:,1:k))/Vnorm;
    err_sigma(i,5) = norm(d-St(1:k),'fro')/Snorm;
    S_value{i,5} = d;
    
    tic;
    [U, V, d] = svd_prototype(A,k,2,opts);
    time(i,6) = toc;
    err_U(i,6) = compare_dist(U,Ut(:,1:k))/Unorm;
    err_V(i,6) = compare_dist(V,Vt(:,1:k))/Vnorm;
    err_sigma(i,6) = norm(d-St(1:k),'fro')/Snorm;
    S_value{i,6} = d;
    
    tic;
    [~,~,~] = svds(A,k);
    time(i,7)=toc;
    
    % plot
%     fs = 10;
%     dir="./fig";
%     figure;
%     plot(grid, St(1:k),'-','LineWidth',2,'Color', [76, 153, 0]/255);         hold on;
%     plot(grid, S_value{i,1},'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 
%     plot(grid, S_value{i,2}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
%     plot(grid, S_value{i,3}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
%     plot(grid, S_value{i,4}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;
%     plot(grid, S_value{i,5}, ':', 'LineWidth',2,'Color', [0, 0, 0]/255);    hold on;
%     plot(grid, S_value{i,6}, '-.x', 'LineWidth',2,'Color', [76, 184, 76]/255);    hold on;
% 
%     ax1 = gca;
%     set(ax1,'FontSize',fs);
%     xlabel(ax1,'The x-th largest singular value','FontName','Arial','FontSize',fs);
%     ylabel(ax1,'\sigma','FontName','Arial','FontSize',fs);
%     title("Dataset 2, distribution "+string(t)+", k = "+string(k));
%     legend('Baseline','linear,c=2k','linear,c=10k','linear,c=50k','proto,q=0','proto,q=1','proto,q=2');
%     saveas(gcf, 'fig/dataset2_k'+string(k)+'_t'+string(t)+".png")
end

%--------------------------------------------------------------------------
% print comparison result

%% print comparison result
label.title1 = ["Algorithm","baseline","Linear time","Prototype randomized"];
label.length1 = [1,1,3,3];
label.title2 = ["-","$c=2k$","$c=10k$","$c=50k$","$q=0$","$q=1$","$q=2$"];
label.col = ["$k=5$","$k=10$","$k=15$","$k=20$"];
opt.H = "None";
% time
opt.caption = "CPU time for Linear time SVD and Prototype randomized SVD with different parameters and $k$";
opt.label = "time2"+string(t);
opt.filename = "./table/dataset2_time_t="+string(t);
maketable(time,label,opt);

label.title1 = ["Algorithm","Linear time","Prototype randomized"];
label.length1 = [1,3,3];
label.title2 = ["$c=2k$","$c=10k$","$c=50k$","$q=0$","$q=1$","$q=2$"];
label.col = ["$k=5$","$k=10$","$k=15$","$k=20$"];


% sigma err
opt.caption = "Relative error of eigenvalue for Linear time SVD and Prototype randomized SVD with different parameters and $k$ on dataset 2 with distribution "+string(t);
opt.label = "sigma2"+string(t);
opt.filename = "./table/dataset2_sigma_t="+string(t);
maketable(err_sigma,label,opt);

% U err
opt.caption = "Relative $\\|\\cdot\\|_F$ error of $U$ for Linear time SVD and Prototype randomized SVD with different parameters and $k$ on dataset 2 with distribution "+string(t);
opt.label = "U2"+string(t);
opt.filename = "./table/dataset2_U_t="+string(t);
maketable(err_U/100,label,opt);

% U err
opt.caption = "Relative $\\|\\cdot\\|_F$ error of $V$ for Linear time SVD and Prototype randomized SVD with different parameters and $k$ on dataset 2 with distribution "+string(t);
opt.label = "V2"+string(t);
opt.filename = "./table/dataset2_V_t="+string(t);
maketable(err_V/100,label,opt);

function dis = compare_dist(U1,U2)
[~,k] = size(U1);
dis = 0;
for i=1:k
    dis = dis+min(norm(U1(:,i)-U2(:,i),'fro')^2,norm(U1(:,i)+U2(:,i),'fro')^2);
end
dis = sqrt(dis);
end
