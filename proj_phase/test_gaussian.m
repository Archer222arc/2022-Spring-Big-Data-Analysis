%% main loop
n_list = [64,128,256];
m_list = [400,800,1200];
step_list = [0.02,0.05,0.1,0.2,0.3];
err_print = zeros(9,5);
iter = 3e3;
grid = 1:iter;
for i=1:length(n_list)
    for j = 1:length(m_list)
        n = n_list(i);
        m = m_list(i);
        [A,AT,y,x] = gen_gaussian(n,m);
        err = zeros(length(step_list),iter);
        for k=1:length(step_list)
            opts.maxstep = step_list(k);
            [z,out] = Wirtinger_flow(A,AT,y,x,opts);
            err(k,:) = out.err;
            err_print(3*i+j-3,k) = err(k,3000);
        end   
    %% semilogy
        fs = 10;
        dir="./fig";
        figure;
        semilogy(grid, err(1,:),'-','LineWidth',2,'Color', [76, 153, 0]/255);         hold on;
        semilogy(grid, err(2,:), '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
        semilogy(grid, err(3,:), '-.x', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
        semilogy(grid, err(4,:), '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;
        semilogy(grid, err(5,:), ':', 'LineWidth',2,'Color', [0, 0, 0]/255);    hold on;
        ax1 = gca;
        set(ax1,'FontSize',fs);
        xlabel('Number of iterations');
        ylabel('Relative L2 error');
        title('Gaussian dataset, n='+string(n)+', m='+string(m));
        legend('step size = '+string(step_list(1)),'step size = '+string(step_list(2)),'step size = '+string(step_list(3)),'step size = '+string(step_list(4)),'step size = '+string(step_list(5)));
        saveas(gcf, './fig/gaussian+'+string(i)+string(j)+'.png');
    end
end
%% make table
for i = 1 : 3
label.title1 = ["$\\mu$","$n = "+string(n_list(i))+" $"];
label.length1 = [5];
label.title2=[];
for j = 1 : length(step_list)
label.title2 = [label.title2,"$"+string(step_list(j))+"$"];
end
label.col = [];
for j = 1 : length(m_list)
label.col = [label.col,dollar(string(m_list(j)))];
end
opt.H = "None";
% time
if i == 3
    opt.caption = "Relative error after $3000$ iterations by Wirtinger Flow on dataset Gaussian";
end
if i == 1
opt.label = "gaussian";
end
opt.filename = "./table/dataset1"+string(i);
maketable(err_print(3*i-2:3*i,:),label,opt);
end
