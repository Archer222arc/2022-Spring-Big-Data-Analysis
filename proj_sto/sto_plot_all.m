% script for plotting all results
% please run this script after sto_test.m
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

fs = 10;

for i=1:lengn
    lg = 1:length(f_adam_m{i});
    figure;
    ax1 = gca;
    plot(lg, f_adam_m{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg, f_mom_m{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg, f_sgd_m{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    xlabel('epoch');
    set(ax1,'FontSize',fs);
    ylabel('L value');
    title("lambda="+string(lambda_list(i)));
    legend('Adam','Momentum','Line search');
    saveas(gcf,'fig/fval_m'+string(i)+'.png');
    
    figure;
    ax2 = gca;
    plot(lg, g_adam_m{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg, g_mom_m{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg, g_sgd_m{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;
    set(ax2,'FontSize',fs);
    xlabel('epoch');
    ylabel('gradient norm');
    title("lambda="+string(lambda_list(i)));
    legend('Adam','Momentum','Line search');
    saveas(gcf,'fig/gnorm_m'+string(i)+'.png');
    
    figure;
    ax3 = gca;
    plot(lg, e_adam_m{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg, e_mom_m{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg, e_sgd_m{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;
    set(ax3,'FontSize',fs);
    xlabel('epoch');
    ylabel('train error');
    title("lambda="+string(lambda_list(i)));
    legend('Adam','Momentum','Line search');
    saveas(gcf,'fig/err_m'+string(i)+'.png');
end
%% covtype
fs = 10;

% for i=1:lengn
for i = 3
    lg = 1:length(f_adam_c{i});
    figure;
    ax1 = gca;
    plot(lg, f_adam_c{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg, f_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg, f_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    xlabel('epoch');
    set(ax1,'FontSize',fs);
    ylabel('L value');
    title("lambda="+string(lambda_list(i)));
    legend('Adam','Momentum','Line search');
    saveas(gcf,'fig/fval_c'+string(i)+'.png');
    
    figure;
    ax2 = gca;
    plot(lg, g_adam_c{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg, g_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg, g_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;
    set(ax2,'FontSize',fs);
    xlabel('epoch');
    ylabel('gradient norm');
    title("lambda="+string(lambda_list(i)));
    legend('Adam','Momentum','Line search');
    saveas(gcf,'fig/gnorm_c'+string(i)+'.png');
    
    figure;
    ax3 = gca;
    plot(lg, e_adam_c{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg, e_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg, e_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;
    set(ax3,'FontSize',fs);
    xlabel('epoch');
    ylabel('train error');
    title("lambda="+string(lambda_list(i)));
    legend('Adam','Momentum','Line search');
    saveas(gcf,'fig/err_c'+string(i)+'.png');
end
% 
