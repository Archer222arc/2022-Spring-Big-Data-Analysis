% script for plotting all results
% please run this script after sto_test.m
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


%% covtype
fs = 10;

% for i=1:lengn
for i = 1:lengn
    lg = 1:length(f_adam_c{i});
    figure;
    ax1 = gca;
%     plot(lg, f_ada_c{i}, '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
%     plot(lg, f_adam_c{i}, '-.+', 'LineWidth',2,'Color', [76, 153, 0]/255);    hold on;
%     plot(lg, f_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
%     plot(lg, f_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    
%     plot(lg, f_rms_c{i},'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 
    semilogy(lg, f_ada_c{i}, '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    semilogy(lg, f_adam_c{i}, '-.+', 'LineWidth',2,'Color', [76, 153, 0]/255);    hold on;
    semilogy(lg, f_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    semilogy(lg, f_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    
    semilogy(lg, f_rms_c{i},'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 

    xlabel('epoch');
    set(ax1,'FontSize',fs);
    ylabel('train error');
    title("lambda="+string(lambda_list(i)));
    legend('Adagrad','Adam','Momentum','Line search','RMSprop');
    saveas(gcf,'fig/fval_c'+string(i)+'.png');
    
    figure;
    ax2 = gca;
    semilogy(lg, g_ada_c{i}, '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    semilogy(lg, g_adam_c{i}, '-.+', 'LineWidth',2,'Color', [76, 153, 0]/255);    hold on;
    semilogy(lg, g_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    semilogy(lg, g_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    
    semilogy(lg, g_rms_c{i},'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 

%     semilogy(lg, f_ada_c{i}, '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
%     semilogy(lg, f_adam_c{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
%     semilogy(lg, f_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
%     semilogy(lg, f_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;   
    set(ax2,'FontSize',fs);
    xlabel('epoch');
    ylabel('gradient norm');
    title("lambda="+string(lambda_list(i)));
    legend('Adagrad','Adam','Momentum','Line search','RMSprop');
    saveas(gcf,'fig/gnorm_c'+string(i)+'.png');
    
    figure;
    ax3 = gca;
%     plot(lg, e_ada_c{i}, '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
%     plot(lg, e_adam_c{i}, '-.+', 'LineWidth',2,'Color', [76, 153, 0]/255);    hold on;
%     plot(lg, e_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
%     plot(lg, e_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on; 
%     plot(lg, e_rms_c{i},'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 
    semilogy(lg, e_ada_c{i}, '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    semilogy(lg, e_adam_c{i}, '-.+', 'LineWidth',2,'Color', [76, 153, 0]/255);    hold on;
    semilogy(lg, e_mom_c{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    semilogy(lg, e_sgd_c{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on; 
    semilogy(lg, e_rms_c{i},'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 
    set(ax3,'FontSize',fs);
    xlabel('epoch');
    ylabel('test error');
    title("lambda="+string(lambda_list(i)));
    legend('Adagrad','Adam','Momentum','Line search','RMSprop');
    saveas(gcf,'fig/err_c'+string(i)+'.png');
end
%% gisette
fs = 10;

% for i=1:lengn
for i = 1:1
    lg = 1:length(f_adam_c{i});
    figure;
    ax1 = gca;
    plot(lg, f_ada_g{i}, '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg, f_adam_g{i}, '-.+', 'LineWidth',2,'Color', [76, 153, 0]/255);    hold on;
    plot(lg, f_mom_g{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg, f_sgd_g{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    
    plot(lg, f_rms_g{i},'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 

    xlabel('epoch');
    set(ax1,'FontSize',fs);
    ylabel('train error');
    title("lambda="+string(lambda_list(i)));
    legend('Adagrad','Adam','Momentum','Line search','RMSprop');
    saveas(gcf,'fig/fval_g'+string(i)+'.png');
    
    figure;
    ax2 = gca;
    plot(lg(1:30), g_ada_g{i}(2:31), '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg(1:30), g_adam_g{i}(2:31), '-.+', 'LineWidth',2,'Color', [76, 153, 0]/255);    hold on;
    plot(lg(1:30), g_mom_g{i}(2:31), '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg(1:30), g_sgd_g{i}(2:31), '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    
    plot(lg(1:30), g_rms_g{i}(2:31),'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 

    xlabel('epoch');
    set(ax2,'FontSize',fs);
    xlabel('epoch');
    ylabel('gradient norm');
    title("lambda="+string(lambda_list(i)));
    legend('Adagrad','Adam','Momentum','Line search','RMSprop');
    saveas(gcf,'fig/gnorm_g'+string(i)+'.png');
    
    figure;
    ax3 = gca;
    plot(lg, e_ada_g{i}, '-.o', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
    plot(lg, e_adam_g{i}, '-.+', 'LineWidth',2,'Color', [76, 153, 0]/255);    hold on;
    plot(lg, e_mom_g{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
    plot(lg, e_sgd_g{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    
    plot(lg, e_rms_g{i},'-.d','LineWidth',2,'Color', [255,0,255]/255);       hold on; 

    xlabel('epoch');
    set(ax3,'FontSize',fs);
    xlabel('epoch');
    ylabel('test error');
    title("lambda="+string(lambda_list(i)));
    legend('Adagrad','Adam','Momentum','Line search','RMSprop');
    saveas(gcf,'fig/err_g'+string(i)+'.png');
end
% fs = 10;
% 
% for i=1:lengn
%     lg = 1:length(f_adam_m{i});
%     figure;
%     ax1 = gca;
%     plot(lg, f_adam_m{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
%     plot(lg, f_mom_m{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
%     plot(lg, f_sgd_m{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;    xlabel('epoch');
%     set(ax1,'FontSize',fs);
%     ylabel('L value');
%     title("lambda="+string(lambda_list(i)));
%     legend('Adam','Momentum','Line search');
%     saveas(gcf,'fig/fval_m'+string(i)+'.png');
%     
%     figure;
%     ax2 = gca;
%     plot(lg, g_adam_m{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
%     plot(lg, g_mom_m{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
%     plot(lg, g_sgd_m{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;
%     set(ax2,'FontSize',fs);
%     xlabel('epoch');
%     ylabel('gradient norm');
%     title("lambda="+string(lambda_list(i)));
%     legend('Adam','Momentum','Line search');
%     saveas(gcf,'fig/gnorm_m'+string(i)+'.png');
%     
%     figure;
%     ax3 = gca;
%     plot(lg, e_adam_m{i}, '-.+', 'LineWidth',2,'Color', [255, 0, 0]/255);    hold on;
%     plot(lg, e_mom_m{i}, '-.<', 'LineWidth',2,'Color', [128, 128, 0]/255);    hold on;
%     plot(lg, e_sgd_m{i}, '-.', 'LineWidth',2,'Color', [0, 153, 76]/255);    hold on;
%     set(ax3,'FontSize',fs);
%     xlabel('epoch');
%     ylabel('train error');
%     title("lambda="+string(lambda_list(i)));
%     legend('Adam','Momentum','Line search');
%     saveas(gcf,'fig/err_m'+string(i)+'.png');
% end
