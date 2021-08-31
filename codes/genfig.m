set(0,'DefaultFigureWindowStyle','docked')

clearvars
profile clear
profile on
set(groot,'defaultLineLineWidth',1.5);

% addpath libsvm
addpath('data');

strF = {'colon-cancer', 'leukemia', 'duke-breast-cancer', 'gisette',...
    'arcene', 'dexter', 'dorothea', 'rcv1'};

% problem type
type= 'lasso';

% maximum number of iteration
maxits = 1e7;


clf;

%% load and rescale data
i_file = 1;

for i_file = 1:8% 1:8
    
    filename = ['LASSO_', strF{i_file}, '.mat'];
    load(filename);
    
    %     filename = strF{i_file};
    %     class_name = [filename, '_sample.mat'];
    %     feature_name = [filename, '_label.mat'];
    %
    %     load(class_name);
    %     load(feature_name);
    %
    %     h = full(h);
    %
    %     A = h;
    %     At = A';
    %     f = l;
    %
    %     [m, n] = size(h);
    %% SAGA
    
    output_xsol = output_saga;
    xsol = output_saga.xsol;
    
    n = numel(xsol);
    %%
    x_sgd = output_sgd.x_sgd;
    its_sgd = output_sgd.its_sgd;
    ek_sgd = output_sgd.ek_sgd;
    dk_sgd = output_sgd.dk_sgd;
    sk_sgd = output_sgd.sk_sgd;
    tk_sgd = output_sgd.tk_sgd;
    %% SGD full-screen
    
    x_fs = output_sgd_fs.x_fs;
    its_fs = output_sgd_fs.its_fs;
    ek_fs = output_sgd_fs.ek_fs;
    dk_fs = output_sgd_fs.dk_fs;
    sk_fs = output_sgd_fs.sk_fs;
    tk_fs = output_sgd_fs.tk_fs;
    %% SGD online-screen
    x_os = output_sgd_os.x_os;
    its_os = output_sgd_os.its_os;
    ek_os = output_sgd_os.ek_os;
    dk_os = output_sgd_os.dk_os;
    sk_os = output_sgd_os.sk_os;
    tk_os = output_sgd_os.tk_os;
    
    jj = 4;
    para_sgd.w = 0.51;
    %% output....
    axesFontSize = 6;
    resolution = 108; % output resolution
    output_size = resolution *[20, 16]; % output size
    
    [its_sgd; its_fs; its_os]
    %%
    %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
%     figure(1022);
%     set(gca,'DefaultTextFontSize',18);
%     set(0,'DefaultAxesFontSize', axesFontSize);
%     set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
%     set(gcf,'papersize',output_size/resolution-[1.15 0.6]);
%     set(gcf,'color','w');
%     
%     
%     subplot(2,4,i_file)
%     
%     % figure(101); clf;
%     p1 = semilogy(tk_sgd, dk_sgd, 'k', 'linewidth',1.25);
%     hold on;
%     p2 = semilogy(tk_fs, dk_fs, 'm', 'linewidth',1.25);
%     p3 = semilogy(tk_os, dk_os, 'r', 'linewidth',1.25);
%     
%     axis([0, tk_sgd(end), min(dk_os), max(dk_os)]);
%     
%     set(gca,'FontSize', 12);
%     
%     title(strF{i_file}, 'FontWeight', 'normal');
%     
%     %%%
%     ylb = ylabel({'$\|\beta_{t}-\beta^\star\|$'}, 'FontSize', 20,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(ylb, 'Units', 'Normalized', 'Position', [-0.165, 0.5, 0]);
%     xlb = xlabel({'$time (s)$'}, 'FontSize', 18,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
%     %%%
%     if i_file==8
%         lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',3);
%         legend('boxoff');
%         set(lg, 'Location', 'NorthEast');
%         set(lg, 'FontSize', 14);
%         pos = [0.45, -0.11, .125, .25];
%         set(lg, 'Position', pos);
%         
%         pdfname = sprintf('LASSO_dk.pdf');
%         % print(pdfname, '-dpdf');
%     end
%     
%     %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
%     figure(101)
%     set(gca,'DefaultTextFontSize',18);
%     set(0,'DefaultAxesFontSize', axesFontSize);
%     set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
%     set(gcf,'papersize',output_size/resolution-[1.15 0.6]);
%     set(gcf,'color','w');
%     
%     subplot(2,4,i_file)
%     
%     % figure(101); clf;
%     p1 = plot(tk_sgd, 'k', 'linewidth',1.5);
%     hold on;
%     p2 = plot(tk_fs, 'm', 'linewidth',1.5);
%     p3 = plot(tk_os, 'r', 'linewidth',1.5);
%     
%     axis([0, numel(tk_fs), 0, tk_sgd(end)]);
%     
%     set(gca,'FontSize', 12);
%     
%     title(strF{i_file}, 'FontWeight', 'normal');
%     
%     ylb = ylabel({'$time~(s)$'},...
%         'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 20);
%     set(ylb, 'Units', 'Normalized', 'Position', [-0.155, 0.5, 0]);
%     xlb = xlabel({'$\#~ of~epochs$'}, 'FontSize', 18,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
%     
%     if i_file==8
%         lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',3);
%         legend('boxoff');
%         set(lg, 'Location', 'NorthWest');
%         set(lg, 'FontSize', 14);
%         pos = [0.45, -0.11, .125, .25];
%         set(lg, 'Position', pos);
%         
%         pdfname = sprintf('LASSO_time.pdf');
%         % print(pdfname, '-dpdf');
%     end
%     
%     
%     %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
%     figure(103);
%     set(gca,'DefaultTextFontSize',18);
%     set(0,'DefaultAxesFontSize', axesFontSize);
%     set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
%     set(gcf,'papersize',output_size/resolution-[1.15 0.6]);
%     set(gcf,'color','w');
%     
%     subplot(2,4,i_file)
%     
%     % figure(101); clf;
%     p1 = semilogy(sk_sgd, 'k', 'linewidth',1.25);
%     hold on;
%     p2 = semilogy(sk_fs, 'm', 'linewidth',1.25);
%     p3 = semilogy(sk_os, 'r', 'linewidth',1.25);
%     
%     axis([1, numel(sk_fs), min(sk_os)/10, max(sk_os)]);
%     
%     set(gca,'FontSize', 12);
%     title(strF{i_file}, 'FontWeight', 'normal');
%     
%     ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 20,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(ylb, 'Units', 'Normalized', 'Position', [-0.155, 0.5, 0]);
%     xlb = xlabel({'$\#~ of~epochs$'}, 'FontSize', 18,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
%     
%     if i_file==8
%         lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',3);
%         legend('boxoff');
%         set(lg, 'Location', 'Best');
%         set(lg, 'FontSize', 15);
%         pos = [0.45, -0.11, .125, .25];
%         set(lg, 'Position', pos);
%         
%         pdfname = sprintf('LASSO_support.pdf');
%         % print(pdfname, '-dpdf');
%     end
%     
%     %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
%     figure(104);
%     set(gca,'DefaultTextFontSize',18);
%     set(0,'DefaultAxesFontSize', axesFontSize);
%     set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
%     set(gcf,'papersize',output_size/resolution-[1.15 0.6]);
%     
%     set(gcf,'color','w');
%     
%     subplot(2,4,i_file)
%     
%     % figure(101); clf;
%     p1 = semilogy(abs(xsol), 'o', 'Color',[0.1,0.1,0.99], 'markersize', 10);
%     hold on;
%     p2 = semilogy(abs(x_sgd), 'd', 'Color',[0.99,0.1,0.1], 'markersize', 8);
%     p3 = semilogy(abs(x_fs), 'ks', 'LineWidth', 1.5, 'markersize', 8);
%     p4 = semilogy(abs(x_os), '.', 'Color',[0.1,0.9,0.1], 'markersize', 16);
%     
%     axis([1, n, 0, 2*max(abs(xsol))]);
%     
%     set(gca,'FontSize', 12);
%     title(strF{i_file}, 'FontWeight', 'normal');
%     
%     ylb = ylabel({'$|\beta_{i}|$'}, 'FontSize', 20,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(ylb, 'Units', 'Normalized', 'Position', [-0.155, 0.5, 0]);
%     xlb = xlabel({'$i$'}, 'FontSize', 18,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
%     
%     if i_file==8
%         lg = legend([p1, p2, p3, p4], 'SAGA', 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',4);
%         legend('boxoff');
%         set(lg, 'Location', 'SouthEast');
%         set(lg, 'FontSize', 15);
%         pos = [0.45, -0.11, .125, .25];
%         set(lg, 'Position', pos);
%         
%         pdfname = sprintf('LASSO_solution.pdf');
%         % print(pdfname, '-dpdf');
%     end
%     
%     % pdfname = sprintf('%s_%s_jj%d_solution.png', type, filename, jj);
%     % print(pdfname, '-dpng');
%     
%     %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
%     % %     v = zeros(m, 1);
%     % %     g = 0;
%     % %     for i=1:m
%     % %         Ajt = At(:, i);
%     % %         v(i) = Ajt'*xsol - f(i);
%     % %
%     % %         g = g + Ajt*v(i);
%     % %     end
%     % %
%     % %     g = g /lam;
%     % %
%     % %     axesFontSize = 6;
%     % %
%     % %     resolution = 108; % output resolution
%     % %     output_size = resolution *[16, 12]; % output size
%     % %
%     % %     fig = figure(105); clf;
%     % %     left_color = [0 0 0];
%     % %     right_color = [1 0 0];
%     % %     set(fig,'defaultAxesColorOrder',[left_color; right_color]);
%     % %     set(0,'DefaultAxesFontSize', axesFontSize);
%     % %     set(gcf,'paperunits','centimeters','paperposition',[-0.75 -0.10 output_size/resolution]);
%     % %     set(gcf,'papersize',output_size/resolution-[0.75 0.5]);
%     % %
%     % %     %%%%%%% Left
%     % %     yyaxis left
%     % %     semilogy(abs(g), 'k.', 'markersize',16);
%     % %     set(gca,'FontSize', 8)
%     % %
%     % %     xlb = xlabel({'\vspace{-1.0mm}';'$i$'}, 'FontSize', 16,...
%     % %         'FontAngle', 'normal', 'Interpreter', 'latex');
%     % %     set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.04, 0]);
%     % %
%     % %     ylb = ylabel({'$|g^\star|$'}, 'FontSize', 16,...
%     % %         'FontAngle', 'normal', 'Interpreter', 'latex');
%     % %     set(ylb, 'Units', 'Normalized', 'Position', [-0.05, 0.5, 0]);
%     % %
%     % %     axis([1, n, 5e-1, 1]);
%     % %
%     % %     %%%%%%% Right
%     % %     yyaxis right
%     % %     ll = round(numel(output_xsol.ek)/n);
%     % %     semilogy(output_xsol.ek(1:ll:end), 'b', 'linewidth', 1.25);
%     % %
%     % %     hold on
%     % %     stem(abs(sign(xsol)), 'ro', 'markersize',10, 'linewidth', 1.5);
%     % %     stem(abs(xsol)/max(abs(xsol))/10, 'md', 'markersize',10, 'linewidth', 1.5);
%     % %
%     % %     grid on
%     % %
%     % %     ylb = ylabel({'$|\mathrm{sign}(x^\star)|$'}, 'FontSize', 16,...
%     % %         'FontAngle', 'normal', 'Interpreter', 'latex');
%     % %     set(ylb, 'Units', 'Normalized', 'Position', [1.05, 0.5, 0]);
%     % %
%     % %     axis([1, n, 0, 1]);
%     % %
%     % %
%     % %     %%%%%%%%%%%%%%%
%     % %     pdfname = sprintf('%s_%s_scale%d_w%02d_maxits%de6_jj%d_xsol.pdf',...
%     % %         type, filename, ss, 100*para_sgd.w, maxits/1e6, jj);
%     % %     print(pdfname, '-dpdf');
%     %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
%     linewidth = 1;
%     
%     axesFontSize = 6;
%     labelFontSize = 11;
%     legendFontSize = 8;
%     
%     resolution = 108; % output resolution
%     output_size = resolution *[16, 12]; % output size
%     
%     fig = figure(106);
%     left_color = [0 0 0];
%     right_color = [1 0 0];
%     set(fig,'defaultAxesColorOrder',[left_color; right_color]);
%     set(0,'DefaultAxesFontSize', axesFontSize);
%     set(gcf,'paperunits','centimeters','paperposition',[-0.5 -0.10 output_size/resolution]);
%     set(gcf,'papersize',output_size/resolution-[0.7 0.5]);
%     
%     set(gcf,'color','w');
%     
%     subplot(2,4,i_file)
%     
%     %%%%%%% Left
%     yyaxis left
%     p1 = semilogy(sk_sgd, 'k-', 'linewidth',1.5);
%     hold on;
%     p2 = semilogy(sk_fs, 'm-', 'linewidth',1.5);
%     p3 = semilogy(sk_os, 'r-', 'linewidth',1.5);
%     
%     axis([1, numel(sk_os), min(sk_os)/10, max(sk_os)]);
%     
%     set(gca,'FontSize', 12);
%     title(strF{i_file}, 'FontWeight', 'normal');
%     
%     ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 18,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(ylb, 'Units', 'Normalized', 'Position', [-0.125, 0.5, 0]);
%     xlb = xlabel({'$\#~ of~epochs$'}, 'FontSize', 16,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
% 
%     %%%%%%% Right
%     yyaxis right
%     % figure(101); clf;
%     p1_ = plot(tk_sgd, 'k--d', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(tk_sgd)/8):numel(tk_sgd),'MarkerSize',10);
%     hold on;
%     p2_ = plot(tk_fs, 'm--o', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(tk_fs)/8):numel(tk_fs),'MarkerSize',10);
%     p3_ = plot(tk_os, 'r--s', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(tk_os)/8):numel(tk_os),'MarkerSize',10);
%     
%     axis([0, numel(tk_fs), 0, tk_sgd(end)]);
%     
%     p1 = semilogy(0,0, 'k-', 'linewidth',1.5);
%     p2 = semilogy(0,0, 'm-', 'linewidth',1.5);
%     p3 = semilogy(0,0, 'r-', 'linewidth',1.5);
%     p4 = semilogy(0,0, 'color',[1,1,1], 'linewidth',1.5);
%     
%     % set(gca,'FontSize', 12);
%     
%     ylb = ylabel({'$time~(s)$'},...
%         'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 18);
%     set(ylb, 'Units', 'Normalized', 'Position', [1.125, 0.5, 0]);
%     xlb = xlabel({'$\#~ of~epochs$'}, 'FontSize', 14,...
%         'FontAngle', 'normal', 'Interpreter', 'latex');
%     set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
%     
%     %%
%     if i_file==8
%         lg = legend([p1,p2,p3, p4, p1_,p2_,p3_], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD',...
%             '         ', ...
%             'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',7);
%         legend('boxoff');
%         set(lg, 'Location', 'West');
%         set(lg, 'FontSize', 15);
%         pos = [0.45, -0.11, .125, .25];
%         set(lg, 'Position', pos);
%         
%         
%         %%%%%%%%%%%%%%%
%         pdfname = sprintf('LASSO_support_time.pdf');
%         % print(pdfname, '-dpdf');
%     end
end