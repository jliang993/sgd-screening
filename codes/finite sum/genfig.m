set(0,'DefaultFigureWindowStyle','docked')

clearvars
profile clear
profile on
set(groot,'defaultLineLineWidth',1.5);

% addpath libsvm
addpath('data');
strF = {'colon', 'leu', 'duke', 'gisette', 'arcene', 'dexter', 'dorothea', 'rcv1'};

% problem type
type= 'lasso';
% type= 'slr';

% maximum number of iteration
maxits = 1e7;
clf;
%% load and rescale data
whichh = 1; % change this from 1 to 5

for i_file = 1:8
    if strcmp(type, 'lasso')
        filename = ['LASSO_', strF{i_file}, '_w51.mat'];
        sS = [2; 2; 2; 2; 2; 2; 2; 2;];
    else
        filename = ['SLR_', strF{i_file}, '_w51.mat'];
        sS = [2; 2; 3; 2; 5; 10; 2; 4;];
    end
    load(filename);
    ss = sS(i_file);
    %% SAGA
    beta_saga = output_saga.betasol;
    
    n = numel(beta_saga);
    %%
    beta_sgd = output_sgd.betasol;
    its_sgd = output_sgd.its;
    ek_sgd = output_sgd.ek;
    dk_sgd = output_sgd.dk;
    sk_sgd = output_sgd.sk;
    tk_sgd = output_sgd.tk;
    %% SGD full-screen
    beta_fs = output_fs.betasol;
    its_fs = output_fs.its;
    ek_fs = output_fs.ek;
    dk_fs = output_fs.dk;
    sk_fs = output_fs.sk;
    tk_fs = output_fs.tk;
    %% SGD online-screen
    beta_os = output_os.betasol;
    its_os = output_os.its;
    ek_os = output_os.ek;
    dk_os = output_os.dk;
    sk_os = output_os.sk;
    tk_os = output_os.tk;
    
    jj = 4;
    para_sgd.w = 0.61;
    
    [its_sgd; its_fs; its_os]
    %% output....
    axesFontSize = 6;
    resolution = 256; % output resolution
    output_size = resolution *[38, 16]; % output size
    
    if whichh == 1
        %
        % %%%% %%%%%% %%%%%% %%%%%% %%%%%%
        figure(1022);
        set(gca,'DefaultTextFontSize',16);
        set(0,'DefaultAxesFontSize', axesFontSize);
        set(gcf,'paperunits','centimeters','paperposition',[-3.0 0.25 output_size/resolution]);
        set(gcf,'papersize',output_size/resolution-[5.95 0.2]);
        set(gcf,'color','w');
        
        subplot(2,4,i_file)
        
        % figure(101); clf;
        p1 = semilogy(tk_sgd, dk_sgd, 'k', 'linewidth',1.25);
        hold on;
        p2 = semilogy(tk_fs, dk_fs, 'b', 'linewidth',1.25);
        p3 = semilogy(tk_os, dk_os, 'r', 'linewidth',1.25);
        
        axis([0, max([tk_fs(:);tk_os(:)])*1.618,...
            min([dk_os(:);dk_fs(:);dk_sgd(:)])/1.1, max([dk_os(:);dk_fs(:);dk_sgd(:)])]);
        
        set(gca,'FontSize', 10);
        title([strF{i_file}, ', \lambda = \lambda_{max}/', num2str(ss)], 'FontWeight', 'normal');
        
        %%%
        ylb = ylabel({'$\|\beta_{t}-\beta^\star\|$'}, 'FontSize', 16,...
            'FontAngle', 'normal', 'Interpreter', 'latex');
        set(ylb, 'Units', 'Normalized', 'Position', [-0.165, 0.5, 0]);
        xlb = xlabel({'time (s)'}, 'FontSize', 16, 'Interpreter', 'latex');
        set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
        %%%
        if i_file==8
            lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',3);
            legend('boxoff');
            set(lg, 'Location', 'NorthEast');
            set(lg, 'FontSize', 13);
            pos = [0.45, -0.1125, .125, .25];
            set(lg, 'Position', pos);
            
            if strcmp(type, 'lasso')
                pdfname = sprintf('LASSO_dk.pdf');
            else
                pdfname = sprintf('SLR_dk.pdf');
            end
            
            print(pdfname, '-dpdf');
        end
    end
    %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
    if whichh == 2
        figure(101)
        set(gca,'DefaultTextFontSize',16);
        set(0,'DefaultAxesFontSize', axesFontSize);
        set(gcf,'paperunits','centimeters','paperposition',[-3.0 0.25 output_size/resolution]);
        set(gcf,'papersize',output_size/resolution-[5.95 0.2]);
        set(gcf,'color','w');
        
        subplot(2,4,i_file)
        
        % figure(101); clf;
        p1 = plot(tk_sgd, 'k', 'linewidth',1.5);
        hold on;
        p2 = plot(tk_fs, 'b', 'linewidth',1.5);
        p3 = plot(tk_os, 'r', 'linewidth',1.5);
        
        axis([0, numel(tk_os), 0, tk_sgd(end)]);
        
        set(gca,'FontSize', 10);
        title([strF{i_file}, ', \lambda = \lambda_{max}/', num2str(ss)], 'FontWeight', 'normal');
        
        ylb = ylabel({'$time~(s)$'},...
            'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 16);
        set(ylb, 'Units', 'Normalized', 'Position', [-0.155, 0.5, 0]);
        xlb = xlabel({'number~of~T'}, 'FontSize', 16, 'Interpreter', 'latex');
        set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
        
        if i_file==8
            lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',3);
            legend('boxoff');
            set(lg, 'Location', 'NorthWest');
            set(lg, 'FontSize', 13);
            pos = [0.45, -0.11, .125, .25];
            set(lg, 'Position', pos);
            
            
            if strcmp(type, 'lasso')
                pdfname = sprintf('LASSO_time.pdf');
            else
                pdfname = sprintf('SLR_time.pdf');
            end
            print(pdfname, '-dpdf');
        end
    end
    %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
    if whichh == 3
        figure(103);
        set(gca,'DefaultTextFontSize',16);
        set(0,'DefaultAxesFontSize', axesFontSize);
        set(gcf,'paperunits','centimeters','paperposition',[-3.0 0.25 output_size/resolution]);
        set(gcf,'papersize',output_size/resolution-[5.95 0.2]);
        set(gcf,'color','w');
        
        subplot(2,4,i_file)
        
        % figure(101); clf;
        p1 = semilogy(sk_sgd, 'k', 'linewidth',1.25);
        hold on;
        p2 = semilogy(sk_fs, 'b', 'linewidth',1.25);
        p3 = semilogy(sk_os, 'r', 'linewidth',1.25);
        
        axis([1, numel(sk_os), min(min(sk_fs)/10, min(sk_os)/10), max(sk_os)]);
        
        set(gca,'FontSize', 10);
        title([strF{i_file}, ', \lambda = \lambda_{max}/', num2str(ss)], 'FontWeight', 'normal');
        
        ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 16,...
            'FontAngle', 'normal', 'Interpreter', 'latex');
        set(ylb, 'Units', 'Normalized', 'Position', [-0.155, 0.5, 0]);
        xlb = xlabel({'number~ of~T'}, 'FontSize', 16, 'Interpreter', 'latex');
        set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
        
        if i_file==8
            lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',3);
            legend('boxoff');
            set(lg, 'Location', 'Best');
            set(lg, 'FontSize', 13);
            pos = [0.45, -0.11, .125, .25];
            set(lg, 'Position', pos);
            
            
            if strcmp(type, 'lasso')
                pdfname = sprintf('LASSO_support.pdf');
            else
                pdfname = sprintf('SLR_support.pdf');
            end
            print(pdfname, '-dpdf');
        end
    end
    %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
    if whichh == 4
        figure(104);
        set(gca,'DefaultTextFontSize',16);
        set(0,'DefaultAxesFontSize', axesFontSize);
        set(gcf,'paperunits','centimeters','paperposition',[-3.0 0.25 output_size/resolution]);
        set(gcf,'papersize',output_size/resolution-[5.95 0.2]);
        set(gcf,'color','w');
        
        subplot(2,4,i_file); %clf;
        
        % figure(101); clf;
        p1 = semilogy(abs(beta_saga), 'o', 'Color',[0.1,0.1,0.99], 'markersize', 12, 'LineWidth', 1.25);
        hold on;
        p2 = semilogy(abs(beta_sgd), 's', 'Color',[0.99,0.1,0.1], 'markersize', 11, 'LineWidth', 1.25);
        p3 = semilogy(abs(beta_fs), 'd', 'Color',[0.1,0.2,0.2], 'markersize', 8, 'LineWidth', 1.25);
        p4 = semilogy(abs(beta_os), '.', 'Color',[0.1,0.9,0.1], 'markersize', 16, 'LineWidth', 1.25);
        
        axis([1, n, 0, 2*max(abs(beta_saga))]);
        
        set(gca,'FontSize', 10);
        title([strF{i_file}, ', \lambda = \lambda_{max}/', num2str(ss)], 'FontWeight', 'normal');
        
        ylb = ylabel({'$|\beta_{i}|$'}, 'FontSize', 16,...
            'FontAngle', 'normal', 'Interpreter', 'latex');
        set(ylb, 'Units', 'Normalized', 'Position', [-0.155, 0.5, 0]);
        xlb = xlabel({'$i$'}, 'FontSize', 16,...
            'FontAngle', 'normal', 'Interpreter', 'latex');
        set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
        
        if i_file==8
            lg = legend([p1, p2, p3, p4], 'SAGA', 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',4);
            legend('boxoff');
            set(lg, 'Location', 'SouthEast');
            set(lg, 'FontSize', 13);
            pos = [0.45, -0.11, .125, .25];
            set(lg, 'Position', pos);
            
            if strcmp(type, 'lasso')
                pdfname = sprintf('LASSO_solution.pdf');
            else
                pdfname = sprintf('SLR_solution.pdf');
            end
            print(pdfname, '-dpdf');
        end
        
        % pdfname = sprintf('%s_%s_jj%d_solution.png', type, filename, jj);
        % print(pdfname, '-dpng');
    end
    %% %%%% %%%%%% %%%%%% %%%%%% %%%%%%
    if whichh == 5
        linewidth = 1;
        
        axesFontSize = 6;
        labelFontSize = 11;
        legendFontSize = 8;
        
        output_size = resolution *[42, 16]; % output size
        
        
        fig = figure(106);
        left_color = [0 0 0];
        right_color = [1 0 0];
        set(gca,'DefaultTextFontSize',16);
        set(0,'DefaultAxesFontSize', axesFontSize);
        set(gcf,'paperunits','centimeters','paperposition',[-3.95 0.25 output_size/resolution]);
        set(gcf,'papersize',output_size/resolution-[6.5 0.2]);
        set(gcf,'color','w');
        
        subplot(2,4,i_file)
        
        %%%%%%% Left
        yyaxis left
        p1 = semilogy(sk_sgd, 'k-', 'linewidth',1.25);
        hold on;
        p2 = semilogy(sk_fs, 'b-', 'linewidth',1.25);
        p3 = semilogy(sk_os, 'r-', 'linewidth',1.25);
        
        axis([1, numel(sk_os), min(sk_os)/10, max(sk_os)]);
        
        set(gca,'FontSize', 10);
        title([strF{i_file}, ', \lambda = \lambda_{max}/', num2str(ss)], 'FontWeight', 'normal');
        
        ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 16,...
            'FontAngle', 'normal', 'Interpreter', 'latex');
        set(ylb, 'Units', 'Normalized', 'Position', [-0.10, 0.5, 0]);
        xlb = xlabel({'number~ of~T'}, 'FontSize', 16, 'Interpreter', 'latex');
        set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
        
        %%%%%%% Right
        yyaxis right
        % figure(101); clf;
        p1_ = plot(tk_sgd, 'k--d', 'linewidth',1.25, 'MarkerIndices', 1:round(numel(tk_sgd)/7):numel(tk_sgd),'MarkerSize',5);
        hold on;
        p2_ = plot(tk_fs, 'b--o', 'linewidth',1.25, 'MarkerIndices', 1:round(numel(tk_fs)/8):numel(tk_fs),'MarkerSize',5);
        p3_ = plot(tk_os, 'r--s', 'linewidth',1.25, 'MarkerIndices', 1:round(numel(tk_os)/9):numel(tk_os),'MarkerSize',5);
        
        axis([0, numel(tk_os), 0, tk_sgd(end)]);
        
        p1 = semilogy(0,0, 'k-', 'linewidth',1.25);
        p2 = semilogy(0,0, 'b-', 'linewidth',1.25);
        p3 = semilogy(0,0, 'r-', 'linewidth',1.25);
        p4 = semilogy(0,0, 'color',[1,1,1], 'linewidth',1.25);
        
        % set(gca,'FontSize', 12);
        
        ylb = ylabel({'$time~(s)$'},...
            'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 18);
        set(ylb, 'Units', 'Normalized', 'Position', [1.1, 0.5, 0]);
        xlb = xlabel({'number~ of~T'}, 'FontSize', 16, 'Interpreter', 'latex');
        set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.1125, 0]);
        
        % %
        if i_file==8
            lg = legend([p1,p2,p3, p4, p1_,p2_,p3_], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD',...
                '         ', ...
                'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',7);
            legend('boxoff');
            set(lg, 'Location', 'West');
            set(lg, 'FontSize', 13);
            pos = [0.45, -0.11, .125, .25];
            set(lg, 'Position', pos);
            
            if strcmp(type, 'lasso')
                pdfname = sprintf('LASSO_support_time.pdf');
            else
                pdfname = sprintf('SLR_support_time.pdf');
            end
            print(pdfname, '-dpdf');
        end
    end
end