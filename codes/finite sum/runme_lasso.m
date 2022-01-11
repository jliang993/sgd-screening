%%
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
%% load and rescale data
% file name
for i_file = 4

    % load file
    filename = strF{i_file};
    load(filename);

    % prepare data
    X0 = data.h;
    Xt0 = X0';
    f0 = data.l;
    Nm = sqrt(sum(abs(Xt0).^2, 2));
    Nm0 = Nm(:);

    %%%%% parameters
    disp(filename);
    [m, n] = size(X0);
    disp([m, n]);

    % one over max Lip. constant
    gammai = 1/ max(data.L);

    % max lambda allowed
    lam_max = max(abs(Xt0*f0)) /m;

    % chosen lambda
    ss = 2;
    lam = lam_max/ ss;

    % initial point
    beta0 = randn(n,1);
    beta0 = beta0 /norm(beta0);

    % generate samples for each step, to make sure the sampling of SGDs are the
    % same
    seed = randi(202110);
    rng(seed);
    rnd_idx = randi(m, 1e8,1);
    %% SAGA
    disp('>>>>>>>> Running SAGA ...');

    % dimension of problem
    para_saga.m = m;
    para_saga.n = n;

    % initial point
    para_saga.beta0 = beta0;

    % data
    para_saga.f = f0;
    para_saga.Xt = Xt0;
    para_saga.Nm0 = Nm0;

    % regularization parameter and Lip. constant
    para_saga.lam = lam;
    para_saga.gamma = gammai;

    % stopping criterion and max number of iteratons
    para_saga.tol = 1e-10;
    para_saga.maxits = 1e7;

    tic;
    [beta_saga, output_saga, betasol_saga] = func_SAGA_LASSO(para_saga, rnd_idx);
    t_xsol = toc;

    fprintf('      CPU-time: %.1fs... xsol\n\n', t_xsol);
    %% SGD setup
    % solution and initial point
    para_sgd.betasol = betasol_saga;
    para_sgd.beta0 = beta0;

    % period for recording and screening
    para_sgd.T = 4* m;

    % data
    para_sgd.f = f0;
    para_sgd.Xt = Xt0;

    % dimension of problem
    para_sgd.m = m;
    para_sgd.n = n;

    % regularization parameter
    para_sgd.lam = lam;

    % max number of iteratons
    maxits = max(3e6, 3e2*m);
    para_sgd.maxits = maxits;

    % step-size for sgd, staircase decay
    W = floor(m/2);
    p = ones(W, 1);
    q = (gammai ./ (1:ceil(2e8/W)).^0.55)';
    Gamma = kron(q, p);
    para_sgd.Gamma = Gamma;
    %% prox-SGD
    disp('>>>>>>>> Running SGD ...');

    tic;
    [beta_sgd, output_sgd, betasol_sgd] = func_proxSGD_LASSO(para_sgd, rnd_idx);
    t_sgd = toc;

    fprintf('      CPU-time: %.1fs...\n\n', t_sgd);

    % the maximum time allowed for screening
    para_sgd.maxtime = t_sgd;
    %% prox-SGD full-screen
    disp('>>>>>>>> Running SGD full-screening ...');

    para_sgd.Nm0 = Nm0;
    para_sgd.prune = 1; % this is for screening
    para_sgd.w = 0.51; % ergodic averaing primal iterates

    tic;
    [beta_fs, output_fs, betasol_fs] = func_proxSGD_LASSO_fullscreen(para_sgd, rnd_idx);
    t_fs = toc;

    fprintf('      CPU-time: %.1fs...\n\n', t_fs);
    %% prox-SGD online-screen
    disp('>>>>>>>> Running SGD online-screening ...');

    % exponent for computing weight, range of w is ]0.5, 1]
    % the smaller the w, the more aggressive the online screening
    para_sgd.w = 0.51;
    para_sgd.thresh = 5; % thresholding for screening

    tic;
    [beta_os, output_os, betasol_os] = func_proxSGD_LASSO_onlinescreen(para_sgd, rnd_idx);
    t_os = toc;

    fprintf('      CPU-time: %.1fs...\n\n', t_os);
    %% save data
    if 1
        save(sprintf('LASSO_%s_w%d.mat', filename, 100*para_sgd.w), ...
            'output_saga', 'output_sgd', 'output_fs', 'output_os');
    end
    %%
end
%% plot figures
if 0
    %% time over T
    axesFontSize = 6;
    resolution = 108; % output resolution
    output_size = resolution *[16, 12]; % output size

    %%%%%%%%%%%%%%%%%%%% time over epochs
    figure(101), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.1 -0.0 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

    p1 = plot(output_sgd.tk, 'k', 'linewidth',1.5);
    hold on;
    p2 = plot(output_fs.tk, 'b', 'linewidth',1.5);
    p3 = plot(output_os.tk, 'r', 'linewidth',1.5);

    grid on;
    axis([0, numel(output_fs.tk), 0, output_sgd.tk(end)]);

    set(gca,'FontSize', 12);

    ylb = ylabel({'$time~(s)$'},...
        'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 21);
    set(ylb, 'Units', 'Normalized', 'Position', [-0.075, 0.5, 0]);
    xlb = xlabel({'$\#~ of~T$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.06, 0]);

    lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',1);
    legend('boxoff');
    set(lg, 'Location', 'NorthWest');
    set(lg, 'FontSize', 12);

    pdfname = sprintf('%s_%s_w%d_time.pdf', type, filename, 100*para_sgd.w);
    print(pdfname, '-dpdf');
    %% error over time
    figure(102), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.1 -0.0 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

    p1 = semilogy(output_sgd.tk, output_sgd.dk, 'k', 'linewidth',1.25);
    hold on;
    p2 = semilogy(output_fs.tk, output_fs.dk, 'b', 'linewidth',1.25);
    p3 = semilogy(output_os.tk, output_os.dk, 'r', 'linewidth',1.25);

    % axis([0, output_sgd.tk(end), output_sgd.dk(end), output_sgd.dk(1)]);

    set(gca,'FontSize', 12);

    ylb = ylabel({'$\|\beta_{t}-\beta^\star\|$'}, 'FontSize', 22,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
    xlb = xlabel({'$time (s)$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.075, 0]);

    lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',1);
    legend('boxoff');
    set(lg, 'Location', 'NorthEast');
    set(lg, 'FontSize', 11);

    pdfname = sprintf('%s_%s_w%d_dk.pdf', type, filename, 100*para_sgd.w);
    print(pdfname, '-dpdf');
    %% support size over epochs
    figure(103), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.1 -0.0 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

    p1 = semilogy(output_sgd.sk, 'k', 'linewidth',1.25);
    hold on;
    p2 = semilogy(output_fs.sk, 'b', 'linewidth',1.25);
    p3 = semilogy(output_os.sk, 'r', 'linewidth',1.25);

    axis([1, numel(output_os.sk), min(output_os.sk)/10, 1.1*max(output_os.sk)]);

    set(gca,'FontSize', 12);

    ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 22,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
    xlb = xlabel({'$\#~ of~T$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);


    lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',1);
    legend('boxoff');
    set(lg, 'Location', 'Best');
    set(lg, 'FontSize', 11);

    pdfname = sprintf('%s_%s_w%d_support.pdf', type, filename, 100*para_sgd.w);
    print(pdfname, '-dpdf');
    %% outputs of different SGDs
    figure(104), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.1 -0.0 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

    p1 = semilogy(abs(betasol_saga), 'o', 'Color',[0.1,0.1,0.99], 'markersize', 10);
    hold on;
    p2 = semilogy(abs(betasol_sgd), 'd', 'Color',[0.99,0.1,0.1], 'markersize', 8);
    p3 = semilogy(abs(betasol_fs), 'ks', 'LineWidth', 1.5, 'markersize', 8);
    p4 = semilogy(abs(betasol_os), '.', 'Color',[0.1,0.9,0.1], 'markersize', 16);

    axis([1, n, 0, 2*max(abs(betasol_saga))]);

    set(gca,'FontSize', 12);

    ylb = ylabel({'$|\beta_{i}|$'}, 'FontSize', 24,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
    xlb = xlabel({'$i$'}, 'FontSize', 18,...
        'FontAngle', 'normal', 'Interpreter' , 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);

    lg = legend([p1, p2, p3, p4], 'SAGA', 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',2);
    set(lg, 'Location', 'SouthEast');
    set(lg, 'FontSize', 10);

    pdfname = sprintf('%s_%s_w%d_solution.pdf', type, filename, 100*para_sgd.w);
    print(pdfname, '-dpdf');
    %% support vs time
    linewidth = 1;
    axesFontSize = 6;
    labelFontSize = 11;
    legendFontSize = 8;

    figure(105), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.175 0.1 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[0.05 0.05]);

    left_color = [0 0 0];
    right_color = [1 0 0];

    set(gcf,'color','w');

    %%%%%%% Left
    yyaxis left
    semilogy(output_sgd.sk, 'k-', 'linewidth',1.5);
    hold on;
    semilogy(output_fs.sk, 'b-', 'linewidth',1.5);
    semilogy(output_os.sk, 'r-', 'linewidth',1.5);

    axis([1, numel(output_os.sk), min(output_os.sk)/10, max(output_os.sk)]);

    set(gca,'FontSize', 12);

    % ss = sS(i_file);
    ss = 2;
    title([strF{i_file}, ', \lambda = \lambda_{max}/', num2str(ss)], 'FontWeight', 'normal');

    ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 18,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
    xlb = xlabel({'$\#~ of~T$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.09, 0]);

    %%%%%%% Right
    yyaxis right
    p1_ = plot(output_sgd.tk, 'k--d', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(output_sgd.tk)/7):numel(output_sgd.tk),'MarkerSize',10);
    hold on;
    p2_ = plot(output_fs.tk, 'b--o', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(output_fs.tk)/8):numel(output_fs.tk),'MarkerSize',10);
    p3_ = plot(output_os.tk, 'r--s', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(output_os.tk)/9):numel(output_os.tk),'MarkerSize',10);

    axis([0, numel(output_fs.tk), 0, output_sgd.tk(end)]);

    p1 = semilogy(0,0, 'k-', 'linewidth',1.5);
    p2 = semilogy(0,0, 'b-', 'linewidth',1.5);
    p3 = semilogy(0,0, 'r-', 'linewidth',1.5);
    p4 = semilogy(0,0, 'color',[1,1,1], 'linewidth',1.5);

    ylb = ylabel({'$time~(s)$'},...
        'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 18);
    set(ylb, 'Units', 'Normalized', 'Position', [1.07, 0.5, 0]);
    xlb = xlabel({'$\#~ of~T$'}, 'FontSize', 14,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.09, 0]);

    pdfname = sprintf('%s_%s_w%d_support_time.pdf', type, filename, 100*para_sgd.w);
    print(pdfname, '-dpdf');
    %%
end