set(0,'DefaultFigureWindowStyle','docked');
clear all;
% clc;
%% synthetic
% dimensin of the problem
% if n is too small, then online screening wont provide CPU time saving
n = 1e4;

% sparse vector
xsol = 4* sign(randn(n,1));

idx = randperm(n);
xsol(idx(1e1:end)) = 0;

% we use m samples to obtain an estimate of the regularization parameter
% and step-size coefficient
m = 1e4;
X = 2*rand(n, m) - 1;
y = (X')*xsol + 1*randn(m,1);

% max(g) returns the largest Lip. constant of sampled functions
g = diag(X'*X);
gammai = 1/ max(g);

% regularization parameter
lambda = norm(X*y, 'inf')/2/m * 1;
%% parameters
% tolerance
para.tol = 1e-10;

% every r iterations, we decrease the step-soze
para.r = 1e4;

% every S steps, we do sampling
para.S = 2e3;
para.xsol = xsol;
para.sig = 1;
para.samp = @(m, n) 2*rand(n, m) - 1;

% how often to record information
para.R = 5e3;

% how often to apply screening
para.T = 5e2;

% regularization parameter and step-size scaling
para.lam = lambda;
para.gammai = gammai;

% % initialisation
% para.X = X;
% para.y = y;

seed = 121;

% max number of iterations
maxits = 5e6 /1;
para.maxits = maxits;

cc = 0.5;
para.maxits2 = cc* maxits;
para.maxits1 = (1-cc)* maxits; % warm start iterations

% starting point
para.x0 = 1e-2*randn(n, 1);
%% SGD
disp('>>>>>>>> Running SGD ...');

tic
[x, output] = func_proxSGD_online(para, seed);
time_sgd = toc;

fprintf('\n');
%% SGD witn online screening
disp('>>>>>>>> Running Online Screening ...');

para.w = .6;
para.n = n;

% default we apply pruning
para.prune = 1;
% threshold for pruning
para.thresh = 3;

tic;
[x_os, output_os] = func_proxSGD_online_screen(para,seed);
time_screen = toc;

fprintf('\n');
%%
tic;
[x_os2, output_os2] = func_proxSGD_online_screen_fullsample(para,seed);
time_screen2 = toc;

fprintf('\n\n');
%%
% CPU time of two schemes
fprintf('CPU-time(s): \n    SGD \t OS-SGD \t OS-SGD\n    %.1f \t %.1f \t\t %.1f\n\n',...
    time_sgd, time_screen, time_screen2);

% output objective function value.
fprintf('Function value: \n    SGD \t OS-SGD \t OS-SGD\n    %.3f \t %.3f \t %.3f\n\n',...
    1/2/m *norm(x'*X-y')^2 + para.lam*norm(x),...
    1/2/m *norm(x_os'*X-y')^2 + para.lam*norm(x_os),...
    1/2/m *norm(x_os'*X-y')^2 + para.lam*norm(x_os2));

fprintf('\n');
%%
beta_os = output_os.betasol;
beta_os = x;

K = 20;
T = 1e3;

G = 0;
expZ = 0;

for i=1:K
    X = 2*rand(n, T) - 1;
    y = (X')*xsol + para.sig*randn(T, 1);

    expZ = expZ + X*(X'*beta_os - y) /T /lambda;

    for j=1:T
        fj = y(j);
        Xjt = X(:, j);

        vj = Xjt' * beta_os - fj;
        gi = vj* Xjt;

        G = max(0, max(abs(gi)));
    end
end

G = G*10;
expZ = expZ /K;
%%
T = 1e3;
X = 2*rand(n, T) - 1;
y = (X')*xsol + para.sig*randn(T, 1);

hatZ = X*(X'*beta_os - y) /T /lambda;

d = n;
e = 1e-2;

T = 20*1e4;

p = 2*d*exp(-2*T*e^2/G);

% hatZ = zeros(n, 1);
% for j=1:T
%     fj = y(j);
%     Xjt = X(:, j);
% 
%     vj = Xjt' * beta_os - fj;
%     hatZ = hatZ - vj* Xjt;
% end
% % hatZ = hatZ /T /lambda;
%% save data
if 0
    save([sprintf('toy_dim_%d_cc_%d.mat', n, 10*cc)], ...
        'x', 'x_os', 'output', 'output_os');
end
%% plot figures
if 1
    axesFontSize = 6;
    resolution = 108; % output resolution
    output_size = resolution *[16, 12]; % output size

    %%%%%%%%%%%%%%%%%%%% time over epochs
    figure(101), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

    % figure(101); clf;
    p1 = semilogy(output.tk, output.sk2, 'k', 'linewidth',1.5);
    hold on;
    
    p3 = semilogy(output_os2.tk, output_os2.sk2, 'b--', 'linewidth',1.25);
    p2 = semilogy(output_os.tk, output_os.sk2, 'r', 'linewidth',1.5);

    axis([0, max(output.tk(end), output_os.tk(end)), 1, n*1.1]);

    set(gca,'FontSize', 10);
    grid on;

    %%%
    ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.09, 0.5, 0]);
    xlb = xlabel({'time (s)'}, 'FontSize', 16, 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);


    lg = legend([p1, p2, p3], 'Prox-SGD', 'OS-Prox-SGD-1', 'OS-Prox-SGD-2', 'NumColumns',1);
    legend('boxoff');
    set(lg, 'Location', 'SouthWest');
    set(lg, 'FontSize', 13);
    % pos = [0.45, -0.1125, .125, .25];
    % set(lg, 'Position', pos);

    pdfname = sprintf('toy_sk.pdf');
    print(pdfname, '-dpdf');
    %% %%%%%%%%%%%%%%%%% time over epochs
    figure(102), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

    % figure(101); clf;
    p1 = semilogy(output.tk(1:20:end), output.ek(1:20:end), 'k', 'linewidth',1.25);
    hold on;
    p2 = semilogy(output_os.tk(1:20:end), output_os.ek(1:20:end), 'r', 'linewidth',1.25);

    p3 = semilogy(output_os2.tk(1:20:end), output_os2.ek(1:20:end), 'b--', 'linewidth',1.25);

    axis([0, max(output.tk(end), output_os.tk(end)), min(output_os.ek), max(output_os.ek)]);

    set(gca,'FontSize', 10);
    grid on;

    %%%
    ylb = ylabel({'$\|\beta_{t}-\beta_{t-1}\|$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.09, 0.5, 0]);
    xlb = xlabel({'time (s)'}, 'FontSize', 16, 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);


    lg = legend([p1, p2, p3], 'Prox-SGD', 'OS-Prox-SGD-1', 'OS-Prox-SGD-2', 'NumColumns',1);
    legend('boxoff');
    set(lg, 'Location', 'SouthWest');
    set(lg, 'FontSize', 13);
    % pos = [0.45, -0.1125, .125, .25];
    % set(lg, 'Position', pos);

    pdfname = sprintf('toy_ek.pdf');
    print(pdfname, '-dpdf');
    %% %%%%%%%%%%%%%%%%%% support size over epochs
    figure(103), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.1 -0.0 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

    p1 = semilogy(output.sk2, 'k', 'linewidth',1.25);
    hold on;
    p2 = semilogy(output_os.sk2, 'r', 'linewidth',1.25);

    p3 = semilogy(output_os2.sk2, 'b-', 'linewidth',1.25);

    axis([1, numel(output_os.sk2), min(output_os.sk2)/10, max(output_os.sk2)*1.1]);

    set(gca,'FontSize', 12);
    grid on;
    ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 22,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
    xlb = xlabel({'$\#~ of~T$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);


    lg = legend([p1, p2, p3], 'Prox-SGD', 'OS-Prox-SGD-1', 'OS-Prox-SGD-2', 'NumColumns',1);
    legend('boxoff');
    set(lg, 'Location', 'SouthWest');
    set(lg, 'FontSize', 11);

    pdfname = sprintf('toy_support.pdf');
    print(pdfname, '-dpdf');
    %% %%%%%%%%%%%%%%%%%% outputs of different SGDs
    figure(104), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.1 -0.0 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

    p1 = semilogy(abs(x), 'o', 'Color',[0.99,0.1,0.1], 'markersize', 8, 'linewidth',1.5);
    hold on;
    p2 = semilogy(abs(x_os), '.', 'Color',[0.1,0.8,0.1], 'markersize', 15, 'linewidth',1.5);

    p3 = semilogy(abs(x_os2), 'd', 'Color',[0.1,0.1,0.9], 'markersize', 7, 'linewidth',1.25);

    axis([1, n, 0, 2*max(abs(x))]);

    set(gca,'FontSize', 12);

    ylb = ylabel({'$|\beta_{i}|$'}, 'FontSize', 24,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
    xlb = xlabel({'$i$'}, 'FontSize', 18,...
        'FontAngle', 'normal', 'Interpreter' , 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);

    lg = legend([p1, p2, p3], 'Prox-SGD', 'OS-Prox-SGD-1', 'OS-Prox-SGD-2', 'NumColumns',1);
    set(lg, 'Location', 'SouthEast');
    set(lg, 'FontSize', 10);

    pdfname = sprintf('toy_solution.pdf');
    print(pdfname, '-dpdf');
    %% %%%%%%%%%%%%%%  support vs time
    linewidth = 1;

    axesFontSize = 6;
    labelFontSize = 11;
    legendFontSize = 8;

    figure(105), clf;
    set(gca,'DefaultTextFontSize',18);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.175 0.45 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[0.05 -0.25]);

    left_color = [0 0 0];
    right_color = [1 0 0];

    set(gcf,'color','w');

    %%%%%%% Left
    yyaxis left
    p1 = semilogy(output.sk2, 'k-', 'linewidth',1.5);
    hold on;
    p2 = semilogy(output_os.sk2, 'r-', 'linewidth',1.5);

    p3 = semilogy(output_os2.sk2, 'b-', 'linewidth',1.5);

    axis([1, numel(output_os.sk2), min(output_os.sk2)/10, max(output_os.sk2)*1.1]);

    set(gca,'FontSize', 12);


    ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 18,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
    xlb = xlabel({'$\#~ of~T$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.09, 0]);

    % lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',2);
    % set(lg, 'Location', 'SouthEast');
    % set(lg, 'FontSize', 10);

    %%%%%%% Right
    yyaxis right
    % figure(101); clf;
    p1_ = plot(output.tk, 'k--d', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(output.tk)/8):numel(output.tk),'MarkerSize',10);
    hold on;
    p2_ = plot(output_os.tk, 'r--s', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(output_os.tk)/9):numel(output_os.tk),'MarkerSize',10);

    p3_ = plot(output_os2.tk, 'b--o', 'linewidth',1.5, 'MarkerIndices', 1:round(numel(output_os2.tk)/9):numel(output_os2.tk),'MarkerSize',10);

    axis([0, numel(output_os.tk), 0, output.tk(end)]);

    p13 = semilogy(0,0, '.', 'color', [1,1,1], 'linewidth',1.5);
    p23 = semilogy(0,0, '.', 'color', [1,1,1], 'linewidth',1.5);
    p33 = semilogy(0,0, '.', 'color', [1,1,1], 'linewidth',1.5);

    % set(gca,'FontSize', 12);

    ylb = ylabel({'$time~(s)$'},...
        'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 18);
    set(ylb, 'Units', 'Normalized', 'Position', [1.07, 0.5, 0]);
    xlb = xlabel({'$\#~ of~T$'}, 'FontSize', 14,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.09, 0]);


    % %'OS-Prox-SGD-1', 'OS-Prox-SGD-2'
    lg = legend([p1,p2,p3, p1_,p2_,p3_], 'Prox-SGD', 'OS-Prox-SGD-1', 'OS-Prox-SGD-2', ...
        'Prox-SGD', 'OS-Prox-SGD-1', 'OS-Prox-SGD-2', 'NumColumns',1);
    legend('boxoff');
    set(lg, 'Location', 'NorthWest');
    set(lg, 'FontSize', 10);
%     pos = [0.45, -0.125, .125, .25];
%     set(lg, 'Position', pos);

    pdfname = sprintf('toy_ssupport_time.pdf');
    print(pdfname, '-dpdf');
end