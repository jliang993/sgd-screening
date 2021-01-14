set(0,'DefaultFigureWindowStyle','docked')

clearvars
profile clear
profile on
set(groot,'defaultLineLineWidth',1.5);

% addpath libsvm
addpath('data');

strF = {'colon-cancer', 'leukemia', 'duke-breast-cancer',...
    'arcene', 'dexter', 'dorothea', 'rcv1'};

% problem type
type= 'lasso';

% maximum number of iteration
maxits = 1e7;
%% load and rescale data
i_file = 1;

filename = strF{i_file};
class_name = [filename, '_sample.mat'];
feature_name = [filename, '_label.mat'];

load(class_name);
load(feature_name);

h = full(h);

% rescale the data
for j=1:size(h,2)
    h(:,j) = rescale(h(:,j), -1, 1);
end

%%%%% parameters
disp(filename)
[m, n] = size(h);

disp([m, n]);

A = h;
At = A';
f = l;

gammai = 1e10;
for i=1:m
    Ai = At(:, i);
    gammai = min(gammai, 1/norm(Ai'*Ai));
end

% generate samples for each step, to make sure the sampling of SGDs are the
% same
seed = randi(314159);
rng(seed);
rnd_idx = randi(m, max(3e2*m, 3e7)+10,1);
%% lam_max
lam_max = max(abs(A'*f));

% scaling
ss = 2;
lam = lam_max/ ss;

%% SAGA
disp('>>>>>>>> Running SAGA ...');

para_saga.m = m;
para_saga.n = n;

para_saga.f = f;
para_saga.At = At;

% regularization parameter and Lip. constant
para_saga.gamma = gammai;
para_saga.lam = lam /m;

% stopping criterion and max number of iteratons
para_saga.tol = 5e-11;
para_saga.maxits = max(3e2*m, 2e7) + 1; % !

tic;
[xsol, output_xsol] = func_SAGA_LASSO(para_saga, rnd_idx);
t_xsol = toc;

fprintf('      CPU-time: %.1fs... xsol\n\n', t_xsol);
%% SGD setup
para_sgd.xsol = xsol;
para_sgd.type = type;

% data
para_sgd.A = A;
para_sgd.f = f;
para_sgd.At = A';

para_sgd.m = m;
para_sgd.n = n;

% regularization parameter
para_sgd.lam = lam;

para_sgd.tol = 1e-14;
para_sgd.maxits = maxits;

para_sgd.x0 = zeros(n,1);

para_sgd.prune = 1; % this is for screening

% step-size
Gamma = gammai/(m) ./ (ceil((1:1e7))).^0.51;
para_sgd.Gamma = Gamma;
%% SGD
disp('>>>>>>>> Running SGD ...');
para_sgd.w = 0.51;

tic;
[x_sgd, its_sgd, ek_sgd, dk_sgd, sk_sgd, tk_sgd] = ...
    func_proxSGD_LASSO(para_sgd, rnd_idx);
t_sgd = toc;

fprintf('      CPU-time: %.1fs...\n\n', t_sgd);

% the maximum time allowed for screening
para_sgd.maxtime = t_sgd;
%% SGD full-screen
disp('>>>>>>>> Running SGD full-screening ...');

% full screening is appled every T steps of iteraton
% result is sensitive to how often we reset the anchor point
para_sgd.T = m*4;

tic;
[x_sgd_fs, its_fs, ek_fs,dk_fs, sk_fs,tk_fs] = ...
    func_proxSGD_LASSO_fullscreen(para_sgd, rnd_idx);
t_sgd_fs = toc;

fprintf('      CPU-time: %.1fs...\n\n', t_sgd_fs);
%% SGD online-screen
disp('>>>>>>>> Running SGD online-screening ...');

jj = 4;
para_sgd.T = m* jj;

para_sgd.thresh = 10; % thresholding for screening
para_sgd.w = 0.51; % exponent for computing weight

tic;
[x_sgd_os, its_os, ek_os,dk_os, sk_os,tk_os] = ...
    func_proxSGD_LASSO_onlinescreen(para_sgd, rnd_idx);
t_sgd_os = toc;

fprintf('      CPU-time: %.1fs...\n\n', t_sgd_os);
%% output....
axesFontSize = 6;
resolution = 108; % output resolution
output_size = resolution *[16, 12]; % output size

%%%%%%%%%%%%%%%%%%%% time over epochs
figure(101), clf;
set(gca,'DefaultTextFontSize',18);
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

p1 = plot(tk_sgd, 'k', 'linewidth',1.5);
hold on;
p2 = plot(tk_fs, 'm', 'linewidth',1.5);
p3 = plot(tk_os, 'r', 'linewidth',1.5);

axis([0, numel(tk_os), 0, tk_sgd(end)]);

set(gca,'FontSize', 12);

ylb = ylabel({'$time~(s)$'},...
    'FontAngle', 'normal', 'Interpreter', 'latex', 'FontSize', 21);
set(ylb, 'Units', 'Normalized', 'Position', [-0.075, 0.5, 0]);
xlb = xlabel({'$\#~ of~epochs$'}, 'FontSize', 16,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.06, 0]);


lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',1);
legend('boxoff');
set(lg, 'Location', 'NorthWest');
set(lg, 'FontSize', 11);

pdfname = sprintf('%s_%s_scale%d_w%02d_maxits%de6_jj%d_time.pdf',...
    type, filename, ss, 100*para_sgd.w, maxits/1e6, jj);
print(pdfname, '-dpdf');

%%%%%%%%%%%%%%%%%%%% error over time
figure(102), clf;
set(gca,'DefaultTextFontSize',18);
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

p1 = semilogy(tk_sgd, dk_sgd, 'k', 'linewidth',1.25);
hold on;
p2 = semilogy(tk_fs, dk_fs, 'm', 'linewidth',1.25);
p3 = semilogy(tk_os, dk_os, 'r', 'linewidth',1.25);

axis([0, tk_sgd(end), dk_sgd(end), dk_sgd(1)]);

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

pdfname = sprintf('%s_%s_scale%d_w%02d_maxits%de6_jj%d_dk.pdf',...
    type, filename, ss, 100*para_sgd.w, maxits/1e6, jj);
print(pdfname, '-dpdf');

%%%%%%%%%%%%%%%%%%%% support size over epochs
figure(103), clf;
set(gca,'DefaultTextFontSize',18);
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

p1 = semilogy(sk_sgd, 'k', 'linewidth',1.25);
hold on;
p2 = semilogy(sk_fs, 'm', 'linewidth',1.25);
p3 = semilogy(sk_os, 'r', 'linewidth',1.25);

axis([1, numel(sk_os), min(sk_os)/10, max(sk_os)]);

set(gca,'FontSize', 12);

ylb = ylabel({'$\mathrm{supp}(\beta_{t})$'}, 'FontSize', 22,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
xlb = xlabel({'$\#~ of~epochs$'}, 'FontSize', 16,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);


lg = legend([p1, p2, p3], 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',1);
legend('boxoff');
set(lg, 'Location', 'Best');
set(lg, 'FontSize', 11);

pdfname = sprintf('%s_%s_scale%d_w%02d_maxits%de6_jj%d_support.pdf',...
    type, filename, ss, 100*para_sgd.w, maxits/1e6, jj);
print(pdfname, '-dpdf');

%%%%%%%%%%%%%%%%%%%% outputs of different SGDs
figure(104), clf;
set(gca,'DefaultTextFontSize',18);
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.175 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[1.15 0.6]);

p1 = semilogy(abs(xsol), 'o', 'Color',[0.1,0.1,0.99], 'markersize', 10);
hold on;
p2 = semilogy(abs(x_sgd), 'd', 'Color',[0.99,0.1,0.1], 'markersize', 8);
p3 = semilogy(abs(x_sgd_fs), 'ks', 'LineWidth', 1.5, 'markersize', 8);
p4 = semilogy(abs(x_sgd_os), '.', 'Color',[0.1,0.9,0.1], 'markersize', 16);

axis([1, n, 0, 2*max(abs(xsol))]);

set(gca,'FontSize', 12);

ylb = ylabel({'$|\beta_{i}|$'}, 'FontSize', 24,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.08, 0.5, 0]);
xlb = xlabel({'$i$'}, 'FontSize', 18,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.07, 0]);

lg = legend([p1, p2, p3, p4], 'SAGA', 'Prox-SGD', 'FS-Prox-SGD', 'OS-Prox-SGD', 'NumColumns',2);
set(lg, 'Location', 'SouthEast');
set(lg, 'FontSize', 10);

pdfname = sprintf('%s_%s_scale%d_w%02d_maxits%de6_jj%d_solution.pdf',...
    type, filename, ss, 100*para_sgd.w, maxits/1e6, jj);
print(pdfname, '-dpdf');
