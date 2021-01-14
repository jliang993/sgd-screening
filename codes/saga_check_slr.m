set(0,'DefaultFigureWindowStyle','docked')

clearvars
% clc
profile clear
profile on
set(groot,'defaultLineLineWidth',1.5);

% addpath libsvm
addpath(['..', filesep, 'data']);
%% problem type
% type= 'lasso';
type = 'logreg';
%% load and scale data
strF = {'colon-cancer', 'leukemia', 'duke-breast-cancer',...
    'gisette', 'arcene', 'dexter',...
    'dorothea', 'rcv1', 'real-sim'};

for i_file = 9 %[1, 6:9]
    filename = strF{i_file};
    class_name = [filename, '_sample.mat'];
    feature_name = [filename, '_label.mat'];
    
    fprintf('\n>>>>>>>>>> %s...\n', filename);
    
    load(class_name);
    load(feature_name);
    
    h = full(h);
    
    % rescale the data
    for j=1:size(h,2)
        h(:,j) = rescale(h(:,j), -1, 1);
    end
    
    %%%%% parameters
    % disp(filename)
    [m, n] = size(h);
    
    A = h;
    At = A';
    f = l;
    
    gammai = 1e10;
    for i=1:m
        Ai = At(:, i);
        gammai = min(gammai, 1/norm(Ai'*Ai));
    end
    
    seed = randi(314159);
    rng(seed);
    rnd_idx = randi(m, max(3e2*m, 3e7)+10,1);
    %% lambda
    f = sign(f);
    lam_max = max(abs(A'*ones(m, 1))) /2;
    
    ss = 3;
    if i_file == 5; ss = 5; end
    if i_file == 6; ss = 10; end
    
    if i_file == 8; ss = 18; end
    if i_file == 9; ss = 50; end
    
    lam = lam_max /ss;
    %% SAGA
    disp('Running SAGA ...');
    
    para_saga.m = m;
    para_saga.n = n;
    
    para_saga.f = f;
    para_saga.At = At;
    
    para_saga.seed = seed;
    
    para_saga.tol = 5e-11;
    
    para_saga.gamma = gammai;
    para_saga.lam = lam /m;
    
    para_saga.maxits = max(3e2*m, 2e7) + 1; % !
    para_saga.type = type;
    
    tic;
    [xsol, output_xsol] = func_SAGA_SLR(para_saga, rnd_idx);
    t_xsol = toc;
    
    fprintf('          CPU-time: %.1fs... xsol\n', t_xsol);
    
    sum(abs(xsol))
    %%
    v = zeros(m, 1);
    g = 0;
    for i=1:m
        Ajt = At(:, i);
        v(i) = -f(i)/(exp(f(i)*Ajt'*xsol)+1);
        
        g = g + Ajt*v(i);
    end
    
    g = g /lam;
    
    % u = A * x;
    % v = - f ./ (exp(f.*u) + 1);
    % g = At * v;
    % v = A*x - f;
    
    
    %%%%%% %%%%%% %%%%%% %%%%%% %%%%%%
    linewidth = 1;
    
    axesFontSize = 6;
    labelFontSize = 11;
    legendFontSize = 8;
    
    resolution = 108; % output resolution
    output_size = resolution *[16, 12]; % output size
    
    fig = figure(100+i_file); clf;
    left_color = [0 0 0];
    right_color = [1 0 0];
    set(fig,'defaultAxesColorOrder',[left_color; right_color]);
    set(0,'DefaultAxesFontSize', axesFontSize);
    set(gcf,'paperunits','centimeters','paperposition',[-0.75 -0.10 output_size/resolution]);
    set(gcf,'papersize',output_size/resolution-[0.75 0.5]);
    
    %%%%%%% Left
    yyaxis left
    semilogy(abs(g), 'k.', 'markersize',16);
    set(gca,'FontSize', 8)
    
    xlb = xlabel({'\vspace{-1.0mm}';'$i$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.04, 0]);
    
    ylb = ylabel({'$|g^\star|$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [-0.05, 0.5, 0]);
    
    axis([1, n, 5e-1, 1]);
    
    %%%%%%% Right
    yyaxis right
    ll = floor(numel(output_xsol.ek)/n) + 1;
    kk = numel(output_xsol.ek(1:ll:end));
    idx = (1:kk) /kk * n;
    semilogy(idx, output_xsol.ek(1:ll:end), 'b', 'linewidth', 1.0);
    
    hold on
    stem(abs(sign(xsol)), 'ro', 'markersize',10, 'linewidth', 1.5);
    stem(abs(xsol)/max(abs(xsol))/10, 'md', 'markersize',10, 'linewidth', 1.5);
    
    grid on
    
    ylb = ylabel({'$|\mathrm{sign}(x^\star)|$'}, 'FontSize', 16,...
        'FontAngle', 'normal', 'Interpreter', 'latex');
    set(ylb, 'Units', 'Normalized', 'Position', [1.05, 0.5, 0]);
    
    axis([1, n, 0, 1]);
    
    
    %%%%%%%%%%%%%%%
    pdfname = sprintf('%s_%s_scale%d_saga_check.pdf', type, filename, ss);
    print(pdfname, '-dpdf');
end


