function [betasol, output] = func_proxSGD_online_screen_fullsample(para, seed)
if seed>0
    rng(seed)
end

maxits = para.maxits;
maxits1 = para.maxits1;
maxits2 = para.maxits2;

S = para.S; % how often to sample
T = para.T; % how often to screen
R = para.R; % how often to record
r = para.r;

lam = para.lam;
samp = para.samp;

%initialisation
beta =  para.x0;
n = length(beta);

pbar = maxits/10;

betasol = para.xsol;

ek = zeros(maxits, 1);
tk = ek;
sk1 = ek;
sk2 = ek;
%% stage 1, plain sgd
tic
t = 1;
tt = 1;
while(t<=maxits1)
    if mod(t-1, S)==0
        % generate samples
        X  = samp(S+1, n);
        y = (X')*betasol + para.sig*randn(S+1,1);
    end
    
    beta_old = beta;
    
    j = mod(t, S) + 1;
    
    fj = y(j);
    Xjt = X(:, j);
    
    gamma = para.gammai/ (t/r + 1)^0.51;
    
    grd = Xjt'*beta - fj;
    tmpx = beta - gamma*grd * Xjt;
    
    beta = max(abs(tmpx)-lam*gamma, 0) .* sign(tmpx);
    
    %record
    if mod(t, R)==0
        tk(tt) = toc;
        ek(tt) = norm(beta - beta_old, 'fro');
        sk1(tt) = sum(abs(beta)>0);
        sk2(tt) = numel(beta);
        
        tt = tt+1;
    end

    if mod(t,pbar)==0
        disp(['    progress: ', num2str(t/pbar*10), '%...']);
    end
    
    t = t + 1;
end
toc;

%% stage 2: online screening
beta_firststage = beta;

Z = zeros(n,1); % dual certificate
Z_m = zeros(n,1); % previous dual certificate
cnorm = zeros(n,1); % norm of A(:,j) for each j
beta_a = beta; % primal anchor point
nmxa = norm(beta_a,1);

Pval = 0; %running primal value
Dval = 0; %running dual value
r_m = 0;
f_ = 0;
mfac = 0;

n_ = n;
nk = zeros(n, 1); % to record screened index using stochastic grad

% objects for recording old values in case we screen everything out..
beta_ = zeros(n,1);
beta_old_ = zeros(n,1);
Z_ = zeros(n,1);
nk_ = zeros(n,1);
cnorm_ = zeros(n,1);

lb = 5;
idx = (1:n)';

PP = zeros(1, maxits2);
DD = zeros(1, maxits2);

w = para.w;
prune = para.prune;
thresh = para.thresh;

kk = 1;
while(t<=maxits)
    beta_old = beta;
    
    %sample for SGD
    if mod(t-1, S)==0
        % generate samples
        X = samp(S+1, n); X = X(idx, :);
        
        y = (X')*betasol + para.sig*randn(S+1,1);
    end
    
    j = mod(t, S) + 1;
    
    fj = y(j);
    Xjt = X(:, j);
    
    gamma = para.gammai /((t-1)/r + 1)^0.51;

    % whos Xjt beta
    
    vj = Xjt' * beta - fj;
    
    % sgd step
    tmpx = beta - gamma*vj * Xjt;
    beta = max(abs(tmpx) - lam*gamma, 0) .* sign(tmpx);
    
    fac = 1/(t-maxits1)^w;
    
    %update primal dual gap
    va = Xjt' * beta_a - fj;
    
    % runing loss function value
    f_ = fac * 1/2 * fj^2 + (1 - fac) * f_;
    
    % primal and dual function values
    Pval = fac * ( 1/2 * (va)^2 + lam * nmxa ) + (1 - fac) * Pval;
    Dval = fac * - 1/2 * (vj^2 + 2*vj*fj) + (1 - fac) * Dval ;
    
    mfac = mfac * (1 - fac);
    
    %build screening certificate
    cnorm = fac * Xjt.^2 + (1 - fac) * cnorm;
    Z = - fac * vj * Xjt + (1 - fac) * Z; %update certificate
    
    % record
    if mod(t, R) == 0
        tk(tt) = toc;
        ek(tt) = norm(beta - beta_old, 'fro');
        sk1(tt) = sum(abs(beta)>0);
        sk2(tt) = numel(beta);
        
        PP(kk) = Pval;
        DD(kk) = Dval;
        
        kk = kk + 1;
        tt = tt + 1;
    end
    
    % apply screening
    if mod(t, floor(T/4)) == 0
        gap = Pval - Dval;
        rk = gap + max(0, norm(Z/lam/(1 - mfac),inf) - 1) * f_ ;
        rk = sqrt(2) * sqrt(rk + r_m * mfac)/lam;
        nk = nk + ((abs((Z + Z_m * mfac)/lam) + rk * sqrt(cnorm))< 1);
        
        beta_a = beta;
    end
    
    % reset every T iterations
    % pruning
    if prune && mod(t, T) == 0 && max(nk)>1*thresh && 1
        supp = (nk <= thresh);
        
        if sum(supp)<length(supp) && sum(supp)>10
            % disp('truncate')
            beta = beta(supp); %
            beta_old = beta_old(supp);
            Z = Z(supp);
            Z_m = Z_m(supp);
            idx = idx(supp);
            nk = nk(supp);
            cnorm = cnorm(supp);
            
            betasol = betasol(supp);
            X = X(supp, :);
            % n_ = numel(betasol);
        end
        
        %do the reset of anchor point and variables
        r_m = r_m * mfac + max(0, norm(Z/lam/(1 - mfac),inf) - 1) * f_;
        Z_m = Z + Z_m * mfac;
        Z = 0 * Z;
        f_ = 0;
        mfac = 1;
        beta_a = beta;
        nmxa = norm(beta,1);
        
        %%%%%%%%%%%%
        beta_(idx) = beta;
        beta_old_(idx) = beta_old;
        Z_(idx) = Z + Z_m * mfac;
        nk_(idx) = nk;
        cnorm_(idx) = cnorm;
    end

    % safety check to ensure no false removal
    if (mod(t, 1e6)==0 && numel(beta)>=lb) || t==maxits-1
        x = zeros(n,1);
        x(idx) = beta;

        T = 5e4;
        X_ = samp(T, n);
        y_ = (X_')*para.xsol + para.sig*randn(T,1);

        phi_a = X_*(X_'*x - y_) /T /lam;

        phi_a2 = phi_a/lam;
        phi_a2 = phi_a2/max(1,norm(phi_a2,inf));
        supp_c = ones(n,1); supp_c(idx) = 0;
        
        if max(abs(phi_a2.*supp_c)) < 0.99 || norm(phi_a/lam,inf)<=1+1e-5
            % disp('>>> good result...continue...');
        else
            disp('>>> bad exit...restart screening...')
            beta = beta_firststage + 1e-5*randn(n, 1);
            betasol = para.xsol;

            Z = zeros(n,1); % dual certificate
            Z_m = zeros(n,1); % previous dual certificate
            cnorm = zeros(n,1); % norm of A(:,j) for each j
            beta_a = beta; % primal anchor point
            nmxa = norm(beta_a,1);

            Pval = 0; %running primal value
            Dval = 0; %running dual value
            r_m = 0;
            f_ = 0;
            mfac = 0;

            % n_ = n;
            nk = zeros(n, 1); % to record screened index using stochastic grad

            % objects for recording old values in case we screen everything out..
            beta_ = zeros(n,1);
            beta_old_ = zeros(n,1);
            Z_ = zeros(n,1);
            nk_ = zeros(n,1);
            cnorm_ = zeros(n,1);

            idx = (1:n)';

            lb = lb *2;
            thresh = thresh + 4; % increase the threshold
            w = min(w+0.1, 0.99);

            t = maxits1;
        end
    end

    if mod(t,pbar)==0
        disp(['    progress: ', num2str(t/pbar*10), '%...']);
    end
    
    t = t + 1;
end

betasol = zeros(n,1);
betasol(idx) = beta;

output.betasol = betasol;

output.t = t;

output.nk = nk;

output.ek = ek(1:tt-1);
output.tk = tk(1:tt-1);
output.sk1 = sk1(1:tt-1);
output.sk2 = sk2(1:tt-1);

output.DD = DD(1:kk-1);
output.PP = PP(1:kk-1);