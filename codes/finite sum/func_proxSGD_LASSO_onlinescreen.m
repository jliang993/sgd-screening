function [beta, output, betasol, f_k,Pval_k] = func_proxSGD_LASSO_onlinescreen(para, rnd_idx)

% load data and parameters
f = para.f;
Xt_ = para.Xt;
Xt2_ = Xt_.^2;
Xt = Xt_;
Xt2 = Xt2_;

% period of screening and wether or not screening
T = para.T;
prune = para.prune;

% regularization parameter and precomputed step-size
lam = para.lam;
Gamma = para.Gamma;

% size of problem
m = para.m;
n = para.n;

% load data and parameters
maxits = para.maxits;
% the running time of fulling screening will not exceed that of pure SGD
maxtime = para.maxtime;

% for computing weight
w = para.w;
% threshold for screening
thresh = para.thresh;
% to record screened index using stochastic grad
nk = zeros(n, 1);

% starting point and solution of SAGA
betasol = para.betasol;
beta0 = para.beta0;
beta = beta0;
beta_a = beta0; % primal anchor point

% dual certificate
Z = zeros(n,1); % dual certificate
Z_m = zeros(n,1); % previous dual certificate
cnorm = zeros(n,1); % norm of A(:,j) for each j
nmxa = norm(beta_a,1); % norm of anchor point

% running primal dual function values
Pval = 0; % running primal value
Dval = 0; % running dual value
r_m = 0;
f_= 0; % loss function value, w.o. regularization

f_k = zeros(maxits, 1);
Pval_k = zeros(maxits, 1);

mfac = 0;

% objects for recording old values in case we screen everything out..
beta_ = zeros(n,1);
beta_old_ = zeros(n,1);
Z_ = zeros(n,1);
Z_m_ = zeros(n,1);
nk_ = zeros(n,1);
cnorm_ = zeros(n,1);

% support for screening
idx= (1:n)';
supp = ones(n, 1);

% recording iterates information
ek = zeros(maxits, 1);
dk = zeros(maxits, 1);
sk = zeros(maxits, 1);
tk = zeros(maxits, 1);

% primal and dual function values
DD = zeros(maxits, 1);
PP = zeros(maxits, 1);

% minimal dimension
lb = 30; %max(10, n/200);

tic
t = 1;
tt = 1;
while(t<maxits)
    beta_old = beta;
    
    %sample for SGD
    j =rnd_idx(t);
    Xjt = Xt(:,j);
    fj = f(j);
    
    % step-size
    gamma = Gamma(t);
    
    % stochastic gradient
    vj = Xjt'*beta - fj;
    sgrad = (vj*gamma) * Xjt;
    
    % stochastic gradient descent
    tmpx = beta - sgrad;
    % soft-thresholding, proximity of l1-norm
    beta = max(abs(tmpx)-lam*gamma, 0) .* sign(tmpx);
    
    if numel(beta)>lb
        % weight
        fac = 1/t^w;
        
        % anchor point
        va = Xjt'*beta_a - fj;
        % runing loss function value
        % f_ = fac * 1/2 * fj^2 + (1-fac)*f_;
        f_val = 1/2 * (va)^2 + lam* nmxa;
        f_ = fac* ( f_val ) + (1-fac) * f_;
        % primal and dual function values
        Pval = fac* ( f_val ) + (1-fac) * Pval;
        Dval = fac* -1/2*(vj^2+2*vj*fj) + (1-fac) * Dval ;

        f_k(t) = f_;
        Pval_k(t) = Pval;
        
        mfac = mfac*(1-fac);
        
        % build screening certificate
        cnorm = fac* Xt2(:,j) + (1-fac) * cnorm;
        Z = (-fac/gamma) *sgrad + (1-fac) * Z; % update certificate
        
        % apply screening
        if mod(t,T/4) == 0
            gap = Pval - Dval ;
            rk = gap + max(0, norm(Z/lam/(1-mfac),inf)-1)*f_; %min(f_, Pval);
            % rk = sqrt(2)*sqrt(rk + r_m*mfac)/lam;
            rk = sqrt(2*(rk + r_m*mfac)/m)/lam;
            nk = nk + ((abs((Z+Z_m*mfac)/lam) + rk*sqrt(cnorm))< 1);
            
            mean_nk = mean(nk);
            median_nk = median(nk);
            mmm = max([mean_nk, median_nk]);
        end
        
        % every T iterations, screening is appled
        if mod(t,T) == 0
            %pruning
            if prune && mod(t,T) == 0 && mmm>1*thresh
                % we only keep position which are not screened out over the
                % past "thresh" times
                % supp = (nk<=thresh);
                supp = (nk<mmm);
                
                % if |supp|=0, we've screened out everything and need to start again
                if ~max(supp)
                    disp('>>> restart screening...')
                    beta = beta_ + 1e-2*randn(size(beta_));
                    beta_old = beta_old_;
                    betasol = para.betasol;
                    Xt = Xt_;
                    Xt2 = Xt2_;
                    nk = 0*nk_;
                    Z = 0*Z_;
                    Z_m =0* Z_m_;
                    mfac = 0;
                    cnorm = cnorm_;
                    idx= (1:n)';
                    supp = ones(n,1)>0; % (abs(phi_a2)>1-1e-8);
                    
                    thresh = thresh+2; % increase the threshold
                    lb = lb *3;
                    
                    ii = randperm(m, 1);
                    Gamma = Gamma(ii:end);
                    w = min(w+0.1, 0.99);
                end
                
                % otherwise, we throw away the useless entries
                if ~min(supp)
                    % truncate
                    beta = beta(supp);
                    beta_old = beta_old(supp);
                    Z = Z(supp);
                    Xt = Xt(supp,:);
                    Xt2 = Xt2(supp,:);
                    Z_m = Z_m(supp);
                    idx = idx(supp);
                    nk = nk(supp);
                    cnorm = cnorm(supp);
                    
                    betasol = betasol(supp);
                    
                    % keep track of things we've screened out to reset later if
                    % necessary
                    beta_(idx) = beta;
                    beta_old_(idx) = beta_old;
                    Z_(idx) = Z + Z_m*mfac;
                    nk_(idx) = nk;
                    cnorm_(idx) = cnorm;
                end
            end
            
            % do the reset of anchor point and variables
            r_m = r_m*mfac + max(0, norm(Z/lam/(1-mfac),inf)-1)*f_;
            Z_m = Z + Z_m*mfac;
            Z = 0*Z;
            f_ = 0;
            mfac = 1;
            beta_a = beta;
            nmxa = norm(beta,1);
        end
        
    end
    
    % safety check to ensure no false removal
    if (mod(t, 5e5)==0 && numel(beta)>lb) || t==maxits-1
        phi_a = zeros(n,1);
        for j=1:m
            fac = 1/j;
            fj = f(j);
            Xjt = Xt(:,j);
            vj = m*(Xjt'*beta-fj);
            
            phi_a = - fac* Xt_(:,j)* vj + (1-fac) * phi_a;
        end
        %
        phi_a2 = phi_a/lam;
        phi_a2 = phi_a2/max(1,norm(phi_a2,inf));
        supp_c = ones(n,1); supp_c(idx) = 0;
        
        if max(abs(phi_a2.*supp_c)) < 0.99 || norm(phi_a/lam,inf)<=1+1e-5
            ;
        else
            disp('>>> bad exit...restart screening...')
            beta = beta0 + 1e-2*randn(n, 1);
            beta_old = beta0;
            betasol = para.betasol;
            Xt = Xt_;
            Xt2 = Xt2_;
            nk = zeros(n, 1);
            Z = 0* phi_a2;
            Z_m = 0*Z;
            cnorm = zeros(n,1);
            idx= (1:n)';
            
            thresh = thresh + 8; % increase the threshold
            lb = lb *3;
            
            r_m =0;
            Z_m = Z + Z_m*mfac;
            Z = 0*Z;
            f_ = 0;
            mfac = 0;
            beta_a = beta;
            nmxa = norm(beta,1);
            
            t = 1;
            ii = randperm(m, 1);
            Gamma = Gamma(ii:end);
            w = min(w+0.1, 0.99);
        end
    end
    
    % recording states
    if mod(t, T)==0
        dk(tt) = norm(beta - betasol, 'fro');
        tk(tt) = toc;
        
        sk(tt) = sum(supp);
        
        DD(tt) = Dval;
        PP(tt) = Pval;
        
        res = norm(beta_old-beta, 'fro');
        ek(tt) = res;
        
        tt = tt + 1;
        
        if toc>maxtime; break; end
    end
    
    t = t + 1;
end

% final solution check based on optimality condition
phi_a = zeros(n,1);
for j=1:m
    fac = 1/j;
    fj = f(j);
    Xjt = Xt(:,j);
    vj = m*(Xjt'*beta-fj);
    
    phi_a = - fac* Xt_(:,j)* vj + (1-fac) * phi_a;
end
%
phi_a2 = phi_a/lam;
phi_a2 = phi_a2/max(1,norm(phi_a2,inf));
supp_c = ones(n,1); supp_c(idx) = 0;
if max(abs(phi_a2.*supp_c)) < 0.99 || norm(phi_a/lam,inf)<=1+1e-5
    disp('>>>>>> good exit!')
else
    disp('>>>>>> bad exit!');
end

betasol = zeros(n,1);
betasol(idx) = beta;

output.ek = ek(1:tt-1);
output.dk = dk(1:tt-1);
output.sk = sk(1:tt-1);
output.tk = tk(1:tt-1);

output.DD = DD(1:tt-1);
output.PP = PP(1:tt-1);

output.theta = Z_m;
output.Z = Z;

output.nk = nk;

output.its = t;
output.betasol = betasol;