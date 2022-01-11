function [beta, output, betasol] = func_SAGA_SLR(para, rnd_idx)


% parameters
n = para.n;
m = para.m;

gamma = para.gamma/1.5;
lam = para.lam;
gamma0 = gamma;

f = para.f;
Xt = para.Xt;
fXt = para.fXt;

Nm = para.Nm;

% stop cnd, max iteration
tol = para.tol;
maxits = para.maxits;

% initial point
beta0 = para.beta0;
beta = beta0;

V = zeros(m, 1);
mG = 0;
for i=1:m
    fXjt = fXt(:, i);
    V(i) = -1 / (1 + exp((fXjt'*beta0)));
    
    mG = mG + V(i)* fXjt;
end
mG = mG /m;

%%% obtain the minimizer x^\star
ek = zeros(maxits, 1);
sk = zeros(maxits, 1);
PP = zeros(maxits, 1);
DD = zeros(maxits, 1);

Z = 0;

stt = 0;

barg = 0;
ikeep = (1:n)';

k = 0;
t = 1;
tt = 1;
while(t<maxits)
    % every T iterations, screening is appled
    if mod(t,2*m) ==0 && 1
        
        beta_a = beta;
        
        % dual certificate
        Z = zeros(length(beta),1);
        
        % dual variable
        d_vec = zeros(m, 1); % derivative
        v_vec = zeros(m, 1); % 
        for j=1:m
            fXjt = fXt(:,j);
            
            vj = fXjt'*beta_a;
            
            d_vec(j) = -1 /(exp(vj)+1);
            v_vec(j) = vj;
            
            Z = - d_vec(j)* fXjt + Z;
        end
        
        % normalization
        % Z = Z/lam;
        c = max(norm(Z,inf), 1);
        theta = d_vec /c;
        
        % computing dality gap
        Dval = 0;
        Pval = 0;
        for j=1:m
            Pval = (-v_vec(j) + log(1+exp(v_vec(j)))) + Pval;

            pj = lam*theta(j)*m;
            Dval = (abs(-pj)>1e-14)*(-pj* log(-pj)) + (abs(1+pj)>1e-14)*((1+pj)* log(1+pj)) + Dval;
        end
        
        % gap-safe rule
        Pval = Pval/m + lam*norm(beta_a,1);
        Dval = -Dval/m; 
        gap = Pval - Dval;
        rk = sqrt(2*gap/m)/lam;
        nk = ((abs(Z/c) + rk*Nm)< 1);
        
        stt = stt + nk;

        % recording info
        PP(tt) = Pval;
        DD(tt) = Dval;
        
        tt = tt + 1;
        
        % pruning
        if 1
            supp = (nk==0);
            
            % truncate
            beta = beta(supp);

            Xt = Xt(supp,:);
            fXt = fXt(supp,:);
            
            ikeep = ikeep(supp);
            Nm = Nm(supp);
            mG = mG(supp);
            
            stt = stt(supp);
            barg = barg(supp);
        end
        
    end
    
    % previous step
    beta_old = beta;
    
    %%%%%%%%%%%
    j = rnd_idx(t);
    fXjt = fXt(:, j);
    vj = -1 / (1 + exp((fXjt'*beta)));
    
    gamma = min(gamma0*1.000000025^t, 2.718*gamma0);
    
    g_diff = (vj - V(j)) *fXjt;
    gk = beta - gamma* g_diff - gamma*mG;
    
    barg = (t-1)/t* barg + 1/t* vj* fXjt;
    
    beta = sign(gk) .* max(abs(gk)-lam * gamma, 0);
    
    V(j) = vj;
    mG = mG + g_diff/m;
    
    %%%%%%%%%%%
    if mod(t,m)==0
        k = k + 1;
        
        %%%%%% update params
        sk(k) = sum(abs(beta)>0);
        
        %%% stop?
        res = norm(beta(:) - beta_old(:), 'fro');
        ek(k) = res;
        if res<tol || res>1e10
            break;
        end
    end
    
    t = t + 1;
    
end
% fprintf('\n');

betasol = zeros(n, 1);
betasol(ikeep) = beta;

output.its = t;
output.betasol = betasol;

output.ek = ek(1:k-1);
output.sk = sk(1:k-1);
output.nk = stt;

output.mG = mG;
output.barg = barg;

output.Z = Z;

output.PP = abs(PP(1:tt-1));
output.DD = abs(DD(1:tt-1));

output.stt = stt;
