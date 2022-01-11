function [beta, output, betasol] = func_SAGA_LASSO(para, rnd_idx)

% parameters
n = para.n;
m = para.m;

gamma = para.gamma /1.5;
lam = para.lam;
gamma0 = gamma;

% load data
f = para.f;
Xt = para.Xt;
Nm = para.Nm0;

% stop cnd, max iteration
tol = para.tol;
maxits = para.maxits;

% initial point
beta0 = para.beta0;
beta = beta0;

% initial averaged gradient
V = zeros(m, 1);
mG = 0;
for i=1:m
    Xjt = Xt(:, i);
    V(i) = Xjt'*beta0 - f(i);
    
    mG = mG + V(i)* Xjt;
end
mG = mG /m;

%%% obtain the minimizer x^\star
ek = zeros(maxits, 1);
sk = zeros(maxits, 1);

idx= (1:n)';
ikeep = (1:n)';

k = 0;
t = 1;
while(t<maxits)
    % every T iterations, screening is appled
    if mod(t,2*m) ==0 && 1
        
        % dual certificate
        Z = zeros(length(beta),1);
        
        % dual variable
        v_vec = zeros(m,1);
        
        % constructing dual certificate and variable
        for j=1:m
            fj = f(j);
            Xjt = Xt(:,j);
            
            vj = Xjt'*beta - fj;
            v_vec(j) = vj;
            
            Z = - vj* Xjt + Z;
        end
        
        % computing dual feasible point, s.t. |X^T\theta|_\infty \leq 1
        c = max(norm(Z,inf), 1);
        theta = -v_vec /c;
        
        % computing primal and dual funciton
        Dval = 0;
        Pval = 0;
        for j=1:m
            Pval = 1/2 * v_vec(j)^2 + Pval;
            Dval = - m*lam^2/2*(theta(j) - f(j)/lam/m)^2 + Dval;
        end
        Pval = Pval/m + lam*norm(beta,1);
        Dval = Dval + norm(f)^2/2/m;
        
        % duality gap
        gap = Pval - Dval;
        
        % radius of safe region
        rk = sqrt(2*gap/m)/lam;
        
        % safe rule
        nk = (abs(Z/c) + rk*Nm) < 1;
                
        % pruning
        if 1 % && mod(t,T) == 0
            supp = (nk==0);
            
            beta = beta(supp);
            Xt = Xt(supp,:);
            
            idx = idx(supp);
            Nm = Nm(supp);
                        
            mG = mG(supp);
            ikeep = ikeep(supp);
        end
        
    end
    
    % previous step
    beta_old = beta;
    
    % sampling
    j = rnd_idx(t);
    Xjt = Xt(:, j);
    
    vj = Xjt'*beta - f(j);
    
    % we gradualy increasing the step-size the gain faster performance
    gamma = min(gamma0*1.000000025^t, 2.718*gamma0);
    
    % SAGA gradient update
    g_diff = (vj - V(j))* Xjt;
    gk = beta - gamma* g_diff - gamma*mG;
    
    % proximal step
    beta = sign(gk) .* max(abs(gk)-lam * gamma, 0);
    
    % update gradient history
    V(j) = vj;
    mG = mG + g_diff /m;
    
    % recording
    if mod(t,m)==0
        k = k + 1;
        
        %%%%
        sk(k) = sum(abs(beta)>0);
        
        %%% stop?
        res = norm(beta(:) - beta_old(:), 'fro');
        ek(k) = res;
        if res<tol || res>1e10; break; end
    end
    
    t = t + 1;
    
end

betasol = zeros(n, 1);
betasol(ikeep) = beta;

output.its = t;
output.betasol = betasol;

output.ek = ek(1:k-1);
output.sk = sk(1:k-1);

