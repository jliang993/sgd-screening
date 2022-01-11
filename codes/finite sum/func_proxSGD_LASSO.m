function [beta, output, betasol] = func_proxSGD_LASSO(para, rnd_idx)

% load data and parameters
f = para.f;
Xt = para.Xt;

% period of recording status
T = para.T;

% regularization parameter and precomputed step-size
lam = para.lam;
Gamma = para.Gamma;

% max number of iterations
maxits = para.maxits;

% recording states
ek = zeros(1, maxits);
dk = zeros(1, maxits);
sk = zeros(1, maxits);
tk = zeros(1, maxits);

% starting point and solution of SAGA
betasol = para.betasol;
beta0 = para.beta0;
beta = beta0;

tic
t = 1;
kk = 0;
while(t<maxits)
    beta_old = beta;
    
    % sampling
    j = rnd_idx(t);
    Xjt = Xt(:,j);
    fj = f(j);
    
    % step-size
    gamma = Gamma(t);
    
    % infeasible dual variable
    vj = Xjt'*beta - fj;
    
    % prox-sgd step
    tmpx = beta - (gamma*vj) * Xjt;
    beta = max(abs(tmpx)-lam*gamma,0) .* sign(tmpx);
    
    % recording
    if mod(t, T)==0
        kk = kk + 1;
        
        dk(kk) = norm(beta - betasol, 'fro');
        tk(kk) = toc;
        
        sk(kk) = sum(abs(beta)>0);
        
        %%%%%%% stop?
        res = norm(beta_old-beta, 'fro');
        ek(kk) = res;
    end
    
    t = t + 1;
    
end

betasol = beta;

output.ek = ek(1:kk-1);
output.dk = dk(1:kk-1);
output.sk = sk(1:kk-1);
output.tk = tk(1:kk-1);

output.its = t;
output.betasol = betasol;