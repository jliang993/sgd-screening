function [beta, output, betasol] = func_proxSGD_SLR(para, rnd_idx)

% load data and parameters
% f = para.f;
% Xt = para.Xt;
fXt = para.fXt;

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
tt = 1;
while(t<maxits)
    beta_old = beta;
    
    % sampling
    j = rnd_idx(t);
    fXjt = fXt(:,j);
    
    % step-size
    gamma = Gamma(t);

    % infeasible dual variable
    vj = -1 /(exp(fXjt'*beta)+1);
    
    % prox-sgd step
    tmpx = beta - gamma*vj * fXjt;
    beta = max(abs(tmpx)-lam*gamma, 0) .* sign(tmpx);
    
    % record
    if mod(t,T)==0
        dk(tt) = norm(beta - betasol, 'fro');
        tk(tt) = toc;
        
        sk(tt) = sum(abs(beta)>0);
        
        %%%%%%% stop?
        res = norm(beta_old-beta, 'fro');
        ek(tt) = res;
        
        tt = tt + 1;
    end
    
    t = t + 1;
    
end

betasol = beta;

output.ek = ek(1:tt-1);
output.dk = dk(1:tt-1);
output.sk = sk(1:tt-1);
output.tk = tk(1:tt-1);

output.its = t;
output.betasol = betasol;