function [x, output] = func_SAGA_LASSO(para, rnd_idx)

% parameters
n = para.n;
m = para.m;

gamma = para.gamma /1.5;
lam = para.lam;
tau = lam * gamma;
gamma0 = gamma;

At = para.At;
f = para.f;

% stop cnd, max iteration
tol = para.tol;
maxits = para.maxits;

% rng(para.seed);
% rnd_idx = randi(m,maxits,1);

% initial point
x0 = ones(n, 1);

V = zeros(m, 1);
mG = 0;
for i=1:m
    Ajt = At(:, i);
    V(i) = Ajt'*x0 - f(i);
    
    mG = mG + V(i)* Ajt;
end
mG = mG /m;

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%%% obtain the minimizer x^\star
ek = zeros(maxits, 1);
sk = zeros(maxits, 1);

x = x0;

k = 0;
its = 1;
while(its<maxits)
    
    x_old = x;
    
    %%%%%%%%%%%
    j = rnd_idx(its);
    Ajt = At(:, j);
    
    vj = Ajt'*x - f(j);
    
    gamma = min(gamma0*1.000000025^its, 2.718*gamma0);
    
    % gj_old = V(j)* Ajt;
    % gj = vj* Ajt;
    g_diff = (vj - V(j))* Ajt;
    
    gk = x - gamma* g_diff - gamma*mG;
    
    x = sign(gk) .* max(abs(gk)-lam * gamma, 0);
    
    V(j) = vj;
    mG = mG + g_diff /m;
    
    %%%%%%%%%%%
    if mod(its,m)==0
        k = k + 1;
        
        %%%%
        sk(k) = sum(abs(x)>0);
        
        %%% stop?
        res = norm(x(:) - x_old(:), 'fro');
        ek(k) = res;
        if res<tol || res>1e10; break; end
    end
    
    its = its + 1;
    
end
% fprintf('\n');

output.its = its;


output.xsol = x;

output.ek = ek(1:k-1);
output.sk = sk(1:k-1);

end

