function [x, output] = func_SAGA_SLR(para, rnd_idx)


% parameters
n = para.n;
m = para.m;

gamma = para.gamma/1.5;
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
    % V(i) = lr_grad(Ajt'*x0,f(i));
    V(i) = -f(i) / (1 + exp(f(i)*(Ajt'*x0))); %lr_grad(Ajt'*x0,f(i));
    
    mG = mG + V(i)* Ajt;
end

mG = mG /m;

% lr_grad(z,f) = -f/(exp(f*z)+1);

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
    % j = randperm(m, 1);
    j = rnd_idx(its);
    Ajt = At(:, j);    
    
    % vj = lr_grad(Ajt'*x,f(j));
    vj = -f(j) / (1 + exp(f(j)*(Ajt'*x))); 
    
    gamma = min(gamma0*1.000000025^its, 2.718*gamma0);
    
    % gj_old = G(:, j);
    % gj =  Ajt* vj;
    g_diff = (vj - V(j)) *Ajt;
    
    gk = x - gamma* g_diff - gamma*mG;
    
    x = sign(gk) .* max(abs(gk)-lam * gamma, 0);
    
    V(j) = vj;
    mG = mG + g_diff/m;
    
    
    %%%%%%%%%%%
    if mod(its,m)==0
        k = k + 1;
        
        %%%%%% update params
        sk(k) = sum(abs(x)>0);
        
        %%% stop?
        res = norm(x(:) - x_old(:), 'fro');
        ek(k) = res;
        if res<tol || res>1e10
            break;
        end
    end
    
    its = its + 1;
    
end
% fprintf('\n');

output.its = its;

output.xsol = x;

output.ek = ek(1:k-1);
output.sk = sk(1:k-1);

end



function x = lr_grad(z,f) %gradient of lr_loss
x = -f/(exp(f*z)+1);
end
