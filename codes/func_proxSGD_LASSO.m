function [x, its, ek,dk, sk,tk] = func_proxSGD_LASSO(para, rnd_idx)

% Output:
%   x: solution
%   its: number of iteration
%   ek: relative error
%   dk: absolute error
%   sk: support size over epochs
%   tk: time over epochs

f = para.f;
At = para.At;
m = para.m;

lam = para.lam;
Gamma = para.Gamma ;

tol = para.tol;
maxits = para.maxits;

ek = zeros(1, maxits);
dk = zeros(1, maxits);
sk = zeros(1, maxits);
tk = zeros(1, maxits);

xsol = para.xsol;
x0 = para.x0;
x = x0;

tic
kk = 0;
its = 1;
while(its<maxits)
    x_old = x;
    
    j = rnd_idx(its);
    Ajt = At(:,j);
    fj = f(j);
    
    gamma = Gamma(its);
    vj = m* (Ajt'*x-fj);
    
    tmpx = x - gamma * Ajt * vj;
    x = max(abs(tmpx)-lam*gamma,0) .* sign(tmpx);
    
    
    % record
    if mod(its,m)==0
        kk = kk + 1;
        
        dk(kk) = norm(x - xsol, 'fro');
        tk(kk) = toc;
        
        sk(kk) = sum(abs(x)>0);
        
        %%%%%%% stop?
        res = norm(x_old-x, 'fro');
        ek(kk) = res;
        if (res/length(x)<tol); break; end
    end
    
    its = its + 1;
    
end

ek = ek(1:kk-1);
dk = dk(1:kk-1);
sk = sk(1:kk-1);
tk = tk(1:kk-1);
