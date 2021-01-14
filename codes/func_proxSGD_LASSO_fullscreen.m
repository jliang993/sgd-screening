function [x, its, ek,dk, sk,tk] = func_proxSGD_LASSO_fullscreen(para,rnd_idx)

% Output:
% x: solution
% its: number of iteration
% ek: relative error
% dk: absolute error
% sk: support size over epochs
% tk: time over epochs

tol = para.tol;
maxits = para.maxits;
maxtime = para.maxtime;

T = para.T;
prune = para.prune;

f= para.f;
At = para.At;

Nm = sqrt(sum(abs(para.A).^2,1));
Nm = Nm(:);

m = para.m;
n = para.n;

w = para.w;

lam = para.lam;
Gamma = para.Gamma ;
%% setup screening variables
xsol = para.xsol;
x0 = para.x0;
xa = x0;
x = x0;

idx= (1:n)';
supp = ones(n, 1);

ek = zeros(1, maxits);
dk = zeros(1, maxits);
sk = zeros(1, maxits);
tk = zeros(1, maxits);

tic
kk = 0;
its = 1;
while(its<maxits)
    
    x_old = x;
    
    % sample for SGD
    j = rnd_idx(its);
    Ajt = At(:,j);
    fj = f(j);
    
    gamma = Gamma(its);
    
    vj = m*(Ajt'*x-fj);
    tmpx = x - gamma * Ajt * vj;
    
    x = max(abs(tmpx)-lam*gamma,0) .* sign(tmpx);
    
    % ergodic average
    fac = 1/its^w;
    xa = fac* x + (1-fac) * xa;
    
    % screening 
    if mod(its,T) ==0
        
        % dual certificate
        phi = zeros(length(x),1);
        v_vec = zeros(m,1);
        x_vec = zeros(m,1);
        for j=1:m
            fac = 1/j;
            fj = f(j);
            Ajt = At(:,j);
            u = Ajt'*xa;
            x_vec(j) = u;
            
            vj = m*(u-fj);
            v_vec(j) = vj;
            
            phi = - fac* Ajt* vj + (1-fac) * phi;
        end
        
        % normalization
        phi = phi/lam;
        c = 1/max(norm(phi,inf), 1);
        
        % computing dality gap
        Dval = 0;
        Pval = 0;
        for j=1:m
            fac = 1/j;
            fj = f(j);
            vj = v_vec(j);
            
            Pval = fac* ( m/2 * (x_vec(j)-fj)^2 ) + (1-fac) * Pval;
            Dval = fac* -1/2*((c*vj)^2/m+2*c*vj*fj) + (1-fac) * Dval ;
        end
        
        % gap-safe rule
        Pval = Pval+lam*norm(xa,1);
        gap = Pval - Dval ;
        rk = sqrt(2)*sqrt(gap)/lam;
        nk = ((abs((phi)*c) + rk*Nm)< 1);
        
        % pruning
        if prune && mod(its,T) == 0
            supp = (nk==0);
            
            % truncate
            x = x(supp);
            xsol = xsol(supp);
            x_old = x_old(supp);
            At = At(supp,:);
            
            idx = idx(supp);
            Nm = Nm(supp);
            xa = xa(supp);
        end
        
    end
    
    % record
    if mod(its,m)==0
        kk = kk + 1;
        
        dk(kk) = norm(x - xsol, 'fro');
        tk(kk) = toc;
        
        sk(kk) = sum(supp);
        
        if toc>maxtime; break; end
        
        %%%%%%% stop?
        res = norm(x_old-x, 'fro');
        ek(kk) = res;
        
        if (res/length(x)<tol)
            if norm(x)>1e10
                x = randn(size(x));
            else
                break;
            end
        end
    end
    
    its = its + 1;
end


xt = zeros(n,1);
xt(idx) = x;
x = xt;

ek = ek(1:kk-1);
dk = dk(1:kk-1);
sk = sk(1:kk-1);
tk = tk(1:kk-1);


