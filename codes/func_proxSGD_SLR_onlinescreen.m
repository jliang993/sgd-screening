function [x, its, ek,dk, sk,tk] = func_proxSGD_SLR_onlinescreen(para, rnd_idx)

% Output:
%   x: solution
%   its: number of iteration
%   ek: relative error
%   dk: absolute error
%   sk: support size over epochs
%   tk: time over epochs

tol = para.tol;
maxits = para.maxits;
maxtime = para.maxtime;

T = para.T;
prune = para.prune;
thresh = para.thresh;

At_ = para.At;
At2_ = At_.^2;
At = At_;
At2 = At2_;
f= para.f;

m = para.m;
n = para.n;

w = para.w;

nk = zeros(n, 1); % to record screened index using stochastic grad

lam = para.lam;
Gamma = para.Gamma ;
%% setup screening variables
xsol = para.xsol;
x0 = para.x0;
x_a = x0; %primal anchor point
x = x0;

phi = zeros(n,1); % dual certificate
phi_m = zeros(n,1); % previous dual certificate
cnorm = zeros(n,1); % norm of A(:,j) for each j
nmxa = norm(x_a,1);

Pval = 0; % running primal value
Dval = 0; % running dual value
r_m = 0;
f_= 0;
mfac = 0;

% objects for recording old values in case we screen everything out..
x_ = zeros(n,1);
x_old_ = zeros(n,1);
phi_ = zeros(n,1);
phi_m_ = zeros(n,1);
nk_  = zeros(n,1);
cnorm_ = zeros(n,1);

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
    j =rnd_idx(its);
    Ajt = At(:,j);
    fj = f(j);
    
    gamma = Gamma(its);
    
    % weight
    fac = 1/its^w;
    
    % update primal dual gap
    va = Ajt'*x_a;
    vj = -fj/(exp(fj*Ajt'*x)+1) *m;
    f_ = fac * log(2)*m + (1-fac)*f_;

    if isinf(exp(-fj*va))
        xx = -fj*va + log(1+exp(fj*va));
    else
        xx = log(1+exp(-fj*va));
    end
    
    Pval = fac* ( xx*m  + lam* nmxa ) + (1-fac) * Pval;
    Dval = fac*  - ( (abs(1 + vj*fj/m)>1e-14)* (1 + vj*fj/m) * log(1 + vj*fj/m) + ...
        (abs(-vj*fj/m)>1e-14)* (-vj*fj/m) * log(-vj*fj/m) )*m  + (1-fac) * Dval;
    
    mfac = mfac*(1-fac);
    
    % build screening certificate
    cnorm =  fac* At2(:,j) + (1-fac) * cnorm;
    phi = - fac*  Ajt* vj + (1-fac) * phi; % update certificate
    
    
    % SGD step
    tmpx = x - gamma * Ajt * vj;
    x = max(abs(tmpx)-lam*gamma,0).*sign(tmpx);
    
    
    % apply screening
    if mod(its,T) ==0
        gap = Pval  - Dval ;
        rk =  gap + max(0, norm(phi/lam/(1-mfac),inf)-1)*f_ ;
        rk = sqrt(2)*sqrt(rk + r_m*mfac)/lam;
        nk =  nk+ ((abs((phi+phi_m*mfac)/lam) + rk*sqrt(cnorm))< 1);
    end
    
    % reset every T iterations
    if mod(its,T) == 0
        % pruning
        if prune && mod(its,T) == 0
            supp =  (nk<=thresh);
            
            % if we've screened out everything then start again
            if ~max(supp)
                disp('restart screening...')
                x = x_ + 1e-4*randn(size(x_));
                x_old = x_old_;
                xsol = para.xsol;
                At = At_;
                At2 = At2_;
                nk = 0*nk_;
                phi = 0*phi_;
                phi_m =0* phi_m_;
                mfac = 0;
                cnorm = cnorm_;
                idx= (1:n)';
                thresh = thresh+10; % increase the threshold
                supp = ones(n,1)>0;% (abs(phi_a2)>1-1e-8);
                
                w = 0.6;
            end
            
            if ~min(supp)
                % truncate
                x = x(supp);
                x_old = x_old(supp);
                phi = phi(supp);
                At = At(supp,:);
                At2 = At2(supp,:);
                phi_m = phi_m(supp);
                idx = idx(supp);
                nk = nk(supp);
                cnorm = cnorm(supp);
                
                xsol = xsol(supp);
                
                % keep track of things we've screened out to reset later if
                % necessary
                x_(idx) = x;
                x_old_(idx) = x_old;
                phi_(idx) = phi + phi_m*mfac;
                nk_(idx) = nk;
                cnorm_(idx) = cnorm;
            end
        end
        
        % do the reset of anchor point and variables
        r_m = r_m*mfac +  max(0, norm(phi/lam/(1-mfac),inf)-1)*f_;
        phi_m = phi + phi_m*mfac;
        phi = 0*phi;
        f_ = 0;
        mfac = 1;
        x_a = x;
        nmxa = norm(x,1);
    end
    
    
    
    if mod(its,m)==0
        kk = kk + 1;
        
        dk(kk) = norm(x - xsol, 'fro');
        tk(kk) = toc;
        
        sk(kk) = sum(supp); % sum(supp); % sum(abs(x)>0);
        
        %%%%%%% stop?
        res = norm(x_old-x, 'fro');
        ek(kk) = res;
        
        if toc>maxtime; break; end
        
        if (res/length(x)<tol) || its==maxits-1
            if norm(x)>1e10
                x = randn(size(x));
            else
                
                phi_a = zeros(n,1);
                
                for j=1:m
                    fac = 1/j;
                    fj = f(j);
                    Ajt = At(:,j);
                    vj = -fj/(exp(fj*Ajt'*x)+1) *m;
                    
                    phi_a =  - fac*  At_(:,j)* vj + (1-fac) * phi_a;
                end
                % 
                phi_a2 = phi_a/lam;
                phi_a2 = phi_a2/max(1,norm(phi_a2,inf));
                supp_c = ones(n,1); supp_c(idx) = 0;
                if max(abs(phi_a2.*supp_c)) < 0.99
                    break;
                else
                    disp('bad exit...restart screening...')
                    x = x_ + 0.001*randn(size(x_));
                    xsol = para.xsol;
                    At = At_;
                    At2 = At2_;
                    nk = 0*nk_;
                    phi = phi_a2;
                    phi_m = 0*phi_m_;
                    cnorm = cnorm_;
                    idx= (1:n)';
                    
                    thresh = thresh*3; % increase the threshold
                    
                    r_m =0;
                    phi_m = phi + phi_m*mfac;
                    phi = 0*phi;
                    f_ = 0;
                    mfac = 1;
                    x_a = x;
                    nmxa = norm(x,1);
                    
                    its = randperm(1e3, 1);
                    
                    % w = min(w*1.05, 0.75);
                    w = 0.6;
                    
                end
            end
        end
    end
    
    its = its + 1;
end


phi_a = zeros(n,1);
for j=1:m
    fac = 1/j;
    fj = f(j);
    Ajt = At(:,j);
    vj = lr_grad(Ajt'*x,fj)*m;
    phi_a =  - fac*  At_(:,j)* vj + (1-fac) * phi_a;
end
% 
phi_a2 = phi_a/lam;
phi_a2 = phi_a2/max(1,norm(phi_a2,inf));
supp_c = ones(n,1); supp_c(idx) = 0;
if max(abs(phi_a2.*supp_c)) < 0.99 || norm(phi_a/lam,inf)<=1+1e-5
    % if max(abs(phi_a2)) > 1
    disp('good exit')
else
    disp('bad exit');
end


xt = zeros(n,1);
xt(idx) = x;
x = xt;

ek = ek(1:kk-1);
dk = dk(1:kk-1);
sk = sk(1:kk-1);
tk = tk(1:kk-1);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = lr_grad(z,f) % gradient of lr_loss
x = -f/(exp(f*z)+1);
end


function  y = logexp(z) % computes log(1+exp(z))
if isinf(exp(z))
    y = z+ log(1+exp(-z));
else
    y = log(1+exp(z));
end
% y = z+ log(1+exp(-z));
end

function y = xlogx(x) % computes xlogx
if abs(x)<1e-14
    y = 0;
else
    y = x*log(x);
end

% y = x*log(x);

end

