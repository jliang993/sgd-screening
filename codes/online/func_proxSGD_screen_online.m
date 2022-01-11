function [x, res,its, supp_sz,rerr_, time_] = func_proxSGD_screen_online(para,seed)
if seed>0
    rng(seed)
end

x0 = para.x0;
maxits = para.maxits;
tol = para.tol;
prune = para.prune;
T = para.T;


m=para.m;
lam = para.lam;
thresh = para.thresh;

n = length(x0);
nk = zeros(n, 1); % to record screened index using stochastic grad


%% setup screening variables
phi = zeros(n,1); %dual certificate
phi_m = zeros(n,1); %previous dual certificate
cnorm = zeros(n,1); %norm of A(:,j) for each j
x_a = x0; %primal anchor point
nmxa = norm(x_a,1);

Pval = 0; %running primal value
Dval = 0; %running dual value
r_m = 0;
f_= 0;
mfac = 0;
w = para.w;


%objects for recording old values in case we screen everything out..
x_ = zeros(n,1);
x_old_ = zeros(n,1);
phi_ = zeros(n,1);
phi_m_ = zeros(n,1);
nk_  = zeros(n,1);
cnorm_ = zeros(n,1);



x = x0;
evals = 1;
idx= (1:n)';
res = length(x);

n = length(x);
samp = para.samp ;

err_ = zeros(maxits,1); e_ = 1;
time_ = err_;
rerr_ = err_;
% 
% %generate samples
% A  = samp(maxits,n);
% y = A*para.xsol + para.sig*randn(maxits,1);
A  = samp(m,n);
y = A*para.xsol + para.sig*randn(m,1);
        
tic
%warm start
for its = 1:para.init
    if mod(its,m)==0
        %generate samples
        A  = samp(m,n);
        y = A*para.xsol + para.sig*randn(m,1);
    end
    
    %sample for SGD
    Aj = A(mod(its,m)+1,:);
    Ajt = Aj';
    fj = y(mod(its,m)+1);
   
    gamma = para.gammai / its^0.51;
    vj = Aj*x-fj;
    
    tmpx = x - gamma * Ajt * vj;
    x = max(abs(tmpx)-lam*gamma,0).*sign(tmpx);
    
end

xsol = para.xsol;
idsol = find(abs(xsol)>0);
nsol = length(idsol);
while(its<maxits)
    
    if mod(its,m)==0
        %generate samples
%         A  = samp(m,n);
%         y = A*para.xsol + para.sig*randn(m,1);
%         A = A(:,idx);

        %this takes advantage of the fact we don't need to observe samples
        %at all positions
        A0 = samp(m,nsol);
        y = A0*para.xsol(idsol) + para.sig*randn(m,1);
        A = samp(m,length(idx));
        [~,id,id2] = intersect(idx,idsol);
        A(:,id) = A0(:,id2);
    end

    %sample for SGD
    x_old = x;
    Aj = A(mod(its,m)+1,:);
    Ajt = Aj';
    fj = y(mod(its,m)+1);
    
    
    gamma = para.gammai / its^0.51;
    fac =1/its^w;
    %update primal dual gap
    va = Aj*x_a;
    vj = (Aj*x-fj);
    f_ = fac * 1/2 * fj^2 + (1-fac)*f_;
    Pval = fac* ( 1/2 * (va-fj)^2  + lam* nmxa ) + (1-fac) * Pval;
    Dval = fac*  -1/2*(vj^2+2*vj*fj)  + (1-fac) * Dval ;
    
    
    mfac = mfac*(1-fac);
    %build screening certificate
    cnorm =  fac* Ajt.^2 + (1-fac) * cnorm;
    phi = - fac*  Ajt* vj + (1-fac) * phi; %update certificate
    
    
    tmpx = x - gamma * Ajt * vj;
    x = max(abs(tmpx)-lam*gamma,0).*sign(tmpx);
    
    %record
    if mod(its,500)==0
        time_(e_) = toc;
        rerr_(e_) = norm(x - x_old, 'fro');
        e_ = e_+1;
    end
    
    
    %apply screening
    if mod(its,T) ==0
        gap = Pval  - Dval;
        rk =  gap + max(0, norm(phi/lam/(1-mfac),inf)-1)*f_ ;
%         if rk + r_m*mfac <0
%             disp('neg')
%         end
        rk = sqrt(2)*sqrt(rk + r_m*mfac)/lam;
        nk =  nk+ ((abs((phi+phi_m*mfac)/lam) + rk*sqrt(cnorm))< 1);
    end
    
    %reset every T iterations
    if mod(its,T) == 0
        %pruning
        if prune && mod(its,T) == 0
            supp =  (nk<=thresh);
            
            %if we've screened out everything then start again
            if sum(supp)==0
                disp('restart screening...')
                x = x_;
                x_old = x_old_;
                nk = 0*nk_;
                phi = 0*phi_;
                phi_m =0* phi_m_;
                mfac = 0;
                cnorm = cnorm_;
                idx= (1:n)';
                thresh = thresh+10; %increase the threshold
                supp = ones(n,1)>0;% (abs(phi_a2)>1-1e-8);
            end
            
            %truncate
            if sum(supp)<n
                %                 disp('truncate')
                x = x(supp);
                x_old = x_old(supp);
                phi = phi(supp);
                phi_m = phi_m(supp);
                idx = idx(supp);
                nk = nk(supp);
                cnorm = cnorm(supp);
                A = A(:,supp);
                
                
                %keep track of things we've screened out to reset later if
                %necessary
                x_(idx) = x;
                x_old_(idx) = x_old;
                phi_(idx) = phi + phi_m*mfac;
                nk_(idx) = nk;
                cnorm_(idx) = cnorm;
            end
        end
        
        %do the reset of anchor point and variables
        r_m = r_m*mfac +  max(0, norm(phi/lam/(1-mfac),inf)-1)*f_;
        phi_m = phi + phi_m*mfac;
        phi = 0*phi;
        f_ = 0;
        mfac = 1;
        x_a = x;
        nmxa = norm(x,1);
        
        res = norm(x_old-x, 'fro');
    end
    
    
    
    supp_sz(evals) = length(x);
    
    %%%%%%% stop?
    
    if (res/length(x)<tol)
        if norm(x)>1e10
            x = randn(size(x));
        else
            break;
        end
    end
    evals = evals+2;
    its = its + 1;
    if its==maxits && norm(x)>1e10
        maxits = maxits*10;
    end
end


xt = zeros(n,1);
xt(idx) = x;
x = xt;



time_ = time_(1:e_-1);
rerr_ = rerr_(1:e_-1);



end


