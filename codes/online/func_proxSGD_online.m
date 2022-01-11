function [beta, output] = func_proxSGD_online(para, seed)
if seed>0
    rng(seed)
end

maxits = para.maxits;
maxits1 = para.maxits1;

S = para.S; % how often to sample
R = para.R; % how often to record
r = para.r;

lam = para.lam;
samp = para.samp;

%initialisation
beta =  para.x0;
n = length(beta);

ek = zeros(ceil(maxits/R),1);
tk = ek;
sk1 = ek;
sk2 = ek;

pbar = maxits/10;

tic
t = 1;
tt = 1;
while(t<=maxits)
    if mod(t-1, S)==0
        % generate samples
        X  = samp(S+1,n);
        y = (X')*para.xsol + para.sig*randn(S+1,1);
    end
    
    beta_old = beta;
    
    j = mod(t, S) + 1;
    
    fj = y(j);
    Ajt = X(:, j);
    
    flg = double(t<=maxits1);
    gamma = para.gammai/ ( flg*t/r + (1-flg)*(t-1)/r + 1)^0.51;
    
    grd = Ajt'*beta - fj;
    tmpx = beta - gamma*grd * Ajt;
    
    beta = max(abs(tmpx)-lam*gamma, 0) .* sign(tmpx);
    
    %record
    if mod(t, R)==0
        tk(tt) = toc;
        ek(tt) = norm(beta - beta_old, 'fro');
        sk1(tt) = sum(abs(beta)>0);
        sk2(tt) = numel(beta);
        
        tt = tt+1;
    end

    if mod(t,pbar)==0
        disp(['    progress: ', num2str(t/pbar*10), '%...']);
    end
    
    t = t + 1;
end

output.t = t;
output.ek = ek(1:tt-1);
output.tk = tk(1:tt-1);
output.sk1 = sk1(1:tt-1);
output.sk2 = sk2(1:tt-1);