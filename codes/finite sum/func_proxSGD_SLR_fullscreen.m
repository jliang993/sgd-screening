function [beta, output, betasol] = func_proxSGD_SLR_fullscreen(para, rnd_idx)

% load data and parameters
f = para.f;
Xt = para.Xt;
fXt = para.fXt;
Nm = para.Nm0;

% period of screening and wether or not screening
T = para.T;
prune = para.prune;

% regularization parameter and precomputed step-size
lam = para.lam;
Gamma = para.Gamma;

% load data and parameters
maxits = para.maxits;
% the running time of fulling screening will not exceed that of pure SGD
maxtime = para.maxtime;

% size of problem
m = para.m;
n = para.n;

% for computing ergodic averaging weight
w = para.w;

% starting point and solution of SAGA
betasol = para.betasol;
beta0 = para.beta0;
beta_a = beta0;
beta = beta0;

% support for screening
idx= (1:n)';
supp = ones(n, 1);

% recording iterates information
ek = zeros(maxits, 1);
dk = zeros(maxits, 1);
sk = zeros(maxits, 1);
tk = zeros(maxits, 1);
% primal and dual function values
PP = zeros(maxits, 1);
DD = zeros(maxits, 1);

tic
t = 1;
tt = 1;
while(t<maxits)
    beta_old = beta;

    % sampling
    j =rnd_idx(t);
    fXjt = fXt(:,j);

    % step-size
    gamma = Gamma(t);

    % stochastic gradient descent
    vj = - 1 /(exp(fXjt'*beta)+1);
    tmpx = beta - (gamma*vj) * fXjt;

    % proximity operator
    beta = max(abs(tmpx)-lam*gamma,0) .* sign(tmpx);

    % ergodic average of iterations
    fac =1/t^w;
    beta_a = fac* beta + (1-fac) * beta_a;

    % screening
    if mod(t,T) ==0

        if numel(beta)>20
            % dual certificate
            Z = zeros(length(beta),1);

            % constructing dual certificate and variable
            u_vec = zeros(m,1);
            v_vec = zeros(m,1);
            for j=1:m
                fXjt = fXt(:,j);

                vj = fXjt'*beta_a;

                u_vec(j) = -1 /(exp(vj)+1);
                v_vec(j) = vj;

                Z = - u_vec(j)* fXjt + Z;
            end

            % computing dual feasible point, s.t. |X^T\theta|_\infty \leq 1
            c = max(norm(Z,inf), 1);
            theta = u_vec /c;

            % computing primal and dual funciton
            Dval = 0;
            Pval = 0;
            for j=1:m
                Pval = (- v_vec(j) + log(1+exp(v_vec(j)))) + Pval;
                pj = lam*theta(j)*m;
                Dval = (abs(-pj)>1e-14)*(-pj* log(-pj)) + (abs(1+pj)>1e-14)*((1+pj)* log(1+pj)) + Dval;
            end
            Pval = Pval/m + lam*norm(beta_a,1);
            Dval = - Dval/m;

            % duality gap
            gap = Pval - Dval;
            % radius of safe region
            rk = sqrt(2*gap/m)/lam;
            % safe rule
            nk = ((abs(Z/c) + rk*Nm) < 1);

            % recording info
            PP(tt) = Pval;
            DD(tt) = Dval;

            % pruning
            if prune
                supp = (nk==0);

                % truncate
                beta = beta(supp);
                betasol = betasol(supp);
                beta_old = beta_old(supp);
                Xt = Xt(supp,:);
                fXt = fXt(supp,:);

                idx = idx(supp);
                Nm = Nm(supp);
                beta_a = beta_a(supp);
            end

        end

        % recording
        dk(tt) = norm(beta - betasol, 'fro');
        tk(tt) = toc;

        sk(tt) = sum(supp);
        % sk(tt) = sum(abs(beta)>0);

        if toc>maxtime; break; end

        %%%%%%% stop?
        res = norm(beta_old-beta, 'fro');
        ek(tt) = res;

        tt = tt + 1;
    end

    t = t + 1;
end

betasol = zeros(n,1);
betasol(idx) = beta;

output.ek = ek(1:tt-1);
output.dk = dk(1:tt-1);
output.sk = sk(1:tt-1);
output.tk = tk(1:tt-1);

output.PP = PP(1:tt-1);
output.DD = DD(1:tt-1);

% output.Z = Z;
% output.theta = theta;

output.its = t;
output.betasol = betasol;

end


