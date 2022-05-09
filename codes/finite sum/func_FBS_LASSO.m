function [beta, output, betasol] = func_FBS_LASSO(para, ifpruning)

% parameters
n = para.n;
m = para.m;

gamma = para.gamma /1.5;
lam = para.lam;
gamma0 = gamma;

% load data
f = para.f;
Xt = para.Xt;
Nm = para.Nm0;
X = Xt';

% stop cnd, max iteration
tol = para.tol;
maxits = para.maxits;

% initial point
beta0 = para.beta0;
beta = beta0;

%%% obtain the minimizer x^\star
ek = zeros(maxits, 1);
sk = zeros(maxits, 1);
ssk = zeros(maxits, 1);
Pk = zeros(maxits, 1);
Dk = zeros(maxits, 1);

idx= (1:n)';
ikeep = (1:n)';

k = 0;
t = 1;
while(t<maxits)
    % every T iterations, screening is appled
    if mod(t, 1) ==0 && ifpruning

        % dual certificate
        Z = zeros(length(beta),1);

        % dual variable
        % constructing dual certificate and variable
        vv = f - X*beta;
        zZ = Xt* vv;

        % computing dual feasible point, s.t. |X^T\theta|_\infty \leq 1
        c = max(norm(zZ/(lam*m),inf), 1);
        theta = vv /c/(lam*m); %/(lam*m);

        % [c, max(abs(Xt*theta))]

        % computing primal and dual funciton
        Dval = 0;
        Pval = 0;
        for j=1:m
            Pval = 1/2 * vv(j)^2 + Pval;
            Dval = m*lam^2/2*(theta(j) - f(j)/(lam*m))^2 + Dval;
        end
        Pval = Pval/m + lam*norm(beta,1);
        Dval = - Dval + norm(f)^2/2/m;

        Pk(t) = Pval;
        Dk(t) = Dval;

        % duality gap
        gap = Pval - Dval;

        % radius of safe region
        rk = sqrt(2*gap/m) /lam;

        % safe rule
        ck = theta;
        nk = (abs(Xt*ck) + rk*Nm) < 1;

        % pruning
        if 1 % && mod(t,T) == 0
            supp = (nk==0);

            ssk(t) = sum(supp);

%             disp(sum(supp))

%             beta = beta(supp);
%             Xt = Xt(supp,:);
%             X = Xt';
% 
%             idx = idx(supp);
%             Nm = Nm(supp);
% 
%             ikeep = ikeep(supp);
        end

        log2([gap; sum(supp); sum(abs(beta)>0)])
    end

    % previous step
    beta_old = beta;


    % we gradualy increasing the step-size the gain faster performance
    gamma = m*gamma0; %min(gamma0*1.000000025^t, 2.718*gamma0);

    % gradient update
    gk = beta - gamma* Xt*(X*beta - f)/m;

    % proximal step
    beta = sign(gk) .* max(abs(gk) - lam*gamma, 0);

    % recording
    if mod(t,1)==0
        k = k + 1;

        %%%%
        sk(k) = sum(abs(beta)>0);

        %%% stop?
        res = norm(beta(:) - beta_old(:), 'fro');
        ek(k) = res;
        if res<tol || res>1e10; break; end
    end

    t = t + 1;

end

betasol = zeros(n, 1);
betasol(ikeep) = beta;

output.its = t;
output.betasol = betasol;

output.ek = ek(1:k-1);
output.sk = sk(1:k-1);
output.ssk = ssk(1:k-1);

output.Pk = Pk(1:k-1);
output.Dk = Dk(1:k-1);

