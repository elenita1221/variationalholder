function op = center_and_whiten(X)
%EPS = eps;
[N, D] = size(X);
C = X - ones(N,1) * mean(X,1); % centered matrix
[V, eigval] = eig(C' * C);
d = diag(eigval);
rot = V * diag(1 ./ sqrt(d)) * V';
op = X * rot;
%covariance = cov(C);
%chol_cov = chol(covariance);
%op = (1/sqrt(N-1)) * C / chol_cov;
end