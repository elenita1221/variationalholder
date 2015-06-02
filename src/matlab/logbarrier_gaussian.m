function [f,g] = logbarrier_gaussian(tau,A)
% logbarrier_gaussian : log-barrier function for gaussian approximations
% log-barrier for uncorrelated gaussian approximations of correlated gaussians

if nargin==0
    d = 4;
    Achol = randn(3,d);A = Achol'*Achol;
    checkderivatives(@(t) logbarrier_gaussian(t,A), randn(d,1)/10)
    return
end

d = length(A);
M = A - diag(tau);
ev = eig(M);
%if any eigenvalue is negative, break
if any(ev<=0)
    f = inf;
    g = nan*ones(d,1);
    return
end

f = -sum(log(ev));
g =  diag(inv(M));
g = g(:);




