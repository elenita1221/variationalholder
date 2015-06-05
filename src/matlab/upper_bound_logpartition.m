function [logIbar, logIbar_grad, logIfg, m, v] = upper_bound_logpartition(params,theta)

logIfg = [0 0];
%[a,b] = factor_scaled_integral_univ(fg{1},theta,rho,delta);
[logIfg(1),logIf_grad1] = factor_scaled_integral_orthant(theta);
[logIfg(2),logIg_grad1] = factor_scaled_integral_gauss(params,-theta);

logIfg = real(logIfg);
logIbar = sum(logIfg);
if nargout>1 %need gradient
    logIbar_grad = logIf_grad1 - logIg_grad1;
end