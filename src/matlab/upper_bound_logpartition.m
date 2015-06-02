function [logIbar,logIbar_grad,logIfg] = upper_bound_logpartition(params,theta,rho)

logIfg = [0 0];
%[a,b] = factor_scaled_integral_univ(fg{1},theta,rho,delta);
[logIfg(1),logIf_grad] = factor_scaled_integral_orthant(theta,rho);
logIf_grad
[logIfg(2),logIg_grad] = factor_scaled_integral_gauss(params,-theta,1-rho);
logIg_grad
logIfg = real(logIfg);

logIbar = sum(logIfg);
logIbar_grad = logIf_grad - logIg_grad;