function [logIbar,logIbar_grad,logIfg] = upper_bound_logpartition(fg,theta,rho,delta)

if nargin<4
    delta = 1;
end
logIfg = [0 0];
[logIfg(1),logIf_grad] = factor_scaled_integral_univ(fg{1},theta,rho,delta);
[logIfg(2),logIg_grad] = factor_scaled_integral_gauss(fg{2},theta,1-rho,-delta);
logIfg = real(logIfg);

logIbar = sum(logIfg);
logIbar_grad = logIf_grad + logIg_grad;