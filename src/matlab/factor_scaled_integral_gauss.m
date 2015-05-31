function [I,I_grad] = factor_scaled_integral_gauss(params,theta,prop,delta)


d = size(params.A,1);

I = prop * gauss_integral(...
    params.A/prop + delta*diag(theta(1:d))/prop,...
    params.b/prop + delta*theta(d+1:end)/prop...
    );

I_grad = 0;