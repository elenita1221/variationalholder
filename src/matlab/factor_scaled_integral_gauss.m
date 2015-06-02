function [I,I_grad] = factor_scaled_integral_gauss(params,theta,prop)


d = size(params.A,1);

[f,g] = gauss_integral(...
    params.A/prop + diag(theta(1:d))/prop,...
    params.b/prop + theta(d+1:end)/prop...
    );

I = prop * f;
I_grad_A = g{1};
I_grad_b = g{2};
I_grad = [I_grad_A; I_grad_b];
