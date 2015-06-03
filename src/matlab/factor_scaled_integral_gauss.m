function [I,I_grad] = factor_scaled_integral_gauss(params,theta,prop)


if nargin<=2
    grad_prop = 1;
    [prop,dprop] = sigmoid(theta(end));
    theta = theta(1:end-1);
else
    grad_prop = 0;
end

    
    
d = size(params.A,1);
Am = params.A + diag(theta(1:d));
bm = params.b + theta(d+1:end);
[f,gCell] = gauss_integral(...
    Am/prop,...
    bm/prop...
    );

I_grad = [diag(gCell{1}); gCell{2}];
I = prop * f;

if grad_prop
    I_grad_prop = dprop * (f - (sum(sum(gCell{1}.*Am)) + gCell{2}'*bm) / prop);
    I_grad = [I_grad; I_grad_prop];
end
