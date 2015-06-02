function [I,I_grad] = factor_scaled_integral_orthant(theta,invalpha)
% factor_scaled_integral_orthant
% Integral of a diagonal covariance Gaussian restricted to the positive orthant

sz = size(theta);
if any(theta<=0)
    I = Inf;
    I_grad = inf * ones(sz);
    return
end

theta = reshape(theta,[],2);
d = size(theta,1);
c = theta(:,2).^2 ./ theta(:,1) / invalpha;
sc = sqrt(c);

[logphi, dlogphi] = lognormcdf(sc);
I = invalpha * ( .5*d*log(2*pi*invalpha) ...
    - .5 * sum(log(theta(:,1))) ...
    + sum(logphi) ...
    + .5 * sum(c) );

if nargout>1
    theta1_ov_alpha = theta(:,1) * invalpha;    
    I_grad_a = .5 * ((- 1 - dlogphi .* sc)./theta(:,1) - c./theta(:,1));
    I_grad_b = sign(theta(:,2)).*(dlogphi./sqrt(theta1_ov_alpha)) + theta(:,2)./theta1_ov_alpha;
    I_grad = invalpha * [I_grad_a I_grad_b];
    I_grad = reshape(I_grad,sz);
end
