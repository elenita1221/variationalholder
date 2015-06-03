function [I,I_grad] = factor_scaled_integral_orthant(theta,rho)
% factor_scaled_integral_orthant
% Integral of a diagonal covariance Gaussian restricted to the positive orthant

if nargin<=1
    grad_rho = 1;
    rho = theta(end);
    theta = theta(1:end-1);
else
    grad_rho = 0;
end

sz = size(theta);
theta = reshape(theta,[],2);

if any(theta(:,1)<=0)
    I = Inf;
    if grad_rho
        I_grad = nan * ones(prod(sz)+1,1);
    else
        I_grad = nan * ones(sz);
    end
    return
end

d = size(theta,1);
c = theta(:,2).^2 ./ theta(:,1) / rho;
sc = sqrt(c) .* sign(theta(:,2));

%sc = theta(:,2) ./ sqrt(theta(:,1)) / sqrt(rho)

[logphi, dlogphi] = lognormcdf(sc);
unscaledI = .5*d*log(2*pi*rho) ...
    - .5 * sum(log(theta(:,1))) ...
    + sum(logphi) ...
    + .5 * sum(c);
I = rho * unscaledI;

if nargout>1
    theta1_ov_alpha = theta(:,1) * rho;    
    I_grad_a = .5 * ((- 1 - dlogphi .* sc)./theta(:,1) - c./theta(:,1));
    I_grad_b = sign(theta(:,2)).*(dlogphi./sqrt(theta1_ov_alpha)) + theta(:,2)./theta1_ov_alpha;
    I_grad = rho * [I_grad_a I_grad_b];
    I_grad = reshape(I_grad,sz);

    if grad_rho
        dc_drho = -c/rho;
        dsc_drho = .5/sc * dc_drho .* sign(theta(:,2)); 
        dlogphi_drho = dlogphi .* dsc_drho;
        I_grad_rho = unscaledI + rho * ...
            (.5 * d / rho + sum(dlogphi_drho) + .5*sum(dc_drho));
        I_grad = [I_grad;I_grad_rho];
    end
end
