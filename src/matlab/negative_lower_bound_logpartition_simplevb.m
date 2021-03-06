function [objfun] = negative_lower_bound_logpartition_simplevb(params,theta)
%function [objfun, objfun_grad] = negative_lower_bound_logpartition_simplevb(params,theta)

n = length(theta) / 2;
assert(n == length(params.b));
sigma = theta(1:n);
mu = theta(n+1:end);
if any(sigma <= 0)
    objfun = inf;
    return;
end
    
%sigma2 = sigma .^2 ;
mu_over_sigma = mu ./ sigma;
mu_over_sigma2 = mu_over_sigma.^2;
normcdf_mu_over_sigma = normcdf(mu_over_sigma);

E_t = mu.*normcdf_mu_over_sigma + sigma / sqrt(2*pi) .* exp(-0.5*(mu_over_sigma2));
E_tt = E_t * E_t';
E_tt(logical(eye(n))) = (mu.^2 + sigma.^2).*normcdf_mu_over_sigma + mu .* sigma / sqrt(2*pi) .* exp(-0.5*(mu_over_sigma2));

elbo = -0.5*trace(params.A * E_tt) + params.b'*E_t +0.5*n*log(2*pi) +0.5*sum(log(sigma));
objfun = -elbo;
% if nargout>1 %need gradient
%     objfun_grad = 
% end