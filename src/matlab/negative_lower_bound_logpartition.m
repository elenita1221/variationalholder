function [objfun] = negative_lower_bound_logpartition(params,theta)
%function [objfun, objfun_grad] = negative_lower_bound_logpartition(params,theta)

n = length(theta) / 2;
assert(n == length(params.b));
sigma = theta(1:n);
mu = theta(n+1:end);
if any(sigma <= 0)
    objfun = inf;
    return;
end

mu_over_sigma = mu ./ sigma;
mu_over_sigma2 = mu_over_sigma.^2;
normcdf_mu_over_sigma = normcdf(mu_over_sigma);

% E_t = mu + sigma / sqrt(2*pi) .* exp(-0.5*(mu_over_sigma2)) ./ normcdf_mu_over_sigma;
% E_tt = E_t * E_t';
% E_tt(logical(eye(n))) = mu.^2 + sigma.^2 + ...
%     mu .* sigma / sqrt(2*pi) .* exp(-0.5*(mu_over_sigma2)) ./ normcdf_mu_over_sigma;

[E_t, e_tt] = truncnorm_moments(mu, sigma);
E_tt = E_t * E_t';
E_tt(logical(eye(n))) = e_tt;

%elbo = -0.5*trace(params.A * E_tt) + params.b'*E_t +0.5*n*log(2*pi) +0.5*sum(log(sigma)) ...
%    +0.5*sum( sigma.^2 - mu .* sigma / sqrt(2*pi) .* exp(-0.5*(mu_over_sigma2)) ./ normcdf_mu_over_sigma);

elbo = -0.5*trace(params.A * E_tt) + params.b'*E_t +0.5*n*(1+log(2*pi)) +sum(log(sigma)) ...
    + sum(log(normcdf_mu_over_sigma)) ...
    -0.5*sum(mu .* exp(-0.5*(mu_over_sigma2)) ./ (sqrt(2*pi) .* sigma .* normcdf_mu_over_sigma));

objfun = -elbo;
% if nargout>1 %need gradient
%     objfun_grad = 
% end