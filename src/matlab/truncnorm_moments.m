function  [m, m2] = truncnorm_moments(mu, sigma)
% x is normal distribution with mean mu, variance sigma truncated at 0

mu_over_sigma = mu ./ sigma;
%normcdf_mu_over_sigma = normcdf(mu_over_sigma);
[lognormcdf_mu_over_sigma, ~] = lognormcdf(mu_over_sigma);
tmp_term = 1/sqrt(2*pi) * exp(-0.5*(mu_over_sigma.^2) -lognormcdf_mu_over_sigma);
m =  mu + sigma .* tmp_term;
v = (sigma.^2) .* (1 - tmp_term.^2 - mu_over_sigma.*tmp_term);
m2 = m.^2 + v;
