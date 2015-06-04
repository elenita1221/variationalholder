function op = truncnormpdf(x, mu, sigma, a)
% x is normal distribution with mean mu, variance sigma truncated below at
% a

op = (x > a) .* normpdf(x, mu, sigma) .* normcdf(mu ./ sigma);