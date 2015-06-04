function op = normpdf(x, mu, sigma)
% x is normal distribution with mean mu, variance sigma 

op = 1 / sqrt(2*pi) ./ sigma .* exp(-0.5 * ((x-mu).^2) ./ sigma);