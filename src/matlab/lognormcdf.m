function [f, g] = lognormcdf(x)
% lognormcdf - logarithm of the normal cumulative density function and its derivative
%
% usage:
% [f, g] = lognormcdf(x)
%
% %example
% checkderivatives(@(x) lognormcdf(x), 1)


%y = -(log(1+exp(0.88-x))/1.5).^2; %is a very bad approximation

f = log(0.5*erfc(-x * sqrt(0.5)));
g = exp(-.5*x.^2 - 0.9189385332 - f);
