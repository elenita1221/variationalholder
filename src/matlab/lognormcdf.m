function [f, g] = lognormcdf(x)
% lognormcdf - logarithm of the normal cumulative density function and its derivative
%
% usage:
% [f, g] = lognormcdf(x)
%
% %example
% checkderivatives(@(x) lognormcdf(x), 1)


%y = -(log(1+exp(0.88-x))/1.5).^2; %is a very bad approximation

f = normcdfln(x);
g = nan;%not implemented

