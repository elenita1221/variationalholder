function y = lognormcdf(x)
y = -(log(1+exp(0.88-x))/1.5).^2;