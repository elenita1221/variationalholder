

addpath('cvx')

N = 10;

cvx_begin
variable x(N)
% cvx raises error due to log_normcdf being concave
% minimize( log_normcdf(x) + x.^2/2 )

% guillaume's approximation
minimize (max(pow_pos(max(0,x-1),2)/2+x-1, -1-log(log(1+exp(-x)))))
cvx_end

