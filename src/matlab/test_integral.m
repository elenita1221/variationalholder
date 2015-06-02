theta = [1 -4]
invalpha = 1;
I1 = integral(@(t) exp((0 - .5*theta(:,1).*t.^2 + theta(:,2).*t)/invalpha),0,inf);
I2 = exp(factor_scaled_integral_orthant(theta,invalpha));

d = 1;
c = (theta(:,2).^2 ./ theta(:,1)) / invalpha;
I3 =  exp(.5*d*log(2*pi*invalpha) ...
    - .5 * log(theta(:,1)) ...
    + sum(log_normcdf(sign(theta(:,2)).*sqrt(c(1)))) ...
    + .5 * sum(c(1)) );

[I1 I2 I3]
clear


theta = [1 .1];
invalpha = 1.5;
%f = @(thet) log(integral(@(t) exp((- .5*thet.*t.^2 + thet.*t)/invalpha),0,inf));
f = @(thet) factor_scaled_integral_orthant(thet,invalpha);

checkderivatives(f, theta)


%% the Gaussian integral and its derivatives
invalpha = .1;
theta = [1 2 3 4 5 6]'/10;
params.A = [1 .1 .3;.1 2 .5;.3 .5 2];
params.b = [1 2 3]';
checkderivatives(@(t) factor_scaled_integral_gauss(params,t,invalpha), theta);

%% checks the derivatives of the full function
checkderivatives(@(t) upper_bound_logpartition(params,t,invalpha), theta);

