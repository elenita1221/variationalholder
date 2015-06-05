function theta = metropolis_hastings_step(logprobafunc,theta,theta_trial)
% metropolis_hastings_step - single step of Markov Chain Monte Carlo sampling with Metropolis algorithm.
%
% by default, a symetric proposal is assumed
% thetanew = metropolis_hastings_step(logprobafunc,theta_cur,theta_prop)
%
% %example:
% m = [5;3];s = 2;lpf = @(x) log(.3*exp(-.5*sum(x.^2,2))+.7/s*exp(-.5/s*(sum(x.^2,2)-2*x*m+sum(m.^2))));
% [x,y] = meshgrid(-4:.3:10,-4:.3:8);z=reshape(exp(lpf([x(:) y(:)])),size(x));contour(x,y,z,20);
% N = 1000;x = zeros(N,2);
% for i=2:N;x(i,:)=metropolis_hastings_step(lpf,x(i-1,:),x(i-1,:)+randn(1,2)/3);end
% line(x(:,1),x(:,2),'Marker','.')

if rand < exp(logprobafunc(theta_trial) -  logprobafunc(theta));
    theta = theta_trial;
end
