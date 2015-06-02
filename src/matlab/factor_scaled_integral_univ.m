function [I,I_grad,Waypoints] = factor_scaled_integral_univ(log_func,theta,invalpha,delta,L)
% factor_scaled_integral_univ
%
% L are lipschitz constants for the factors derivatives

theta = reshape(theta,[],2);
d = size(theta,1);
theta_mod = delta.*theta/invalpha;

if nargin<5
    L = ones(length(log_func),1)*.01; %to avoid integrating the step function over reals
end
ints = zeros(d,1);
for i=1:d
    if L(i)/invalpha < theta_mod(i,1)  %numerical check that the integral is finite      
        wp = 1/sqrt(abs(theta_mod(i,1)));        
    %    u = [theta_mod(i,1), log(integral(@(t) exp(log_func(t)/prop -.5*theta_mod(i,1).*t.^2+theta_mod(i,2).*t),-inf,inf,'Waypoints',wp))]
        ints(i) = log(integral(@(t) exp(log_func{i}(t)/invalpha -.5*theta_mod(i,1).*t.^2+theta_mod(i,2).*t),-inf,inf,'Waypoints',[-wp 0 wp]));
    else   
        ints(i) = inf;
        break
    end
    %   WAYPOINTS (but they do not improve accuracy)
    %         [~,~,Waypoints] = log_func(0);%returns Waypoints
    %         a = log(integral(@(t) exp(log_func(t)/prop -.5*theta_mod(i,1).*t.^2+theta_mod(i,2).*t),-inf,inf));
    %         b = log(integral(@(t) exp(log_func(t)/prop -.5*theta_mod(i,1).*t.^2+theta_mod(i,2).*t),-inf,inf,'Waypoints',Waypoints));
    %         if abs(a-b)>1e-5
    %             i=1;
    %             ddd
    %         end
    %         ints(i) = log(integral(@(t) exp(log_func(t)/prop -.5*theta_mod(i,1).*t.^2+theta_mod(i,2).*t),-inf,inf,'Waypoints',Waypoints));
    
end
% F2 = @(a,b) sqrt(2*pi)*exp(b^2/2/a)/sqrt(a)*normcdf2(b/sqrt(a))

I = invalpha * sum(ints);

I_grad = 0;

