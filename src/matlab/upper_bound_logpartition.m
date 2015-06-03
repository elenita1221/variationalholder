function [logIbar, logIbar_grad1, logIbar_grad2,logIfg] = upper_bound_logpartition(params,theta,rho)

if isstr(rho) %can specify that the proportion rho is vectorized (last entry of the vector)
    if  strcmp(rho,'vectorized')
        vec = 1;
        rho = theta(end);%sigmoid(theta(end));
        theta = theta(1:end-1);
    else
        error('rho must be a real or ''vectorized'' ')
    end
else
    vec = 0;
end

logIfg = [0 0];
%[a,b] = factor_scaled_integral_univ(fg{1},theta,rho,delta);
[logIfg(1),logIf_grad1,logIf_grad2] = factor_scaled_integral_orthant(theta,rho);
[logIfg(2),logIg_grad1,logIg_grad2] = factor_scaled_integral_gauss(params,-theta,1-rho);

logIfg = real(logIfg);
logIbar = sum(logIfg);
if nargout>1 %need gradient
    logIbar_grad1 = logIf_grad1 - logIg_grad1;

    if vec %gradient of the proportion at the end
        gp = logIf_grad2 - logIg_grad2; % the proportion gradient is just the difference between the two gradients
        logIbar_grad1 = [logIbar_grad1;gp]; %put it in the last dimension
    end
end