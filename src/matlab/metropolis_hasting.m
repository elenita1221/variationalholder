function [theta,lprob,nrep,stepsize] = metropolis_hasting(logprobafunc,proposal,theta_cur,varargin)
%metropolis_hasting - Markov Chain Monte Carlo sampling with Metropolis algorithm.
%
% usage:
%
% theta = metropolis_hasting(logprobafunc,proposal,theta0)
%
% returns 1000 samples obtained by Metropolis-Hasting algorithm after 100
% steps burnin
%
%
% %example: 
% m = [5;3];s = 2;lpf = @(x) log(.3*exp(-.5*sum(x.^2,2))+.7/s*exp(-.5/s*(sum(x.^2,2)-2*x*m+sum(m.^2))));
% [x,y] = meshgrid(-4:.3:10,-4:.3:8);z=reshape(exp(lpf([x(:) y(:)])),size(x));contour(x,y,z,20);
% prop = @(x,sz) x + sz*randn(1,2);
% xc =  metropolis_hasting(lpf,prop,[0 0]);
% x = cat(1,xc{:});
% line(x(:,1),x(:,2),'Marker','.')


nburnin = 100;
niter = 1000;
verbose = 0;
repeat_rejects = 1; %repeat points when the next step is rejected  (yes=1/no=0)
symetric = 1;
adapt = 1;%adaptation
stepsize = 1;

SIMPLE_ARGS_DEF


theta = cell(1,niter);
nrep = zeros(1,niter);
lprob = zeros(1,niter);

i = 0;
lp_cur = logprobafunc(theta_cur);
accepts = zeros(1,niter+nburnin+1);


% for it=-nburnin:niter
it = -nburnin;
while 1
    if verbose && mod(it,verbose)==0 && it>-nburnin
        fprintf('iteration %d: stepsize=%4.3f, acceptance rate=%3.1f%%\n',it,stepsize,mean(accepts(1:it+nburnin))*100)
    end
    theta_old = theta_cur;
    lp_old = lp_cur;
    
    if ~symetric
        [theta_trial,lr] = proposal(theta_old,stepsize); %proposes a value for theta
        lp_trial = logprobafunc(theta_trial);
        accept = rand<exp(lp_trial - lp_old - lr);
    else
        theta_trial = proposal(theta_old,stepsize); %proposes a value for theta
        lp_trial = logprobafunc(theta_trial);
        accept = rand<exp(lp_trial - lp_old);
    end
        
    if accept
        theta_cur = theta_trial;
        lp_cur = lp_trial;
        if it>0 %record 
            i = i + 1;
            theta{i} = theta_cur;
            lprob(i) = lp_trial;
            nrep(i) = 1;
        elseif adapt %adaptation
            stepsize = stepsize*(1+1./(1+it+nburnin).^.5); %increase
        end
    else
        if it>0 && i>0%record
            nrep(i) = nrep(i) + 1;
        elseif adapt %adaptation
            stepsize = stepsize*(1-.33./(1+it+nburnin).^.5); %decrease
        end
    end    
    accepts(it+nburnin+1)=accept;
    it = it+1;
    if it>niter%stopping criterion 
       if i>0 || it>10*niter 
        break
       end
    end
end


theta = theta(1:i);
nrep = nrep(1:i);
lprob = lprob(1:i);

if repeat_rejects       
    inds = steps(nrep);
    theta = theta(inds);
end
