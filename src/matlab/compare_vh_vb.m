
clear all; close all; clc;
warning('off','all')
warning

%K = 3 %number of settings = number of rows in the plot
kappa_values = [1];
d_values = 3:4;
center_values = [1:2];
verbose = 0;
for d = d_values
    for k = 1:d
        for c = center_values
            for kappa = kappa_values
                
                v = randn(d, k);
                params.A = v * v' + kappa * eye(d);
                params.Achol = chol(params.A);
                if c == 1
                    center = zeros(d,1);
                elseif c == 2
                    center = linspace(-d/2,d/2,d)';
                end
                params.b = params.A * center;
   
                log_step_func = @(t) -1e10*real(t<0);
                
                %first function: step function in each of the directions
                log_f = @(t) log_step_func(t(:,1)) + log_step_func(t(:,2));
                %second function: Correlated Gaussian
                log_g = @(t) -.5*sum((t*params.Achol').^2,2) + t*b;
                %pivot function: diagonal covariance Gaussian
                log_r = @(t,theta) -.5*sum((t.^2).*(ones(size(t,1),1)*theta(1:d)'),2) + t*theta(d+1:end);
                %    fg = {log_f,params};
                fg = {{log_step_func, log_step_func},params};
                
                %% PLOT
                %     ng = 100; % grid size
                %     [gx,gy] = meshgrid(linspace(lims(1,1),lims(2,1),ng),linspace(lims(1,2),lims(2,2),ng));
                %     gridpoints = [gx(:),gy(:)];
                %     valf = exp(log_f(gridpoints)); %first function
                %     valg = exp(log_g(gridpoints)); %second function
                %
                if d == 2
                    % optimal integral
                    Istar = log(integral2(@(x,y) reshape(exp(log_f([x(:) y(:)])+log_g([x(:) y(:)])),size(x)),-inf,inf,-inf,inf));
                  % exact first moment in each of the dimension
                    mxstar = exp(-Istar)*integral2(@(x,y) x.*reshape(exp(log_f([x(:) y(:)])+log_g([x(:) y(:)])),size(x)),-inf,inf,-inf,inf);
                    mystar = exp(-Istar)*integral2(@(x,y) y.*reshape(exp(log_f([x(:) y(:)])+log_g([x(:) y(:)])),size(x)),-inf,inf,-inf,inf);
                else
                % compute optimal integral using Metropolis-Hastings
                
                end
                
                %% VH bound
                theta0 = [diag(params.A)*0.1;params.b/2];
                res0 = [theta0;logodd(.5)];
                objfun = @(t) upper_bound_logpartition(params,t);
                
                  UB0 = upper_bound_logpartition(params,res0);
%                fprintf('The exact integral is %4.3f\n', Istar)
                %fprintf('The variational holder bound gives %4.3f for the initial pivot function with parameters %s', UB0, num2str(theta0))
                
                
                [res1,UBopt1] = fminunc(objfun,res0,optimset('Display','off','MaxFunEvals',10000,'TolX',1e-7));
                
                [UB1,~,IfIg] = upper_bound_logpartition(params,res1);
                
                rho1 = sigmoid(res1(end)); %the first coefficient
                theta1=res1(1:end-1);
                
                if d == 2
                    %   I_fr = integral2(@(x,y) reshape(exp(1./rho1*log_f([x(:) y(:)])+(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    I_fr = integral2(@(x,y) reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    mx_fr = 1/I_fr*integral2(@(x,y) x.*reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    my_fr = 1/I_fr*integral2(@(x,y) y.*reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    
                    
                    I_gr = integral2(@(x,y) reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    mx_gr = 1/I_gr*integral2(@(x,y) x.*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    my_gr = 1/I_gr*integral2(@(x,y) y.*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    %
                    log_valr = log_r(gridpoints,theta1);
                    valr = exp(log_valr);
                    val_approx_fr = valf.^(1./rho1)     .* valr.^(1./rho1);
                    val_approx_gr = valg.^(1./(1-rho1)) ./ valr.^(1./(1-rho1));
                end
                
%                 ep=.00001;%nearly 0 so that the integral is a standard one \int{f*g}
%                 % if only the diagonal elements of A are used
%                 IDiago = factor_scaled_integral_univ({log_step_func, log_step_func},[diag(params.A);params.b],1-ep,1/ep);
%                 % if we remove the truncation
%                 IGauss = factor_scaled_integral_gauss(params,zeros(d*2,1),1-ep);
%                 
%                 % truncated gaussian with diagonal covariance
%                 Ifr = factor_scaled_integral_univ({log_step_func, log_step_func},theta1,1-ep,1/ep);
%                 
%                 % correlated gaussian without truncation
%                 params_gr.A = params.A + diag(theta1(1:d));
%                 params_gr.b = params.b + theta1(d+1:end)
%                 
%                 Igr = factor_scaled_integral_gauss(params_gr,zeros(d*2,1),1-ep);
%                 
%                 UB1
%                 [Istar IDiago IGauss UB0 UB1 Ifr Igr]
%                 
                
                
                %% VB approximation
                
                objfun_vb = @(t) negative_lower_bound_logpartition(params,t);
                
                Ainv = inv(params.A); mu_tmp = params.A\params.b;
                initial_soln = [0.1*(diag(Ainv).^0.5); mu_tmp];
                [final_soln,final_objfun_vb] = fminunc(objfun_vb,initial_soln,...
                    optimset('Display','off','MaxFunEvals',10000,'TolX',1e-7));
                final_soln_sigma = final_soln(1:length(final_soln)/2);
                final_soln_mu = final_soln(length(final_soln)/2+1:length(final_soln));
                holder_soln_sigma = 1./sqrt(theta1(1:end/2));
                holder_soln_mu = theta1(end/2+1:end);
                mx_vb = final_soln_mu(1); my_vb = final_soln_mu(2);
                
                
            end
        end
    end
end
