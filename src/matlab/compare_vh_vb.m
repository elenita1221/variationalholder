
clear all; close all; clc;
addpath('cunninghamTruncGauss');
warning('off','all')
warning

%K = 3 %number of settings = number of rows in the plot
kappa_values = [0.1 1];
d_values = [5 20 50]% [2 5 10 20 100];%3:4;
center_values = [1];
verbose = 0;
num = 0;%index of the experiment
results0 = {};
results1 = {};
results2 = {};

for d = d_values
    for k = [1]%[1 d]
        for c = center_values
            for kappa = kappa_values
                num = num + 1;
                fprintf('\nExperiment %d\nd=%d\nk=%d\nc=%d\nkappa=%3.2f\n',num,d,k,c,kappa)
                params = make_expt_params(d,k,c,kappa);
                
                
                log_step_func = @(t) -1e10*real(t<0);
                %first function: step function in each of the directions
                log_f = @(t) log_step_func(t(:,1)) + log_step_func(t(:,2));
                %second function: Correlated Gaussian
                log_g = @(t) -.5*sum((t*params.Achol').^2,2) + t*params.b;
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
                theta0 = [diag(params.A)*0.01;params.b/2];
                res0 = [theta0;logodd(.5)];
                objfun = @(t) upper_bound_logpartition(params,t);
                
                UB0 = upper_bound_logpartition(params,res0);
                %                fprintf('The exact integral is %4.3f\n', Istar)
                %fprintf('The variational holder bound gives %4.3f for the initial pivot function with parameters %s', UB0, num2str(theta0))
                
                
                [res1,UBopt1] = fminunc(objfun,res0,optimset('Display','off','MaxFunEvals',10000,'TolX',1e-7));
                
                [UB1,~,IfIg] = upper_bound_logpartition(params,res1);
                
                
                rho1 = sigmoid(res1(end)); %the first coefficient
                res1 = [zeros(2,1);zeros(2,1);-5];
                theta1=res1(1:end-1);
                [I_gr,~, m_gr, m2_gr] = factor_scaled_integral_gauss(params, -res1);
                [I_fr,~, m_fr, m2_fr] = factor_scaled_integral_orthant(res1);
                %  inv_alpha2 = 1 - rho1;
                %  I_gr_debug  = factor_scaled_integral_gauss_debug(params, res1(1:end-1), inv_alpha2);
                I_vh = I_fr + I_gr;
                % compute weighted moments
                m_vh = rho1*m_fr+ (1-rho1)*m_gr;
                m2_vh = rho1*m2_fr + (1-rho1)*m2_gr;
                
                if d == 2
                    %   I_fr = integral2(@(x,y) reshape(exp(1./rho1*log_f([x(:) y(:)])+(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    I_fr2 = integral2(@(x,y) reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    I_gr2 = integral2(@(x,y) reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    m_fr2 = [1/I_fr2*integral2(@(x,y) x.*reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);...
                        1/I_fr2*integral2(@(x,y) y.*reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf)];
                    m2_fr2 = [1/I_fr2*integral2(@(x,y) (x.^2).*reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);...
                        1/I_fr2*integral2(@(x,y) (y.^2).*reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf)];
                    
                    m_gr2 = [1/I_gr2*integral2(@(x,y) x.*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);...
                        1/I_gr2*integral2(@(x,y) y.*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf)];
                    m2_gr2_mxx = 1/I_gr2*integral2(@(x,y) (x.^2).*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    m2_gr2_myy = 1/I_gr2*integral2(@(x,y) (y.^2).*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    m2_gr2_mxy = 1/I_gr2*integral2(@(x,y) (x.*y).*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
                    m2_gr2 = [m2_gr2_mxx m2_gr2_mxy; m2_gr2_mxy m2_gr2_myy];
                    
                    
                    %[I_gr log(I_gr2)*(1-rho1) ] %debug
                    
                    %[m_gr m_gr2]%debug
                    %ddd%debug
                    
                    %[m2_gr m2_gr2]% debug
                    %ddd%debug
                    
                    %[I_fr log(I_fr2)*rho1 ] %debug
                    %ddd
                    
                    %[m_fr m_fr2]%debug
                    %ddd%debug
                    
                    %[diag(m2_fr) m2_fr2]% debug
                    %ddd%debug
                    
                end
                
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
                initial_soln = [0.5*(diag(Ainv).^0.5); mu_tmp];
                [final_soln,final_objfun_vb] = fminunc(objfun_vb,initial_soln,...
                    optimset('Display','off','MaxFunEvals',10000,'TolX',1e-7));
                final_soln_sigma = final_soln(1:length(final_soln)/2);
                final_soln_mu = final_soln(length(final_soln)/2+1:length(final_soln));
                holder_soln_sigma = 1./sqrt(theta1(1:end/2));
                holder_soln_mu = theta1(end/2+1:end);
                mx_vb = final_soln_mu(1);
                my_vb = final_soln_mu(2);
                [m_vb, m2_vb] = truncnorm_moments(final_soln_mu, final_soln_sigma);
                m2_vb = diag(m2_vb);
                I_vb = -final_objfun_vb;
                %rho_vb = 1;
                %theta_vb = [final_soln_sigma.^(-2); final_soln_mu.*(final_soln_sigma.^(-2))];
                %        [m, m2] = truncnorm_moments(theta(:,2)./theta(:,1), 1./sqrt(theta(:,1)/rho));
                %I_vb = factor_scaled_integral_orthant(theta_vb, rho_vb)
                
                % % EP approximation
                cov_gaussian = inv(params.A);
                mean_gaussian = params.A \ params.b;
                C = eye(d);
                lB = zeros(d,1);
                uB = inf*ones(d,1);
                [logZ_ep, m_ep, Sigma_ep, extras_ep] = epmgp(mean_gaussian,cov_gaussian,C,lB,uB);%,maxSteps,alphaCorrection)
                m2_ep = Sigma_ep + m_ep*m_ep';
                I_ep = logZ_ep - 0.5*logdet(params.A/(2*pi)) -0.5*params.b'*mean_gaussian;
                
                I_genz = genz_integral(params);
                results0{end+1} = [kappa d I_genz I_ep  I_vb I_vh ];
                [I_vh I_vb I_ep I_genz]
             %   [m_vh m_vb m_ep]
             %   [m2_vh m2_vb m2_ep]
              %  pause
              err_m_vh = norm(m_vh - m_ep);
              err_m2_vh = norm(m2_vh - m2_ep);
              err_m_vb = norm(m_vb - m_ep);
              err_m2_vb = norm(m2_vb - m2_ep);
              diff_m_vb_vh = norm(m_vh - m_vb);
              diff_m2_vb_vh = norm(m2_vh - m2_vb);
              [err_m_vh err_m2_vh]
              [err_m_vb err_m2_vb]
              results1{end+1} = [kappa d err_m_vb err_m_vh  diff_m_vb_vh ];
              results2{end+1} = [kappa d err_m2_vb err_m2_vh  diff_m2_vb_vh ];
            end
        end
    end
end

% I=cat(1,results0{:});
% matrix2latex(I, 'integrals')
% err_m=cat(1,results1{:});
% err_m2=cat(1,results2{:});
% matrix2latex(err_m, 'errm')
% matrix2latex(err_m2, 'errm2')
