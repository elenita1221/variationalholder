addpath('cvx')

N = 2;
k = 1;
if k==1
    A = [[0.5, 0.57]; [0.57, 1]];
    b = [1.6; 2.1];
elseif k==2
    A =[[0.3, 0.1]; [0.1, 0.7]];
    b = [1; 1.5];
elseif k==3
    A = [[0.5, 0.57]; [0.57, 1]];
    b = [3.1; 4.3];
end

log_2_pi = log(2*pi);
inv_alpha_1 = 0.5;
tau_2 = [0; 0];


% % overall objective function
% minimize( 0.5*N*(inv_alpha_1-1)*log_2_pi - 0.5*inv_alpha_1*(-N*log(inv_alpha_1) ...
% + sum(log(tau_1))) + 0.5*sum((tau_2.^2).*inv_pos(tau_1)) ...
% + sum(inv_alpha_1*log_normcdf(tau_2.*inv_pos(sqrt(inv_alpha_1*tau_1)))) ...
% +0.5*N*(1-inv_alpha_1)*log(1-inv_alpha_1) ...
% -0.5*(1-inv_alpha_1)*log_det(A-diag(tau_1)) ...
% + 0.5*matrix_frac(b-tau_2, A-diag(tau_1)) )

for iter = 1:10
    
    % optimize tau_1, tau_2 and alpha_1
    
    % optimize tau_1
    cvx_begin
    variable tau_1(N)
    minimize( - 0.5*inv_alpha_1*(sum(log(tau_1))) + 0.5*sum((tau_2.^2).*inv_pos(tau_1)) ...
        + inv_alpha_1*sum(log_normcdf(tau_2.*inv_pos(sqrt(inv_alpha_1*tau_1)))) ...
        -0.5*(1-inv_alpha_1)*log_det(A-diag(tau_1)) ...
        + 0.5*matrix_frac(b-tau_2, A-diag(tau_1)) )
    subject to
    tau_1 >= 0; A - diag(tau_1) == semidefinite(N);
    cvx_end
    
    cvx_begin
    variable tau_2(N)
    minimize(  0.5*sum((tau_2.^2).*inv_pos(tau_1)) ...
        + inv_alpha_1*sum(log_normcdf(tau_2.*inv_pos(sqrt(inv_alpha_1*tau_1)))) ...
        + 0.5*matrix_frac(tau_2-b, A-diag(tau_1)) )
    cvx_end
    
    
    % % optimize alpha_1
    
    % clear inv_alpha_1
    %   cvx_begin
    %  variable alpha_1
    % % minimize( 0.5*N*(1/alpha_1-1)*log_2_pi - 0.5/alpha_1*(N*log(alpha_1) ...
    % % + sum(log(tau_1)))   ...
    % % + sum(1/alpha_1*log_normcdf(tau_2.*inv_pos(sqrt((1/alpha_1)*tau_1)))) ...
    % % +0.5*N*(1-1/alpha_1)*log(1-(1/alpha_1)) ...
    % % -0.5*(1-1/alpha_1)*log_det(A-diag(tau_1)) )
    %
    %
    %  minimize( - 0.5/alpha_1*(N*log(alpha_1) + sum(log(tau_1))) )
    % %  + sum(1/alpha_1*log_normcdf(tau_2.*inv_pos(sqrt((1/alpha_1)*tau_1)))) ...
    % %  +0.5*N*(1-1/alpha_1)*log(1-(1/alpha_1)) ...
    % %  -0.5*(1-1/alpha_1)*log_det(A-diag(tau_1)) ...
    % %)
    % subject to
    % alpha_1 >= 1;
    %  cvx_end
    %  inv_alpha_1 = 1 / alpha_1;
    
end