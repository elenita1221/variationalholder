clear all
%dataset = 'toy2d'
dataset = 'toy1dnew'
if strcmp(dataset, 'toy1d')
    X = 1 * [1; -1]; y = [1; -1];
elseif strcmp(dataset, 'toy1dnew')
    n=100;
    p=1;
    mag = 3
    y=sign(randn(n,1));
    X = randn(n,p) + mag * (y*ones(1,p)) .* ones(n, p);
    X = center_and_whiten(X);
elseif strcmp(dataset, 'toy2d')
    n=100;
    p=2;
    mag = 3;
    y=sign(randn(n,1));
    X = randn(n,p) + mag * (y*ones(1,p)) .* ones(n, p);
    X = center_and_whiten(X);
elseif strcmp(dataset, 'toy10d')
    n=100;
    p=10;
    y=sign(randn(n,1));
    X = randn(n,p) + (y*ones(1,p)) .* ones(n, p);
    X = center_and_whiten(X);
elseif strcmp(dataset, 'heart')
    data = load('statlog.heart.data');
    %%X = standardizeCols(data(:,1:13)); y = sign(data(:,14)-1.5);
    X = center_and_whiten(data(:,1:13)); y = sign(data(:,14)-1.5);
end
[n,p] = size(X);



%lambda = 2;
lambda = 1;
%X = [ones(n,1) X];
%prior = [0 zeros(1,p);zeros(p,1) lambda*eye(p)];
prior = [lambda*eye(p)];

train_probit2Sample = 0;
nSamples = 5000;
n_rep = 20;%0;
s_rep = zeros(p, nSamples, n_rep);
s2_rep = zeros(p, nSamples, n_rep);
%z2_rep = zeros(nSamples, n_rep);

for rep = 1:n_rep
    if train_probit2Sample
        s = probit2Sample(X,y,prior,nSamples);
        s_rep(:, :, rep) = s;
    end
    [s2, z2] = probit2GibbsSample(X,y,prior,nSamples);
    s2_rep(:, :, rep) = s2;
    
    % z2_rep(:, :, rep) = z2;
    % burn_in = nSamples / 2;
    % mean(s2(:,burn_in:end))
    % mean(z2(:,burn_in:end),2)
    % burn_in = nSamples / 2;
    % var(s2(:,burn_in:end))
    % var(z2(:,burn_in:end),0,2)
    fprintf('----- rep: %5d -----\n', rep)
    if train_probit2Sample
        disp('mean of s')
        mean(s,2)'
    end
    disp('mean of s2')
    mean(s2,2)'
end

if train_probit2Sample
    disp('quantiles of s')
    s_rep_mean_avg = mean(s_rep_mean, 3);
    s_rep_mean_5p = quantile(s_rep_mean, 0.05, 3);
    s_rep_mean_95p = quantile(s_rep_mean, 0.95, 3);
    
    s_rep_mean_avg(:, end)'
    s_rep_mean_5p(:, end)'
    s_rep_mean_95p(:, end)'
    %save demo_probit_results_heart.mat
end

disp('quantiles of s2')
s_rep_mean = zeros(p, nSamples, n_rep);
s2_rep_mean = zeros(p, nSamples, n_rep);
weight = (1 ./ (1:nSamples)') * ones(1, n_rep);
for p_tmp = 1:p
    s_rep_mean(p_tmp, :, :) = weight .* cumsum(squeeze(s_rep(p_tmp, :, :)), 1);
    s2_rep_mean(p_tmp, :, :) = weight .* cumsum(squeeze(s2_rep(p_tmp, :, :)), 1);
end


s2_rep_mean_avg = mean(s2_rep_mean, 3);
s2_rep_mean_5p = quantile(s2_rep_mean, 0.05, 3);
s2_rep_mean_95p = quantile(s2_rep_mean, 0.95, 3);

s2_rep_mean_avg(:, end)'
s2_rep_mean_5p(:, end)'
s2_rep_mean_95p(:, end)'

%figure(7)
%plot(s2_rep_mean)
%figfont()
%saveFigure('albertchib')



% s2_rep_mean_avg = mean(s2_rep_mean, 2);
% s2_rep_mean_5p = quantile(s2_rep_mean, 0.05, 2);
% s2_rep_mean_95p = quantile(s2_rep_mean, 0.95, 2);
% figure(8)
% hold on
% plot(s2_rep_mean_avg)
% plot(s2_rep_mean_5p, '--')
% plot(s2_rep_mean_95p, '--')
% figfont()
% saveFigure('albertchibquantiles')
%
% save demo_probit_results.mat

% 1 d example:
% mean over multiple runs is 0.8477
% 90% quantile is [0.8204, 0.8769]
