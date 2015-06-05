function params = make_expt_params(d,k,c,kappa)

v = randn(d, k);
params.A = v * v' + kappa * eye(d);
params.Achol = chol(params.A);
if c == 1
    center = zeros(d,1);
elseif c == 2
    %center = linspace(-d/2,d/2,d)';
    center = ones(d,1);
end
params.b = params.A * center; % -.5*x'*A*x + b'*x = -.5*(x-inv(A)*b)'*A*(-inv(A)*b) + b'*x + c


