function I_genz = genz_integral(params)
N = 1e5;
m = params.A\params.b;
d = length(params.b);
[p, e] = qsclatmvnv(N, inv(params.A), -m, eye(d), inf*ones(d,1));
I_genz = log(p) - 0.5*logdet(params.A/(2*pi)) -0.5*params.b'*m;
