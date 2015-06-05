function I = factor_scaled_integral_gauss_debug(params,theta,inv_alpha2)


d = size(params.A,1);
mat = (params.A - diag(theta(1:d)));
bm = (params.b - theta(d+1:end));
e = eig(mat);
logdet = sum(log(e));

I = inv_alpha2*0.5*d*log(2*pi) -0.5*inv_alpha2 * (-d*log(inv_alpha2) + logdet) ...
    + 0.5 * (bm' * (mat\ bm));
%I_debug_terms = [0.5*d*log(2*pi) -0.5*(-d*log(inv_alpha2) + logdet) ...
%    0.5/inv_alpha2 * (bm' * (mat\ bm))]

