function [J,Jgrad] = gauss_integral(A,b)
% gauss_integral - unnormalized Gaussian integral
% gauss_integral(A,b) computes the integral of
%
% exp(-.5*t*A*t' + b'*t)
%
% [f,g] = gauss_integral(A,b)
% g{1} returns the gradients with respect to the diagonal entries of A
% g{2} returns the gradients with respect to b

[eigvec, eigv] = eig(A);
eigv = diag(eigv);
if any(eigv<eps)
    J = inf;
    Jgrad = {nan*ones(size(A)), nan*ones(size(b))};
else
    d = size(A,1);
    Ainvb = A\b;
    J = d/2*log(2*pi) - .5*sum(log(eigv)) + .5*b'*Ainvb;
    if nargout>1
        %        Jgrad = {diag(-.5 * inv(A) - .5 * Ainvb*Ainvb'), Ainvb};
        %eigvec
        Jgrad = {-.5 * (eigvec * diag(1./eigv) * eigvec' + Ainvb * Ainvb'), Ainvb};
    end
end