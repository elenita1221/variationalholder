function [J,Jgrad] = gauss_integral(A,b)

detA = det(A);
if detA<eps
    J = inf;
    Jgrad = inf;
else
    d = size(A,1);
    J = d/2*log(2*pi) - .5*log(detA) + .5*b'*(A\b);
    Jgrad = 0;
end