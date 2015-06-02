function [grad,hess] = checkderivatives(func,x,varargin)
%  CHECKDERIVATIVES - Check if a function returns the correct gradient and hessian matrix
% 
% checkderivatives(str,p0);
%
% Check the Hessian matrix: 
%  [grad,hess] = checkderivatives(f,p0,'hessian',1)
% %example with cos function
% [grad,hess] = checkderivatives(@(x) twoargs_func(x,@(x) {sin(x),cos(x),-sin(x)}),1,'hessian',1)


hessian = 0;
% options

[d1,d2] = size(x);
x=x(:);
d = length(x); 
dec = 1e-5;
add = ceil(d*log(d));
n = 1+d+d*d*hessian+add;
projFunc = [];

SIMPLE_ARGS_DEF




% if ~isempty(varargin)
%     a=strcat('varargin{',num2str((1:length(varargin))'),'},')';a(end)=')';
%     command = [str '(xcur,' a(:)' ';'];
% else
%     command = [str '(xcur);'];
% end

%computation 
xcur = reshape(x,d1,d2);

if hessian
    [lval,grad0,hess0] = func(xcur);
else
    [lval,grad0] = func(xcur);
    hess0 = [];
end
grad0 = grad0(:);    

%checking
f=zeros(n,1);
z = (rand(n,d)-.5)*dec; %random directions
for i = 1:n
    xcur = reshape(x+z(i,:)',d1,d2);
    if ~isempty(projFunc)
        xcur = projFunc(xcur);
    end
    f(i) = func(xcur);
end
indlower = find(tril(ones(d),-1));
[indlower_r,indlower_c] = ind2sub([d,d],indlower');

M = [z,.5*z.^2,z(:,indlower_r).*z(:,indlower_c)];
R = M\(f-lval);

grad = R(1:d);
low = zeros(d);low(indlower) = R(2*d+1:end);
hess = diag(R(d+1:2*d)) + low + low';

    

if ~isempty(grad0)
    falsegrad = find(abs(grad0-grad)>sqrt(dec));
    if ~isempty(falsegrad)
        disp(['gradient in directions ' num2str(falsegrad') ' is false']);
    else
        disp('gradient seems to be OK');
    end
    if nargout==0 & isempty(hess0)
        disp('[gradient_of_function|numeric_gradient]');
        disp(full([grad0 grad]))
        clear grad
    end
end


if ~isempty(hess0)
    [fhr,fhc] = find(abs(hess0-hess)>100*sqrt(dec));
    if ~isempty(fhr)
            disp('hessian is false in the following positions:')
            for i=1:length(fhr)
                fprintf('(%d,%d) --> returned %f instead of %f\n',fhr(i),fhc(i),hess(fhr(i),fhc(i)),hess0(fhr(i),fhc(i)));
            end
    else
            disp('hessian seems to be OK');
    end
end



