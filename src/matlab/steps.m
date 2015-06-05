function x = steps(p)
% STEPS - stairs function where steps are located in given positions
%
% x = steps(p) 
%   p: length of the steps
%   x: stairs function of length sum(p)
% 
% example:
% --------
% 
% steps([3 4 1 0 2 0])
% % compares to 
% cumsum([1;full(sparse(cumsum([3 4 1 0 2 0]),ones(1,6),ones(1,6)))])

[val,~,nums] = find(p(:));
pos = cumsum([1;nums(:)]);
%n = pos(end)-1;
lp = length(pos);
cs = cumsum(accumarray(pos,ones(lp,1)));
x = val(cs(1:end-1));
