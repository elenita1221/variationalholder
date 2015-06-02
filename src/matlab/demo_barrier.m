% demo_constr

Achol = randn(3,2);A = Achol'*Achol;

checkderivatives(@(t) logbarrier_gaussian(t,A), [0,0]')

% Achol = randn(3,2);A = Achol'*Achol;
% 
% 
% xp = linspace(-1,10,100);
% yp = linspace(-1,10,100);
% 
% [xg,yg] = meshgrid(xp,yp);
% 
% zg = [xg(:), yg(:)];
% 
% fg = zg(:,1);
% for i=1:length(zg)
%     fg(i) = f(zg(i,:));
% end
%contour(xg,yg,reshape(fg,size(xg)))

