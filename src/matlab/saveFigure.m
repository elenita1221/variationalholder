function saveFigure(figname)
% saveFigure(figname) saves the current figure as both figname.fig and figname.eps
saveas(gcf,[ figname '.fig']);
set(gcf,'PaperPositionMode','auto')
tmpeval=['print -depsc -r0 ' figname '.eps'];eval(tmpeval)