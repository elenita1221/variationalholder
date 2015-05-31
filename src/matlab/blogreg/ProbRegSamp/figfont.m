% modifies marker size, linewidth, legend size etc

setaxs = 'set(get(gca,';
fontsty = [' ''Fontname'',''times'', ''FontSize'',12, ''Fontweight'',''normal'', '...
            ,' ''Fontangle'',''normal''); '];
%fontsty = [' ''FontSize'',11); '];
        
eval( [setaxs '''xlabel''' '),' fontsty] );
eval( [setaxs '''ylabel''' '),' fontsty] );
eval( [setaxs '''zlabel''' '),' fontsty] );
eval( [setaxs '''title'''  '),' fontsty] );
eval( ['set(gca,'               fontsty] );

hand=get(gca,'children');
set(hand,'LineWidth',1.7);
set(hand,'MarkerSize',5);

leg = legend;
eval( ['set(leg,'  fontsty] );

