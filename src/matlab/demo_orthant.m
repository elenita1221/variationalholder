% demo_gaussian_univariate_integrals with orthants or box constraints

figure(1)
clf;
box = 0; % do we constraint the space to be the [-1 1]^d hypercube?
d = 2; % dimension of the truncated Gaussian distribution to be integrated
toprint = 0;%create pdf file with the figures at the end
K = 3 %number of settings = number of rows in the plot

for k=1:K % loop over 3 experiments 
    % The 3 settings show different values for Holder exponents p and q
    if box
        if k==1||k==3
            cor = -0.7;
        else
            cor = .01;
        end
        sc = diag(sqrt([10 5]));
        A = sc*[1 -cor;-cor 1]*sc;
        if k==2
            A = A/2;
        elseif k==3
            A = A*1.3;
        end
        Achol = chol(A);
        if k==1
            b = A*[.5;.5];
        elseif k==2
            b = A*[.8;.5];            
        elseif k==3
            b = A*[.2;0.1];
        end
        lims = [-1,-1;1 1]*1.3;
    else
        if k==1
            A =     [0.5    0.57;  0.57    1];
            b = [ 1.6; 2.1]
        elseif k==2
            A =     [0.3    0.1;  0.1    .7];
            b = [ 1; 1.5]
        elseif k==3
            A =     [0.5    0.57;  0.57    1];
            b = [ 3.1; 4.3]
        end
        lims = [-1,-1;8 6];
    end
    params.b = b;
    params.A = A;
    params.Achol = chol(A);

            
    %log_step_func = @(t) cell2args({-1e10*real(t<0),0,[0]});
    if box
        log_step_func = @(t) -1e10*real(t<-1 | t>1);
    else
        log_step_func = @(t) -1e10*real(t<0);
    end
    
    %first function: step function in each of the directions
    log_f = @(t) log_step_func(t(:,1)) + log_step_func(t(:,2)); 
    %second function: Correlated Gaussian
    log_g = @(t) -.5*sum((t*params.Achol').^2,2) + t*b; 
    %pivot function: diagonal covariance Gaussian
    log_r = @(t,theta) -.5*sum((t.^2).*(ones(size(t,1),1)*theta(1:d)'),2) + t*theta(d+1:end);
    %    fg = {log_f,params};
    fg = {{log_step_func, log_step_func},params};
    
    %% PLOT
    ng = 100; % grid size
    [gx,gy] = meshgrid(linspace(lims(1,1),lims(2,1),ng),linspace(lims(1,2),lims(2,2),ng));
    gridpoints = [gx(:),gy(:)];
    valf = exp(log_f(gridpoints)); %first function
    valg = exp(log_g(gridpoints)); %second function
    
    % optimal integral
    Istar = log(integral2(@(x,y) reshape(exp(log_f([x(:) y(:)])+log_g([x(:) y(:)])),size(x)),-inf,inf,-inf,inf));
    
    % exact first moment in each of the dimension
    mxstar = exp(-Istar)*integral2(@(x,y) x.*reshape(exp(log_f([x(:) y(:)])+log_g([x(:) y(:)])),size(x)),-inf,inf,-inf,inf);
    mystar = exp(-Istar)*integral2(@(x,y) y.*reshape(exp(log_f([x(:) y(:)])+log_g([x(:) y(:)])),size(x)),-inf,inf,-inf,inf);
    
    %% checks that the VH bound is an upper bound
    rho0 = .5; % rho = 1/alpha_1 = 1 - 1/alpha_2
    theta0 = [diag(A)*.1;b/2];
    UB0 = upper_bound_logpartition(fg,theta0,rho0);
    fprintf('The exact integral is %4.3f\n', Istar)
    fprintf('The variational holder bound gives %4.3f for the initial pivot function with parameters %s', UB0, num2str(theta0))
    
    res0 = [theta0;logodd(rho0)];
    
    objfun = @(t) upper_bound_logpartition(fg,t(1:end-1),sigmoid(t(end)));
    
    [res1,UBopt1] = fminunc(objfun,res0,optimset('Display','iter','MaxFunEvals',10000,'TolX',1e-7));
    %     [res2,UBopt2] = fminunc(objfun,-res0,optimset('Display','iter','MaxFunEvals',1000));
    %     [res1,res2]
    %     [UBopt1,UBopt2]
    
    theta1 = res1(1:end-1)
    rho1 = sigmoid(res1(end))
    [UB1,~,IfIg] = upper_bound_logpartition(fg,theta1,rho1);
    
    
%    I_fr = integral2(@(x,y) reshape(exp(1./rho1*log_f([x(:) y(:)])+(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
    I_fr = integral2(@(x,y) reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
    mx_fr = 1/I_fr*integral2(@(x,y) x.*reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
    my_fr = 1/I_fr*integral2(@(x,y) y.*reshape(exp(1./rho1*log_f([x(:) y(:)])+1./rho1*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
    
    
    I_gr = integral2(@(x,y) reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
    mx_gr = 1/I_gr*integral2(@(x,y) x.*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
    my_gr = 1/I_gr*integral2(@(x,y) y.*reshape(exp(1./(1-rho1)*log_g([x(:) y(:)])-1./(1-rho1)*log_r([x(:) y(:)],theta1)),size(x)),-inf,inf,-inf,inf);
    %
    log_valr = log_r(gridpoints,theta1);
    valr = exp(log_valr);
    val_approx_fr = valf.^(1./rho1)     .* valr.^(1./rho1);
    val_approx_gr = valg.^(1./(1-rho1)) ./ valr.^(1./(1-rho1));
    
    
    ep=.00001;%nearly 0 so that the integral is a standard one \int{f*g}
    % if only the diagonal elements of A are used
    IDiago = factor_scaled_integral_univ({log_step_func, log_step_func},[diag(params.A);params.b],1-ep,1/ep);
    % if we remove the truncation
    IGauss = factor_scaled_integral_gauss(params,zeros(d*2,1),1-ep,1/ep);
    
    % truncated gaussian with diagonal covariance
    Ifr = factor_scaled_integral_univ({log_step_func, log_step_func},theta1,1-ep,1/ep);
    
    % correlated gaussian without truncation
    params_gr.A = params.A + diag(theta1(1:d));
    params_gr.b = params.b + theta1(d+1:end)
    
    Igr = factor_scaled_integral_gauss(params_gr,zeros(d*2,1),1-ep,1/ep);
    
    UB1
    [Istar IDiago IGauss UB0 UB1 Ifr Igr]
    
    %% plots
    
    marg=.05;
    lw1 = 1*(0+1);
    lw2 = 2*(0+1);
    lw3 = 2*(0+1);
    fg_ls = '-';
    fg_col = [1 1 1]*.1;
    fg_mk = 'x';
    approx_mk = '+';
    r_col = 'k';
    fr_col = [.2 .2 1];
    gr_col = [.2 1 .2];
    ms = 15;
    adjy = 2;
    titsz = 10;
    newlinec = sprintf('\n');
    
    fg2plot = reshape(valf.*valg/Istar,size(gx));
    fr2plot = reshape(val_approx_fr/I_fr,size(gx));
    gr2plot = reshape(val_approx_gr/I_gr,size(gx));
    maxi = max(max(fg2plot));
    nc = linspace(0,maxi*1.01,5);
    maxifr = max(max(fr2plot));
    ncfr = linspace(0,maxifr*1.01,5);
    maxigr = max(max(gr2plot));
    ncgr = linspace(0,maxigr*1.01,5);
    
    
    %% first plot: original function and the optimal variational function
    %figure(1);clf;axes('Position',[marg*1.9,marg,1-2.9*marg,1-(2+adjy*(k==1))*marg])
    subplot(K,3,(k-1)*K+1)
    contour(gx,gy,fg2plot,nc,'LineWidth',lw1,'LineStyle',fg_ls,'Color',fg_col);
    grid on
    hold on
    line(mxstar,mystar,'Color',fg_col,'LineWidth',lw1/2,'LineStyle','none','Marker',fg_mk,'MarkerFaceColor',fg_col,'MarkerSize',ms);
    %[h,g] = contour(gx,gy,reshape(log_valr/1000,size(gx)),5,'Color',r_col,'LineWidth',lw3);
    % set(g,'Color',[1 1 1]*.8)
    if box
        line([-1 -1 1 1 -1],[-1 1 1 -1 -1],'Color','r','LineWidth',lw3,'LineStyle','-');
    else
        line([0 0 lims(2,1)],[lims(2,2) 0 0],'Color','r','LineWidth',lw3,'LineStyle','-');
    end
    if toprint
        if k==1
            ylhand = get(gca,'ylabel');set(ylhand,'string','high corr., large trunc.','fontsize',titsz)
        elseif k==2
            ylhand = get(gca,'ylabel');set(ylhand,'string','low corr., large trunc.','fontsize',titsz)
        elseif k==3
            ylhand = get(gca,'ylabel');set(ylhand,'string','high corr., small trunc.','fontsize',titsz)
        end
    end
    if k==1
        th=title(sprintf('function fg (black)\ntruncation (red)'));
        if toprint
            set(th,'fontsize',titsz)
        end
    end
    
    xlabel(sprintf('log(Z) = %3.2f < %3.2f%s%3.2f',Istar,IfIg(1),repmat('+',IfIg(2)>0),IfIg(2)),'fontsize',titsz);
    
    %% f.*r approximation
    %     figure(2);clf;axes('Position',[marg,marg,1-2*marg,1-(2+adjy*(k==1))*marg])
    subplot(K,3,(k-1)*K+2)
    contour(gx,gy,fg2plot,nc,'LineWidth',lw1,'LineStyle',fg_ls,'Color',fg_col);
    grid on
    hold on
    line(mxstar,mystar,'Color',fg_col,'LineWidth',lw1/2,'LineStyle','none','Marker',fg_mk,'MarkerFaceColor',fg_col,'MarkerSize',ms);
    
    if box
        line([-1 -1 1 1 -1],[-1 1 1 -1 -1],'Color','r','LineWidth',lw3,'LineStyle','-');
    else
        line([0 0 lims(2,1)],[lims(2,2) 0 0],'Color','r','LineWidth',lw3,'LineStyle','-');
    end
    
    [h,g] = contour(gx,gy,fr2plot,ncfr,'LineWidth',lw2,'Color',fr_col);
    line(mx_fr,my_fr,'Color',fr_col,'LineWidth',lw2,'LineStyle','none','Marker',approx_mk,'MarkerFaceColor',fr_col,'MarkerSize',ms);
    
    xlabel(sprintf('p=%3.2f, log(I_f) = %3.2f',1/rho1,IfIg(1)),'fontsize',titsz)
    if k==1
        th=title(['truncated approximation' newlinec 'f^pr^p (blue)']);
        if toprint
            set(th,'fontsize',titsz)
        end
    end
    
    
    %% g.*r approximation
    %     figure(3);clf;axes('Position',[marg,marg,1-2*marg,1-(2+adjy*(k==1))*marg])
    subplot(K,3,(k-1)*K+3)
    contour(gx,gy,fg2plot,nc,'LineWidth',lw1,'LineStyle',fg_ls,'Color',fg_col);
    grid on
    hold on
    line(mxstar,mystar,'Color',fg_col,'LineWidth',lw1/2,'LineStyle','none','Marker',fg_mk,'MarkerFaceColor',fg_col,'MarkerSize',ms);
    
    if box
        line([-1 -1 1 1 -1],[-1 1 1 -1 -1],'Color','r','LineWidth',lw3,'LineStyle','-');
    else
        line([0 0 lims(2,1)],[lims(2,2) 0 0],'Color','r','LineWidth',lw3,'LineStyle','-');
    end
    [h,g] = contour(gx,gy,gr2plot,ncgr,'LineWidth',lw2,'Color',gr_col);
    line(mx_gr,my_gr,'Color',gr_col,'LineWidth',lw2,'LineStyle','none','Marker',approx_mk,'MarkerFaceColor',gr_col,'MarkerSize',ms);
    
    xlabel(sprintf('q = %3.2f,log(I_g) = %3.2f',1/(1-rho1),IfIg(2)),'fontsize',titsz)
    if k==1
        th=title(['Gaussian approximation' newlinec 'g^qr^{-q} (green)']);
        if toprint
            set(th,'fontsize',titsz)
        end
    end
    
    if toprint
        orient portrait
        %set(findobj('Type','patch'),'LineWidth',4)
        set(findobj('Type','Axes'),'FontSize',titsz)
        gopts = {'-dpng'};
        if box
            print(gopts{:},['TruncGaussBox' num2str(k) '.pdf'])
        else
            print(gopts{:},['TruncGaussOrthant' num2str(k) '.pdf'])
        end
    end
end
