#!/usr/bin/env python

import numpy as np
from utils import logodds, sigmoid, empty, flatten, reshape
from functions_orthant import *
from matplotlib import pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.optimize import fmin, fmin_bfgs


box = False # do we constraint the space to be the [-1 1]^d hypercube?
d = 2 # dimension of the truncated Gaussian distribution to be integrated
toprint = False # create pdf file with the figures at the end
K = 3 # number of settings = number of rows in the plot
PLOT = False
INF = np.inf
assert not PLOT     # code corresponding to PLOT=True has not been checked yet
if PLOT:
    plt.figure(1)
    plt.clf()
s_print = '*' * 20

for k in np.arange(1, K+1): # loop over 3 experiments 
    # The 3 settings show different values for Holder exponents p and q
    print '\n%s k = %s %s' % (s_print, k, s_print)
    if box:
        if k==1 or k==3:
            cor = -0.7
        else:
            cor = .01
        sc = np.diag(np.sqrt([10, 5]))
        A = np.dot(np.dot(sc, np.array([[1, -cor], [-cor, 1]])), sc)
        if k==2:
            A = A / 2.
        elif k==3:
            A = A * 1.3
        Achol = chol(A)
        if k==1:
            b = np.dot(A, np.array([.5, .5]))
        elif k==2:
            b = np.dot(A, np.array([.8, .5])) 
        elif k==3:
            b = np.dot(A, np.array([.20, .1])) 
        lower_lims = np.array([-1,-1])
        upper_lims = np.array([1, 1]) * 1.3
    else:
        if k==1:
            A = np.array([[0.5, 0.57], [0.57, 1]])
            b = np.array([1.6, 2.1])
        elif k==2:
            A = np.array([[0.3, 0.1], [0.1, 0.7]])
            b = np.array([1, 1.5])
        elif k==3:
            A = np.array([[0.5, 0.57],  [0.57, 1]])
            b = np.array([3.1, 4.3])
        b = b * 0.
        lower_lims = np.array([-1,-1])
        upper_lims = np.array([8, 6])
    params = empty()
    params.b = b
    params.A = A
#    params.Achol = chol(A)

# FIXME: not sure why real is required below: is it float?
    if box:
        #        log_step_func = lambda t: -1e10 * np.real(np.logical_or(t<-1, t>1))
        log_step_func = lambda t: -1e10 * np.logical_or(t<-1, t>1)
    else:
#        log_step_func = lambda t: -1e10 * np.real(t<0)
        log_step_func = lambda t: -1e10 * (t<0)
    
    #first function: step function in each of the directions
    log_f_vec = lambda t: log_step_func(t[:,0]) + log_step_func(t[:,1]) 
    log_f = lambda t: log_step_func(t[0]) + log_step_func(t[1]) 
    # second function: Correlated Gaussian
# #    log_g = lambda t: -.5*sum((t*params.Achol.T).^2,2) + t*b 
    log_g_vec = lambda t: -0.5 * np.sum(t * np.dot(t, params.A), 1) + np.sum(t * params.b, 1) 
    log_g = lambda t: -0.5 * np.sum(t * np.dot(t, params.A)) + np.sum(t * params.b) 
    # pivot function: diagonal covariance Gaussian
    log_r_vec = lambda t, theta: np.sum(-0.5*np.power(t,2)*theta[:d] + t*theta[d:], 1)
    log_r = lambda t, theta: -0.5*np.sum(np.power(t,2) * theta[:d]) + np.sum(t*theta[d:])
    fg = ((log_step_func, log_step_func), params)   # FIXME: tuple or list?
    
    ## PLOT
    ng = 100 # grid size
#    [gx,gy] = meshgrid(linspace(lims(1,1),lims(2,1),ng),linspace(lims(1,2),lims(2,2),ng))
    gx, gy = np.meshgrid(np.linspace(lower_lims[0], upper_lims[0], ng), np.linspace(lower_lims[1], upper_lims[1], ng))
    gridpoints = np.c_[flatten(gx), flatten(gy)]
    valf= np.exp(log_f_vec(gridpoints)) #first function
    valg = np.exp(log_g_vec(gridpoints)) #second function
    
    # optimal integral
    Istar = np.log(dblquad(lambda y, x: np.exp(log_f(np.array([x, y])) + log_g(np.array([x, y]))),-INF,INF,lambda x:-INF,lambda x: INF)[0])
    # exact first moment in each of the dimension
    mxstar = np.exp(-Istar)*dblquad(lambda y, x: x*np.exp(log_f(np.array([x, y]))+log_g(np.array([x, y]))),-INF,INF,lambda x: -INF,lambda x: INF)[0]
    mystar = np.exp(-Istar)*dblquad(lambda y, x: y*np.exp(log_f(np.array([x, y]))+log_g(np.array([x, y]))),-INF,INF,lambda x:-INF,lambda x: INF)[0]
    print 'Istar %.3f, mxstar = %.3f, mystar = %.3f' % (Istar, mxstar, mystar)
    
    ## checks that the VH bound is an upper bound
    rho0 = .5 # rho = 1/alpha_1 = 1 - 1/alpha_2
    theta0 = np.r_[np.diag(A)*0.1, b/2.]
    print theta0.shape
    UB0 = upper_bound_logpartition(fg,theta0,rho0)[0]
    print 'The exact integral is %4.3f' % Istar
    print 'The variational holder bound gives %4.3f for the initial pivot function with parameters %s' % (UB0, theta0)
    
    res0 = np.r_[theta0, logodds(rho0)]
    objfun = lambda t: upper_bound_logpartition(fg,t[:-1],sigmoid(t[-1]))[0]
    
#    [res1,UBopt1] = fminunc(objfun,res0,optimset('Display','iter','MaxFunEvals',10000,'TolX',1e-7))
    res1 = fmin(objfun, res0, xtol=1e-7)
#    res1 = fmin_bfgs(objfun, res0)     # FIXME: seems to be numerically unstable
    
    theta1 = res1[:-1]
    rho1 = sigmoid(res1[-1])
    UB1, tmp, IfIg = upper_bound_logpartition(fg,theta1,rho1)
    print 'optimized upper bound = %f' % UB1
    
    # NOTE: I_fr and I_gr are raw values (before applying logarithm, unlike Istar)
    I_fr = dblquad(lambda y, x: np.exp(1./rho1*log_f(np.array([x, y])) + 1./rho1*log_r(np.array([x, y]),theta1)),-INF,INF,lambda x: -INF, lambda x: INF)[0]
    mx_fr = 1/I_fr*dblquad(lambda y, x: x*np.exp(1./rho1*log_f(np.array([x, y])) + 1./rho1*log_r(np.array([x, y]),theta1)),-INF,INF,lambda x:-INF,lambda x: INF)[0]
    my_fr = 1/I_fr*dblquad(lambda y, x: y*np.exp(1./rho1*log_f(np.array([x, y])) + 1./rho1*log_r(np.array([x, y]),theta1)),-INF,INF,lambda x:-INF,lambda x: INF)[0]
    print 'I_fr %.3f, mx_fr = %.3f, my_fr = %.3f' % (I_fr, mx_fr, my_fr)

    I_gr = dblquad(lambda y, x: np.exp(1./(1-rho1)*log_g(np.array([x, y])) - 1./(1-rho1)*log_r(np.array([x, y]),theta1)),-INF,INF,lambda x: -INF, lambda x: INF)[0]
    mx_gr = 1/I_gr*dblquad(lambda y, x: x*np.exp(1./(1-rho1)*log_g(np.array([x, y])) - 1./(1-rho1)*log_r(np.array([x, y]),theta1)),-INF,INF,lambda x:-INF,lambda x: INF)[0]
    my_gr = 1/I_gr*dblquad(lambda y, x: y*np.exp(1./(1-rho1)*log_g(np.array([x, y])) - 1./(1-rho1)*log_r(np.array([x, y]),theta1)),-INF,INF,lambda x:-INF,lambda x: INF)[0]
    print 'I_gr %.3f, mx_gr = %.3f, my_gr = %.3f' % (I_gr, mx_gr, my_gr)

    UB2 = np.log(I_fr)*rho1 + np.log(I_gr)*(1.-rho1)
    print 'alpha_1 = %.3f, log(I_fr)/alpha_1 = %.3f, alpha_2 = %.3f, log(I_gr)/alpha_2 = %.3f, UB2 = %.3f' % \
            (1./rho1, np.log(I_fr)*rho1, 1./(1-rho1), np.log(I_gr)*(1.-rho1), UB2)


#     log_valr = log_r_vec(gridpoints,theta1)
#     valr = np.exp(log_valr)
#     val_approx_fr = np.power(valf, 1./rho1) * np.power(valr, 1./rho1)
#     val_approx_gr = np.power(valg, 1./(1-rho1)) / np.power(valr, 1./(1-rho1))
# 
#     ep=.00001 #nearly 0 so that the integral is a standard one \int{f*g}
#     # if only the diagonal elements of A are used
#     IDiago = factor_scaled_integral_univ((log_step_func, log_step_func),np.r_[np.diag(params.A), params.b],1-ep,1/ep)[0]
#     # if we remove the truncation
#     IGauss = factor_scaled_integral_gauss(params,np.zeros(d*2),1-ep,1/ep)[0]
#     
#     # truncated gaussian with diagonal covariance
#     Ifr = factor_scaled_integral_univ((log_step_func, log_step_func),theta1,1-ep,1/ep)[0]
#     
#     # correlated gaussian without truncation
#     params_gr = empty()
#     params_gr.A = params.A + np.diag(theta1[:d])
#     params_gr.b = params.b + theta1[d:]
#     
#     Igr = factor_scaled_integral_gauss(params_gr,np.zeros(d*2),1-ep,1/ep)[0]
#     
#     print 'Istar = %.3f, IDiago = %.3f, IGauss = %.3f, UB0 = %.3f, UB1 = %.3f, Ifr = %.3f, Igr = %.3f' % \
#             (Istar, IDiago, IGauss, UB0, UB1, Ifr, Igr)
    #print np.array([Istar, IDiago, IGauss, UB0, UB1, Ifr, Igr])
#  
#    if PLOT:
#    ## plots
#        marg=.05
#        lw1 = 1*(0+1)
#        lw2 = 2*(0+1)
#        lw3 = 2*(0+1)
#        fg_ls = '-'
#        fg_col = [1 1 1]*.1
#        fg_mk = 'x'
#        approx_mk = '+'
#        r_col = 'k'
#        fr_col = [.2 .2 1]
#        gr_col = [.2 1 .2]
#        ms = 15
#        adjy = 2
#        titsz = 10
#        newlinec = sprintf('\n')
#        
#        fg2plot = reshape(valf.*valg/Istar,size(gx))
#        fr2plot = reshape(val_approx_fr/I_fr,size(gx))
#        gr2plot = reshape(val_approx_gr/I_gr,size(gx))
#        maxi = max(max(fg2plot))
#        nc = linspace(0,maxi*1.01,5)
#        maxifr = max(max(fr2plot))
#        ncfr = linspace(0,maxifr*1.01,5)
#        maxigr = max(max(gr2plot))
#        ncgr = linspace(0,maxigr*1.01,5)
#        
#        
#        ## first plot: original function and the optimal variational function
#        #figure(1)clfaxes('Position',[marg*1.9,marg,1-2.9*marg,1-(2+adjy*(k==1))*marg])
#        subplot(K,3,(k-1)*K+1)
#        contour(gx,gy,fg2plot,nc,'LineWidth',lw1,'LineStyle',fg_ls,'Color',fg_col)
#        grid on
#        hold on
#        line(mxstar,mystar,'Color',fg_col,'LineWidth',lw1/2,'LineStyle','none','Marker',fg_mk,'MarkerFaceColor',fg_col,'MarkerSize',ms)
#        #[h,g] = contour(gx,gy,reshape(log_valr/1000,size(gx)),5,'Color',r_col,'LineWidth',lw3)
#        # set(g,'Color',[1 1 1]*.8)
#        if box
#            line([-1 -1 1 1 -1],[-1 1 1 -1 -1],'Color','r','LineWidth',lw3,'LineStyle','-')
#        else
#            line([0 0 lims(2,1)],[lims(2,2) 0 0],'Color','r','LineWidth',lw3,'LineStyle','-')
#        end
#        if toprint
#            if k==1
#                ylhand = get(gca,'ylabel')set(ylhand,'string','high corr., large trunc.','fontsize',titsz)
#            elif k==2
#                ylhand = get(gca,'ylabel')set(ylhand,'string','low corr., large trunc.','fontsize',titsz)
#            elif k==3
#                ylhand = get(gca,'ylabel')set(ylhand,'string','high corr., small trunc.','fontsize',titsz)
#            end
#        end
#        if k==1
#            th=title(sprintf('function fg (black)\ntruncation (red)'))
#            if toprint
#                set(th,'fontsize',titsz)
#            end
#        end
#        
#        xlabel(sprintf('log(Z) = #3.2f < #3.2f#s#3.2f',Istar,IfIg(1),repmat('+',IfIg(2)>0),IfIg(2)),'fontsize',titsz)
#        
#        ## f.*r approximation
#        subplot(K,3,(k-1)*K+2)
#        contour(gx,gy,fg2plot,nc,'LineWidth',lw1,'LineStyle',fg_ls,'Color',fg_col)
#        grid on
#        hold on
#        line(mxstar,mystar,'Color',fg_col,'LineWidth',lw1/2,'LineStyle','none','Marker',fg_mk,'MarkerFaceColor',fg_col,'MarkerSize',ms)
#        
#        if box
#            line([-1 -1 1 1 -1],[-1 1 1 -1 -1],'Color','r','LineWidth',lw3,'LineStyle','-')
#        else
#            line([0 0 lims(2,1)],[lims(2,2) 0 0],'Color','r','LineWidth',lw3,'LineStyle','-')
#        end
#        
#        [h,g] = contour(gx,gy,fr2plot,ncfr,'LineWidth',lw2,'Color',fr_col)
#        line(mx_fr,my_fr,'Color',fr_col,'LineWidth',lw2,'LineStyle','none','Marker',approx_mk,'MarkerFaceColor',fr_col,'MarkerSize',ms)
#        
#        xlabel(sprintf('p=#3.2f, log(I_f) = #3.2f',1/rho1,IfIg(1)),'fontsize',titsz)
#        if k==1
#            th=title(['truncated approximation' newlinec 'f^pr^p (blue)'])
#            if toprint
#                set(th,'fontsize',titsz)
#            end
#        end
#        
#        
#        ## g.*r approximation
#        subplot(K,3,(k-1)*K+3)
#        contour(gx,gy,fg2plot,nc,'LineWidth',lw1,'LineStyle',fg_ls,'Color',fg_col)
#        grid on
#        hold on
#        line(mxstar,mystar,'Color',fg_col,'LineWidth',lw1/2,'LineStyle','none','Marker',fg_mk,'MarkerFaceColor',fg_col,'MarkerSize',ms)
#        
#        if box:
#            line([-1 -1 1 1 -1],[-1 1 1 -1 -1],'Color','r','LineWidth',lw3,'LineStyle','-')
#        else:
#            line([0 0 lims(2,1)],[lims(2,2) 0 0],'Color','r','LineWidth',lw3,'LineStyle','-')
#        [h,g] = contour(gx,gy,gr2plot,ncgr,'LineWidth',lw2,'Color',gr_col)
#        line(mx_gr,my_gr,'Color',gr_col,'LineWidth',lw2,'LineStyle','none','Marker',approx_mk,'MarkerFaceColor',gr_col,'MarkerSize',ms)
#        
#        xlabel(sprintf('q = #3.2f,log(I_g) = #3.2f',1/(1-rho1),IfIg(2)),'fontsize',titsz)
#        if k==1:
#            th=title(['Gaussian approximation' newlinec 'g^qr^{-q} (green)'])
#            if toprint:
#                set(th,'fontsize',titsz)
#        
#        if toprint:
#            set(findobj('Type','Axes'),'FontSize',titsz)
#            gopts = {'-dpng'}
#            if box:
#                print(gopts{:},['TruncGaussBox' num2str(k) '.pdf'])
#            else:
#                print(gopts{:},['TruncGaussOrthant' num2str(k) '.pdf'])
