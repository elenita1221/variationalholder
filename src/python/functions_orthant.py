#!/usr/bin/env python

import numpy as np
from utils import logodds, sigmoid, reshape, flatten
from scipy.integrate import quad, dblquad
from numpy.linalg import det


def factor_scaled_integral_gauss(params, theta, inv_alpha, delta):
    d = params.A.shape[0]
    alpha = 1. / inv_alpha
    theta_scaled = delta * theta
    #I = inv_alpha * gauss_integral(np.dot(params.A, alpha) + np.diag(theta_scaled[:d]) * alpha,\
    I = inv_alpha * gauss_integral(params.A * alpha + np.diag(theta_scaled[:d]) * alpha,\
    			params.b * alpha + theta_scaled[d:] * alpha)
    I_grad = 0
    return (I, I_grad)


def gauss_integral(A, b):
    detA = det(A);
    if detA < 1e-12:
        J = np.inf;
#        Jgrad = np.inf;
    else:
        d = A.shape[0]
        J = d / 2 *np.log(2*np.pi) - 0.5*np.log(detA) + 0.5*np.sum(b*np.linalg.solve(A, b))
#        Jgrad = 0
    return J


def factor_scaled_integral_univ(log_func,theta,inv_alpha,delta,L=None):
    """
    factor_scaled_integral_univ

    L are lipschitz constants for the factors derivatives
    """
    theta = reshape(theta,(theta.size/2,2))
    d = theta.shape[0]
    theta_mod = delta * theta / inv_alpha
    if L is None:
        L = np.ones(len(log_func)) * 0.01 # to avoid integrating the step function over reals
    ints = np.zeros(d)
    for i in range(d):
        if L[i]/inv_alpha < theta_mod[i,0]:  # numerical check that the integral is finite      
            wp = 1/np.sqrt(np.abs(theta_mod[i,0]));        
#            ints[i] = log(integral(lambda t: np.exp(log_func[i](t)/inv_alpha - 0.5*theta_mod[i,0]*np.power(t, 2) + theta_mod[i,1]*t),-inf,inf,'Waypoints',[-wp 0 wp]));
            ints[i] = np.log(quad(lambda t: np.exp(log_func[i](t)/inv_alpha - 0.5*theta_mod[i,0]*np.power(t, 2) + theta_mod[i,1]*t),-np.inf,np.inf)[0])#,'Waypoints',[-wp 0 wp]));
        else:   
            ints[i] = np.inf
            break
    I = inv_alpha * np.sum(ints)
    I_grad = 0
    return (I, I_grad)


def upper_bound_logpartition(fg,theta,inv_alpha,delta=1):
    logIfg = np.zeros(2)
    [logIfg[0], logIf_grad] = factor_scaled_integral_univ(fg[0],theta,inv_alpha,delta)
    [logIfg[1], logIg_grad] = factor_scaled_integral_gauss(fg[1],theta,1-inv_alpha,-delta)
    logIfg = np.real(logIfg)
    logIbar = np.sum(logIfg)
    logIbar_grad = logIf_grad + logIg_grad
    return (logIbar, logIbar_grad, logIfg)
