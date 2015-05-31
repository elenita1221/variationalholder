#!/usr/bin/env python

import numpy as np
from utils import logodds, sigmoid, empty, flatten, reshape, reset_random_seed
from functions_probit import gauss_integral
from matplotlib import pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.optimize import fmin, fmin_bfgs
from scipy.stats import norm

D = 2
#D = 0
#N = 2
N = 0   #100
#assert N == 2
PLOT = False
INF = np.inf
#INF2 = 1e10
INF2 = np.inf
assert not PLOT     # code corresponding to PLOT=True has not been checked yet
if PLOT:
    plt.figure(1)
    plt.clf()
s_print = '*' * 20


def compute_moments(tau, inv_alpha_1):
    if len(tau) == 2*(D+N):
        tau_1, tau_2 = tau[:D+N], tau[D+N:]
    else:
        tau_1, tau_2 = tau, np.zeros(D+N)
        print tau_1, tau_2
    alpha_1 = 1.0 / inv_alpha_1
    inv_alpha_2 = 1 - inv_alpha_1
    alpha_2 = 1.0 /inv_alpha_2
    moment_2 = -alpha_2 * tau_2
    moment_1 = tau_2 / tau_1    # moment[D:] will be corrected below
    tau_1_N, tau_2_N = tau_1[D:], tau_2[D:]
    tmp = alpha_1 / tau_1_N * np.power(tau_2_N, 2)
    tmp2 = np.exp(-0.5 * tmp) / (np.sqrt(2 * np.pi * alpha_1 * tau_1_N) * (1. - norm.cdf(-np.sqrt(tmp))))
    moment_1[D:] += tmp2
    moment = inv_alpha_1 * moment_1 + inv_alpha_2 * moment_2
    return (moment, moment_1, moment_2)


def log_step_func(t):
    return -1e10 * (t<0)    # FIXME: use INF?


def upper_bound_logpartition(tau, inv_alpha_1):
    tau_1, tau_2 = tau[:D+N], tau[D+N:]
    tau_1_N, tau_2_N = tau_1[D:], tau_2[D:]     # first D values correspond to w
    alpha_1 = 1.0 / inv_alpha_1
    inv_alpha_2 = 1 - inv_alpha_1
    if np.any(tau_1 <= 0):
#    if np.any(tau_1 <= 0) or np.any(tau_1 > min_eigvals_A):
        print 'one of the tau_1 <= 0: setting integral_1 to INF'
        integral_1 = INF2
    else:
        integral_1 = inv_alpha_1 * (-0.5 * ((D+N)*np.log(alpha_1) + np.sum(np.log(tau_1)) ) \
                        + np.sum(norm.logcdf(np.sqrt(alpha_1)*tau_2_N/np.sqrt(tau_1_N)))) \
                        + 0.5 * np.sum(np.power(tau_2, 2) / tau_1)
    mat = A - np.diag(tau_1)
    sign, logdet = np.linalg.slogdet(mat)
    if (sign <= 0) or np.isinf(logdet):
        print 'sign = %s, logdet = %s, setting integral_2 to INF' % (sign, logdet)
        integral_2 = INF2
    else:
        try:
            integral_2 = -0.5 * inv_alpha_2 * ((D+N)*np.log(inv_alpha_2) + logdet) \
                            + 0.5 * np.sum(tau_2 * np.linalg.solve(mat, tau_2))
        except np.linalg.linalg.LinAlgError:
            integral_2 = INF2
    integral = integral_1 + integral_2
    return integral


def upper_bound_logpartition_2(tau_1, inv_alpha_1):
    """
    for mode = 2
    tau_2 is set to 0
    """
    alpha_1 = 1.0 / inv_alpha_1
    inv_alpha_2 = 1 - inv_alpha_1
    if np.any(tau_1 <= 0):
        print 'one of the tau_1 <= 0: setting integral_1 to INF'
        integral_1 = INF2
    else:
        integral_1 = inv_alpha_1 * (-0.5 * ((D+N)*np.log(alpha_1) + np.sum(np.log(tau_1)) ) + N*np.log(0.5))
    mat = A - np.diag(tau_1)
    sign, logdet = np.linalg.slogdet(mat)
    #if sign < 0:
    if (sign < 0) or np.isinf(logdet):
        print 'sign = %s, logdet = %s, setting integral_2 to INF' % (sign, logdet)
        integral_2 = INF2
    else:
        integral_2 = inv_alpha_2 * (-0.5) * ((D+N)*np.log(inv_alpha_2) + logdet)
    integral = integral_1 + integral_2
    return integral


def upper_bound_logpartition_4(tau, inv_alpha_1):
    tau_1, tau_2 = tau[:D+N], tau[D+N:]
    tau_1_N, tau_2_N = tau_1[D:], tau_2[D:]     # first D values correspond to w
    #assert len(tau_1_N) == len(tau_2_N) == 0
    alpha_1 = 1.0 / inv_alpha_1
    inv_alpha_2 = 1 - inv_alpha_1
    #if np.any(tau_1 <= 0):
    #    print 'one of the tau_1 <= 0: setting integral_1 to INF'
    if np.any(tau_1 <= 0.01 / inv_alpha_1):
        print 'one of the tau_1 <= 0.01/inv_alpha_1: setting integral_1 to INF'
        integral_1 = INF2
    else:
        integral_1 = inv_alpha_1 * (-0.5 * ((D+N)*np.log(alpha_1) + np.sum(np.log(tau_1))) \
                        + np.sum(norm.logcdf(np.sqrt(alpha_1)*tau_2_N/np.sqrt(tau_1_N)))) \
                        + 0.5 * np.sum(np.power(tau_2, 2) / tau_1)
    mat = A - np.diag(tau_1)
    sign, logdet = np.linalg.slogdet(mat)
    if (sign < 0) or np.isinf(logdet):
        print 'sign = %s, logdet = %s, setting integral_2 to INF' % (sign, logdet)
        integral_2 = INF2
    else:
        try:
            integral_2 = inv_alpha_2 * (-0.5) * ((D+N)*np.log(inv_alpha_2) + logdet) + 0.5 * np.sum(tau_2 * np.linalg.solve(mat, tau_2))
        except np.linalg.linalg.LinAlgError:
            integral_2 = INF2
    integral = integral_1 + integral_2
    return integral


def upper_bound_logpartition_5(tau, inv_alpha_1):
    tau_1, tau_2 = tau[:D+N], tau[D+N:]
    #tau_1_N, tau_2_N = tau_1[D:], tau_2[D:]     # first D values correspond to w
    #assert len(tau_1_N) == len(tau_2_N) == 0
    alpha_1 = 1. / inv_alpha_1
    inv_alpha_2 = 1. - inv_alpha_1
    alpha_2 = 1. / inv_alpha_2
    integral_2 = inv_alpha_2 * gauss_integral(alpha_2 * (A - np.diag(tau_1)), -alpha_2 * tau_2)
    L = np.ones(D) * 0.01 # to avoid integrating the step function over reals
    ints = np.zeros(D)
    tau_1_mod = alpha_1 * tau_1
    tau_2_mod = alpha_1 * tau_2
    for i in range(D):
        if L[i]/inv_alpha_1 < tau_1_mod[i]:  # numerical check that the integral is finite      
            ints[i] = np.log(quad(lambda t: np.exp(log_step_func(t)/inv_alpha_1 - 0.5*tau_1_mod[i]*np.power(t, 2) + tau_2_mod[i]*t),-np.inf,np.inf)[0])
            #ints[i] = np.log(quad(lambda t: np.exp(- 0.5*tau_1_mod[i]*np.power(t, 2) + tau_2_mod[i]*t),-np.inf,np.inf)[0])
        else:   
            ints[i] = np.inf
            break
    integral_1 = inv_alpha_1 * np.sum(ints)
    integral = integral_1 + integral_2
    assert np.imag(integral) == 0.
    return integral


reset_random_seed(123)

#mode = 1
#mode = 2
#mode = 3
#mode = 4
mode = 5

if not (mode == 4 or mode == 5):
    if N == 2:
        Y = np.array([1, -1])
    else:
        Y = 2 * np.round(np.random.rand(N)) - 1
    mag = 1. / np.sqrt(D)     # how separable should the features be?
    X = 0. * np.random.randn(N, D) + mag * np.ones((N, D)) * Y[:, np.newaxis]
    print Y
    print X

    precision_vec = np.random.rand(D) * 1
    precision_mat = np.diag(precision_vec)
    Z = X * Y[:, np.newaxis]
    A = np.r_[ np.c_[np.dot(Z.T, Z)+precision_mat, -Z.T], np.c_[-Z, np.eye(N)] ]
    sign_A, logdet_A = np.linalg.slogdet(A)
    eigvals_A = np.linalg.eigvalsh(A)
    #print A
    #print 'sorted eigen values are '
    #print np.sort(eigvals_A)
    min_eigvals_A = np.min(np.real(eigvals_A))
    assert sign_A > 0


if mode == 1:
    rho0 = inv_alpha_1_init = 0.5
    #theta0 = tau_init = np.r_[np.random.rand(D+N), np.random.randn(D+N)]
    #theta0 = tau_init = np.r_[np.diag(A) * 0.1, np.zeros(D+N)]
    theta0 = tau_init = np.r_[np.ones(D+N) * min_eigvals_A * 0.5, np.zeros(D+N)]
    #theta0 = tau_init = np.r_[np.ones(D+N) * min_eigvals_A * 0.9, np.zeros(D+N)]
    res0 = np.r_[theta0, logodds(rho0)]
    objfun = lambda t: upper_bound_logpartition(t[:-1], sigmoid(t[-1]))
    UB0 = upper_bound_logpartition(tau_init, inv_alpha_1_init)
    print 'The variational holder bound gives %4.3f for the initial pivot function' % UB0
    #res1 = fmin(objfun, res0, xtol=1e-7)
    res1 = fmin_bfgs(objfun, res0)
    theta1 = res1[:-1]
    rho1 = sigmoid(res1[-1])
    UB1 = upper_bound_logpartition(theta1,rho1)
elif mode == 2:
    # optimize only alpha_1
    # set tau_2 to zero and set tau_1 heuristically 
    rho0 = inv_alpha_1_init = 0.5
    max_iter = 10
    inv_alpha = rho0
    for itr in range(max_iter):
        #tau_current = np.r_[np.diag(A) * inv_alpha, np.zeros(D+N)]
        tau_current = np.diag(A) * inv_alpha
        UB = upper_bound_logpartition_2(tau_current, inv_alpha)
        print 'itr = %3d, UB = %.3f, inv_alpha = %.3f' % (itr, UB, inv_alpha)
        objfun = lambda logodds_rho: upper_bound_logpartition_2(tau_current, sigmoid(logodds_rho))
        res = logodds(inv_alpha)
        #res1 = fmin(objfun, res, xtol=1e-7)
        res_op = fmin_bfgs(objfun, res)
        inv_alpha = sigmoid(res_op)
    theta1, rho1 = tau_current, res     # setting to penultimate value
    UB1 = upper_bound_logpartition_2(theta1,rho1)
elif mode == 3:
    # optimize only alpha_1 and tau_1
    rho0 = inv_alpha_1_init = 0.5
    theta0 = np.diag(A) * 0.1
    UB0 = upper_bound_logpartition_2(theta0, rho0)
    print 'The variational holder bound gives %4.3f for the initial pivot function' % UB0
    objfun = lambda t: upper_bound_logpartition_2(t[:-1], sigmoid(t[-1]))
    res0 = np.r_[theta0, rho0]
    res1 = fmin_bfgs(objfun, res0)
    theta1, rho1 = res1[:-1], sigmoid(res1[-1])
    UB1 = upper_bound_logpartition_2(theta1,rho1)
elif mode == 4 or mode == 5:
    #assert N == 2 and D == 0
    assert D == 2 and N == 0
    k = 1
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
    rho0 = inv_alpha_1_init = 0.5
    #theta0 = tau_init = np.r_[np.random.rand(D+N), np.random.randn(D+N)]
    theta0 = tau_init = np.r_[np.diag(A) * 0.1, np.zeros(D+N)]
    res0 = np.r_[theta0, logodds(rho0)]
    if mode == 4:
        objfun = lambda t: upper_bound_logpartition_4(t[:-1], sigmoid(t[-1]))
        UB0 = upper_bound_logpartition_4(tau_init, inv_alpha_1_init)
    else:
        objfun = lambda t: upper_bound_logpartition_5(t[:-1], sigmoid(t[-1]))
        UB0 = upper_bound_logpartition_5(tau_init, inv_alpha_1_init)
    print 'The variational holder bound gives %4.3f for the initial pivot function' % UB0
    print 'bound: exact = %f, numerical = %f' % (upper_bound_logpartition_4(tau_init, inv_alpha_1_init), \
            upper_bound_logpartition_5(tau_init, inv_alpha_1_init))
    res1 = fmin(objfun, res0, xtol=1e-7)
    #res1 = fmin_bfgs(objfun, res0)
    theta1 = res1[:-1]
    rho1 = sigmoid(res1[-1])
    if mode == 4:
        UB1 = upper_bound_logpartition_4(theta1,rho1)
    else:
        UB1 = upper_bound_logpartition_5(theta1,rho1)
    print 'optimized rho1 = %.3f, alpha_1 = %.3f' % (rho1, 1./rho1)


print 'optimized upper bound = %f (initial = %f)' % (UB1, UB0)
moment, moment_1, moment_2 = compute_moments(theta1, rho1)
print 'moment = %s\nmoment_1 = %s\nmoment_2 = %s' % (moment, moment_1, moment_2)
