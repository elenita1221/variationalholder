#!/usr/bin/env python

import cvxpy as cvx
import numpy as np
from scipy.stats import norm

N = 2
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
#b = b * 0.
#A = np.r_[np.c_[1, 0], np.c_[0, 1]]
log_2_pi = np.log(2 * np.pi)
log_normcdf = norm.logcdf

# Declare optimization variables
tau_1 = cvx.Variable(N)
#tau_2 = cvx.Variable(N)
tau_2 = np.zeros(N)
print tau_2, b
print b.shape, tau_2.shape
#inv_alpha_1 = cvx.Variable(1)
inv_alpha_1 = 0.5

# specify objective function
#obj = cvx.Minimize(-cvx.sum_entries(cvx.log(tau_1)) - cvx.log_det(A - cvx.diag(tau_1)))
#obj = cvx.Minimize(-cvx.sum_entries(cvx.log(tau_1)) - cvx.log_det(A - cvx.diag(tau_1)))
# original
#obj = cvx.Minimize( 0.5*N*(inv_alpha_1-1)*log_2_pi - 0.5*inv_alpha_1*(N*cvx.log(1/inv_alpha_1) + cvx.sum_entries(cvx.log(tau_1))) + 0.5*cvx.sum_entries(cvx.square(tau_2)/tau_1) + inv_alpha_1*cvx.sum_entries(log_normcdf(tau_2*cvx.sqrt(1/(inv_alpha_1*tau_1)))) +0.5*N*(1-inv_alpha_1)*cvx.log(1-inv_alpha_1) -0.5*(1-inv_alpha_1)*cvx.log_det(A-cvx.diag(tau_1)) + 0.5*cvx.matrix_frac(b-tau_2, A-cvx.diag(tau_1)) )
# modifications
#obj = cvx.Minimize( 0.5*N*(inv_alpha_1-1)*log_2_pi - 0.5*inv_alpha_1*(-N*cvx.log(inv_alpha_1) + cvx.sum_entries(cvx.log(tau_1))) + 0.5*cvx.matrix_frac(tau_2, cvx.diag(tau_1)) + inv_alpha_1*cvx.sum_entries(log_normcdf(tau_2.T*cvx.inv_pos(cvx.sqrt(inv_alpha_1*tau_1)))) +0.5*N*(1-inv_alpha_1)*cvx.log(1-inv_alpha_1) -0.5*(1-inv_alpha_1)*cvx.log_det(A-cvx.diag(tau_1)) + 0.5*cvx.matrix_frac(b-tau_2, A-cvx.diag(tau_1)) )
obj = cvx.Minimize( 0.5*N*(inv_alpha_1-1)*log_2_pi - 0.5*inv_alpha_1*(-N*cvx.log(inv_alpha_1) + cvx.sum_entries(cvx.log(tau_1))) + 0.5*cvx.matrix_frac(tau_2, cvx.diag(tau_1)) + cvx.sum_entries(inv_alpha_1*log_normcdf(cvx.inv_pos(cvx.sqrt(inv_alpha_1*tau_1)))) )# +0.5*N*(1-inv_alpha_1)*cvx.log(1-inv_alpha_1) -0.5*(1-inv_alpha_1)*cvx.log_det(A-cvx.diag(tau_1)) + 0.5*cvx.matrix_frac(b-tau_2, A-cvx.diag(tau_1)) )


#def upper_bound_logpartition(tau, inv_alpha_1):
#    tau_1, tau_2 = tau[:D+N], tau[D+N:]
#    tau_1_N, tau_2_N = tau_1[D:], tau_2[D:]     # first D values correspond to w
#    alpha_1 = 1.0 / inv_alpha_1
#    inv_alpha_2 = 1 - inv_alpha_1
#    if np.any(tau_1 <= 0):
#        integral_1 = INF2
#    else:
#        integral_1 = inv_alpha_1 * (-0.5 * ((D+N)*np.log(alpha_1) + np.sum(np.log(tau_1)) ) \
#                        + np.sum(norm.logcdf(np.sqrt(alpha_1)*tau_2_N/np.sqrt(tau_1_N)))) \
#                        + 0.5 * np.sum(np.power(tau_2, 2) / tau_1)
#    mat = A - np.diag(tau_1)
#    sign, logdet = np.linalg.slogdet(mat)
#    if (sign <= 0) or np.isinf(logdet):
#        integral_2 = INF2
#    else:
#        try:
#            integral_2 = -0.5 * inv_alpha_2 * (-(D+N)*np.log(inv_alpha_2) + logdet) \
#                            + 0.5 * np.sum(tau_2 * np.linalg.solve(mat, tau_2))
#        except np.linalg.linalg.LinAlgError:
#            integral_2 = INF2
#    integral = integral_1 + integral_2
#    return integral

# specify contraints
#constraints = [tau_1 >= 0, inv_alpha_1 <= 1, inv_alpha_1 >= 0, A - cvx.diag(tau_1) == cvx.Semidef(N)]
constraints = [tau_1 >= 0, A - cvx.diag(tau_1) == cvx.Semidef(N)]

# Form and solve optimization problem
prob = cvx.Problem(obj, constraints)
prob.solve()
if prob.status != cvx.OPTIMAL:
    raise Exception('CVXPY Error')

print 'value of objective function = %.2f' % prob.value
print 'solution of tau_1'
print tau_1.value
#print 'solution of inv_alpha_1'
#print inv_alpha_1.value

