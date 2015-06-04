#!/usr/bin/env python

import scipy.stats
from scipy.stats import truncnorm, norm
import numpy as np

# VB where approximating distribution is univariate truncated gaussian

mu = -1.
sd = 10.
#
low = 0.
up = np.inf
#a = 0
#b = np.inf
a, b = (low - mu) / sd, (up - mu) / sd

r = truncnorm.rvs(a, b, loc=mu, scale=sd, size=1000000)
emp = np.mean((r - mu) ** 2)
mu_over_sd = mu / sd
#theory = sd ** 2 - mu * sd * np.exp(-0.5*(mu_over_sd**2)) / np.sqrt(2*np.pi) / scipy.stats.norm.cdf(mu_over_sd)
theory = sd ** 2 - mu * sd / np.sqrt(2*np.pi) * np.exp(-0.5*(mu_over_sd**2) -scipy.stats.norm.logcdf(mu_over_sd))

print 'expectation of interest: theory = %.3f, empirical = %.3f' % (theory, emp)



# simpler VB: VB where approximating distribution is univariate gaussian
mu = -10.
sd = 10.
mu_over_sd = mu / sd
x = norm.rvs(loc=mu, scale=sd, size=10000000)
theory_moments = [ scipy.stats.norm.cdf(mu_over_sd) * (mu + sd / np.sqrt(2*np.pi) * np.exp(-0.5*(mu_over_sd**2) -scipy.stats.norm.logcdf(mu_over_sd))), \
                    scipy.stats.norm.cdf(mu_over_sd) * (mu**2 + sd**2 + mu*sd / np.sqrt(2*np.pi) * np.exp(-0.5*(mu_over_sd**2) -scipy.stats.norm.logcdf(mu_over_sd)))]
empirical_moments = [np.mean(x * (x>0)), np.mean((x ** 2) * (x > 0))]
print 'moments: theoretical = %s, practical = %s' % (theory_moments, empirical_moments)
                    
