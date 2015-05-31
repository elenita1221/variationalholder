#!/usr/bin/env python


import numpy as np
import random


def center_and_whiten(mat):
    n, d = mat.shape
    c = mat - np.mean(mat, 0)
    cov = np.dot(c.T, c)
    e, v = np.linalg.eig(cov)
    #rot = np.dot( np.dot(v, np.diag(1. / np.sqrt(e))), v.T) / np.sqrt(n-1)
    rot = np.dot( np.dot(v, np.diag(1. / np.sqrt(e))), v.T)
    op = np.dot(c, rot)
    return op


def flatten(mat):
    return np.ravel(mat, order='F')     # equivalent to mat(:) in matlab


def reshape(mat, mat_size):
    return np.reshape(mat, mat_size, order='F')


def sigmoid(ip):
    op = 1. / (1 + np.exp(-ip))
    return op


def logodds(ip):
    """
    ip should be a numpy array with entries between 0 and 1
    """
    op = np.log(ip) - np.log(1. - ip)
    return op


class empty(object):
    def __init__(self):
        pass


def reset_random_seed(init_id):
    # Resetting random seed
    np.random.seed(init_id * 1000)
    random.seed(init_id * 1000)
