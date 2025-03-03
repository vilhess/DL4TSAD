# Highly copied from:
"""
Created on Mon Dec 12 10:08:16 2016

@author: Alban Siffer 
@company: Amossys
@license: GNU GPLv3
"""

import numpy as np 
from scipy.optimize import minimize
from math import log

def SetInitThreshold(data, seuil=0.98):
    sorted = np.sort(data)
    return sorted[int(seuil*len(sorted))]

def rootsFinder(fun, jac, bounds, npoints):

    step = (bounds[1] - bounds[0]) / (npoints + 1)
    X0 = np.arange(bounds[0] + step, bounds[1], step)

    def objFun(X, f, jac):
        g = 0
        j = np.zeros(X.shape)
        i = 0
        for x in X:
            fx = f(x)
            g = g + fx ** 2
            j[i] = 2 * fx * jac(x)
            i = i + 1
        return g, j

    opt = minimize(lambda X: objFun(X, fun, jac), X0,
                    method='L-BFGS-B',
                    jac=True, bounds=[bounds] * len(X0))

    X = opt.x
    np.round(X, decimals=5)
    return np.unique(X)

def log_likelihood(Y, gamma, sigma):
    n = Y.size
    if gamma != 0:
        tau = gamma / sigma
        L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
    else:
        L = n * (1 + log(Y.mean()))
    return L

def grimshaw(peaks, epsilon=1e-8, n_points=10):


    def u(s):
        return 1 + np.log(s).mean()

    def v(s):
        return np.mean(1 / s)

    def w(Y, t):
        s = 1 + t * Y
        us = u(s)
        vs = v(s)
        return us * vs - 1

    def jac_w(Y, t):
        s = 1 + t * Y
        us = u(s)
        vs = v(s)
        jac_us = (1 / t) * (1 - vs)
        jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
        return us * jac_vs + vs * jac_us

    Ym = peaks.min()
    YM = peaks.max()
    Ymean = peaks.mean()

    a = -1 / YM
    if abs(a) < 2 * epsilon:
        epsilon = abs(a) / n_points

    a = a + epsilon
    b = 2 * (Ymean - Ym) / (Ymean * Ym)
    c = 2 * (Ymean - Ym) / (Ym ** 2)

    # We look for possible roots
    left_zeros = rootsFinder(lambda t: w(peaks, t),
                                    lambda t: jac_w(peaks, t),
                                    (a + epsilon, -epsilon),
                                    n_points)
                                    
    right_zeros = rootsFinder(lambda t: w(peaks, t),
                                    lambda t: jac_w(peaks, t),
                                    (b, c),
                                    n_points)

    # all the possible roots
    zeros = np.concatenate((left_zeros, right_zeros))

    # 0 is always a solution so we initialize with it
    gamma_best = 0
    sigma_best = Ymean
    ll_best = log_likelihood(peaks, gamma_best, sigma_best)

    # we look for better candidates
    for z in zeros:
        gamma = u(1 + z * peaks) - 1
        sigma = gamma / z
        ll = log_likelihood(peaks, gamma, sigma)
        if ll > ll_best:
            gamma_best = gamma
            sigma_best = sigma
            ll_best = ll

    return gamma_best, sigma_best, ll_best

def quantile(peaks, n, gamma, sigma, init_threshold, proba=1e-4):
    r = n * proba / len(peaks)
    if gamma!=0:
        return init_threshold + (sigma/gamma) * (pow(r, -gamma) - 1)
    else:
        return init_threshold - sigma * log(r)
    

def spot(init_set, stream_set, init_seuil=0.98, proba=1e-4, n_points=10):
    N = len(init_set)

    t = SetInitThreshold(init_set, init_seuil)
    peaks = init_set[init_set>t] - t
    gamma, sigma, _ = grimshaw(peaks, n_points=n_points)
    new_t = quantile(peaks, N, gamma, sigma, t, proba)
    
    anomalies = []
    values = []
    thresholds = [new_t]

    for i, x in enumerate(stream_set):
        if x>new_t:
            anomalies.append(i)
            values.append(x)
        elif x>t:
            y = x-t
            N = N+1
            peaks = np.append(peaks, y)
            gamma, sigma, _ = grimshaw(peaks)
            new_t = quantile(peaks, N, gamma, sigma, t, proba)
        else:
            N = N+1
        thresholds.append(new_t)
    return anomalies, thresholds[:-1]

def dspot(init_set, stream_set, ws=10, init_seuil=0.98, proba=1e-4, n_points=10):

    N = len(init_set)

    window = init_set[:ws]
    m = window.mean()

    init_set = init_set[ws:]
    Xs = []
    for val in init_set:
        X = val - m
        Xs.append(X)
        window = window[1:]
        window = np.append(window, val)
        m = window.mean()

    Xs = np.array(Xs)

    t = SetInitThreshold(Xs, init_seuil)
    peaks = Xs[Xs>t] - t
    gamma, sigma, _ = grimshaw(peaks, n_points=n_points)
    new_t = quantile(peaks, N, gamma, sigma, t, proba)

    anomalies = []
    values = []
    thresholds = [new_t]
    normalized = []

    for i, x in enumerate(stream_set):
        Xs = x - m
        normalized.append(Xs)
        if Xs>new_t:
            anomalies.append(i)
            values.append(Xs)

        elif Xs>t:
            y = Xs - t
            N = N+1
            peaks = np.append(peaks, y)
            gamma, sigma, _ = grimshaw(peaks)
            new_t = quantile(peaks, N, gamma, sigma, t, proba)
            window = window[1:]
            window = np.append(window, x)
            m = window.mean()
        
        else:
            N = N+1
            window = window[1:]
            window = np.append(window, x)
            m = window.mean()

        thresholds.append(new_t)
    return anomalies, thresholds, normalized, values