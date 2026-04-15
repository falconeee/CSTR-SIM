import numpy as np
from sklearn.preprocessing import StandardScaler


def t2t(X, alfa=0.95, limvar=0.95, force_a=None):
    n, m = X.shape
    if n == 0 or m == 0:
        return None
    # assert n > 0, 't2t> Need at least one sample'
    # assert m > 0, 't2t> Need at least one variable'

    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler(with_std=False) # only centralize, do not make unit variance
    # print('X before scaling=\n', X, 'shape=', X.shape)
    scaler.fit(X)
    X = scaler.transform(X)
    # print('mu=\n', scaler.mean_, 'shape=', scaler.mean_.shape)
    # if scaler.scale_ is not None:
    #     print('scale=\n', scaler.scale_, 'shape=', scaler.scale_.shape)
    # print('X after scaling=\n', X, 'shape=', X.shape)
    # print('mu new=\n', X.mean(axis=0))
    # print('std new=\n', X.std(axis=0))
    # raise Exception()

    M = {}
    M['n'] = n  # Number of samples used for training
    M['m'] = m  # Number of variables of original data matrix
    M['alfa'] = alfa  # used in 'threshold'
    M['limvar'] = limvar
    M['mu'] = scaler.mean_
    if scaler.scale_ is not None:
        M['st'] = scaler.scale_
    else:
        M['st'] = X.std(axis=0, ddof=1)

    if m == 1:  # only one variable
        M['P'] = np.eye(1)  # Loading vectors
        M['Ptil'] = None
        var = X.var()
        M['S'] = np.eye(1) * var
        M['s'] = np.array([var, ])
        M['a'] = m  # Number of principal components = 1
        M['r_var'] = None
        return M

    ddof = None
    # ddof = 1
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    Sigma = np.cov(X, rowvar=False, ddof=ddof)  # Covariance matrix
    # print('Sigma=\n', Sigma, 'shape=', Sigma.shape)
    [u, s, vh] = np.linalg.svd(Sigma, full_matrices=False)
    # print('u=\n', u, 'shape=', u.shape)
    # print('s=\n', s, 'shape=', s.shape)
    # print('vh=\n', vh, 'shape=', vh.shape)

    v = u   # Matlab svd returns v as u
    if limvar > 0.99:  # Use all variables as PCs
        a = m
    else:
        cumvar = np.cumsum(s) / np.sum(s)
        print('force_a=', force_a, 'limvar=', limvar, 'cumvar=', cumvar)
        a = 1 + np.argmax(cumvar > limvar)
    # print('a=', a)
    assert a <= m, 'Number of PC too high'
    if force_a is not None and m > 1:
        print('Forcing number of PC to: ', force_a)
        a = force_a  # Force 2-D (good for ellipse plot)
    P = v[:, :a]
    Ptil = v[:, a:]
    # print('P=\n', P, 'shape=', P.shape)
    M['P'] = P  # Loading vectors, dimension (m x a)
    M['Ptil'] = Ptil  # Loading vectors of residuals, dimension (m x m-a)
    M['S'] = np.diag(s)  # Eigenvalue matrix, dimension (m x m)
    M['s'] = s  # Eigenvalue vector, dimension m
    M['a'] = a  # Number of principal components. 1 <= a <= m
    if a < m:
        r = np.dot(X, np.eye(m) - np.dot(P, P.T))
        M['r_var'] = np.var(r, axis=0, ddof=1)  # Variance of residuals
    else:
        M['r_var'] = None
    return M
