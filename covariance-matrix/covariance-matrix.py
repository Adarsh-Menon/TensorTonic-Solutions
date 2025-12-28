import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        return None

    N , D = X.shape

    if N < 2:
        return None

    mew = np.mean(X,axis=0)


    x_cen = X - mew
    cov = (1/(N-1)) * (x_cen.T @ x_cen)
    return cov