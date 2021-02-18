from math import exp

def rbf_network_cython(X, beta):


    N = X.shape[0]
    D = X.shape[1]
    cdef double Y[N]

    for i in range(N):
        for j in range(N):
            r = 0
            for d in range(D):
                r += (X[j, d] - X[i, d]) ** 2
            r = r**0.5
            Y[i] += beta[j] * exp(-(r * theta)**2)

    return Y


# Cython  implementation of a Radial Basis Function (RBF) approximation scheme
# 
# TODO: Write the Cython implementation in a separate fastloop.pyx file, compile and import it here
# 
# from fastloop import rbf_network_cython
