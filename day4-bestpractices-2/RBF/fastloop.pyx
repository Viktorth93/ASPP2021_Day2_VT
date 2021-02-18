from math import exp

def rbf_network_cython(X,  beta, double theta):


    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef double Y[1000]
    cdef int i,j,d
    cdef double r
    cdef double xj, xi

    for i in range(N):
        for j in range(N):
            r = 0
            for d in range(D):
                xj = X[j,d]
                xi = X[i,d]
                #r += (X[j, d] - X[i, d]) ** 2
                #r += (xj - xi) ** 2
                r += (xj - xi) * (xj - xi)
            r = r**0.5
            Y[i] += beta[j] * exp(-(r * theta)**2)

    return Y


# Cython  implementation of a Radial Basis Function (RBF) approximation scheme
# 
# TODO: Write the Cython implementation in a separate fastloop.pyx file, compile and import it here
# 
# from fastloop import rbf_network_cython
