# Program to multiply two matrices using nested loops
import random
import numpy as np

#N = 250

# NxN matrix
@profile
def matmult(N=250):
    #X = []
    #X = np.empty([N,N])
    #for i in range(N):
    #    X.append([random.randint(0,100) for r in range(N)])
    X = np.random.randint(0,100, size=(N,N))
# Nx(N+1) matrix
    #Y = []
    #for i in range(N):
    #    Y.append([random.randint(0,100) for r in range(N+1)])
    Y = np.random.randint(0,100, size=(N,N+1))

    np_X = np.array(X)
    np_Y = np.array(Y)
    result = np.matmul(X,Y)

# result is Nx(N+1)
    #result = []
    #for i in range(N):
        #result.append([0] * (N+1))

# iterate through rows of X
    #for i in range(len(X)):
    # iterate through columns of Y
        #for j in range(len(Y[0])):
        # iterate through rows of Y
            #for k in range(len(Y)):
                #result[i][j] += X[i][k] * Y[k][j]

    for r in result:
        print(r)

matmult(250)
