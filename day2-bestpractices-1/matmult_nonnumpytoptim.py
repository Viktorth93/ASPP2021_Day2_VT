# Program to multiply two matrices using nested loops
import random

#N = 250

# NxN matrix
@profile
def matmult(N=250):
    X = []
    for i in range(N):
        X.append([random.randint(0,100) for r in range(N)])
# Nx(N+1) matrix
    Y = []
    for i in range(N):
        Y.append([random.randint(0,100) for r in range(N+1)])

# result is Nx(N+1)
    result = []
    for i in range(N):
        result.append([0] * (N+1))

# iterate through rows of X
    #for i in range(len(X)):
    # iterate through columns of Y
        #for j in range(len(Y[0])):
        # iterate through rows of Y
            #for k in range(len(Y)):
                #result[i][j] += X[i][k] * Y[k][j]
    result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in Y] for X_row in X]
    #result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in zip(*X)]


    for r in result:
        print(r)

matmult(250)
