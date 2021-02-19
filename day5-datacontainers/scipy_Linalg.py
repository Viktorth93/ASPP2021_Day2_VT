from scipy import linalg as la
import numpy as np

# a) Define matrix
A = np.array([[1,2,3], [4,5,6],[7,8,9]])
#A = np.array([[4,4,4], [1,2,0],[1,2,5]])
print(A)

# b) Define vector
b = np.array([1.,2.,3.])
print(b)

#c) Solve the linear system of equations Ax = b
res = la.solve(A,b)
print(res)

#d) Check result
print(np.matmul(A,res))
print((np.matmul(A,res))==b)

#e) Now using a random 3x3 matrix instead
B = np.random.random_sample([3,3])
print(B)

resMat = la.solve(A,B)
resMatLstSq, r, rnk, s = la.lstsq(A,B)
print(resMat)
print(resMatLstSq)


print("Attempt at Exact solution")
print(np.matmul(A,resMat))
print("Least squares solution")
print(np.matmul(A,resMatLstSq))

#f) Solve the eigenvalue problem for A
eigenvals, eigenvecs = la.eig(A)

print("Eigenvalues: ", eigenvals)
print("Eigenvectors: ", eigenvecs)
#print(np.matmul(A,eigenvecs[:,1]))
#print(np.dot(eigenvals[1],eigenvecs[:,1]))

print("Difference between A*eigvec and eigval*eigvec: ", la.norm(np.matmul(A,eigenvecs[:,1])-np.dot(eigenvals[1],eigenvecs[:,1])))

#g) Calculate the inverse, determinant of A
invA = la.inv(A)
print("Inverse of A: ", invA)

detA = la.det(A)
print("Determinant of A: ", detA) 

#Calculate norm of A with different orders
norm2A = la.norm(A, ord=2)
print("Matrix 2-norm of A: ", norm2A)

normMaxRowsA = la.norm(A, ord=1)
print("Max of sum over rows of A: ", normMaxRowsA)

normMaxColsA = la.norm(A, ord=np.inf)
print("Max of sum over cols of A: ", normMaxColsA)
