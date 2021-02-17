import numpy as np

#Create a null vector of size 10 but the fifth value is 5
a = np.empty([10])
a[4] = 1

print(a)

#Create a vector with values ranging from 10 to 49
b = np.arange(10,50)

print(b)

#Reverse a vector
c = np.flip(b)

#print(b) # just to make sure that b is unaffected
print(c)

# Create a 3x3 matrix with values ranging from 0 8
d = np.arange(0,9).reshape((3,3))
print(d)

# Find indices of non-zeros elements from [1,2,0,0,4,0]
e = np.array([1,2,0,0,4,0])
f = e != 0 # boolean indices
print(f)
f = np.argwhere(e != 0) # numerical indices
print(f)

#Create a random vector of size 30 and find the mean value
g = np.random.random_sample([30])
print(g)
print(np.mean(g))

# Create a 2d array with 1 on the border and 0 inside
h = np.ones([4,4])
print(h)
h[1:-1,1:-1] = 0
print(h)

# Create a 8x8 matrix and fill it with a checkerboard pattern
i = np.zeros([8,8])
print(i)
i[::2,::2] =1; i[1::2,1::2] =1
print((i))

# Create a checkerboard 8x8 matrix using the tile function
j = np.tile(np.array([[1, 0], [0,1]]), (4,4))
print(j)

# Given a 1D array, negate all elements which are between 3 and 8 in place
k = np.arange(11)
k[np.logical_and(k>=3, k<=8)] *=-1
print(k)

# Create a random vector of size 10 and sort it
l = np.random.randint(0,10,[10])
print(l)
l = np.sort(l)
print(l)

# Consider two random arrays A and B, check if they are equal
m = np.random.randint(0,2,5)
n = np.random.randint(0,2,5)
print(m)
print(n)
equal = np.array_equal(m,n)
print(equal)

# How to convert an integer (32 bits) array into a float (32 bits) in place?
o = np.arange(10, dtype=np.int32)
print(o.dtype)
o = o.astype(dtype=np.float32, copy=False)
print(o.dtype)

# How to get the diagonal of a dot product
p = np.arange(9).reshape(3,3)
q = p + 1
r = np.dot(p,q)
print(r)
s = np.diagonal(r)
print(s)

