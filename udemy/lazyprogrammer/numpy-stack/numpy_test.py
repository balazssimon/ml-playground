import numpy as np

L = [1,2,3]
A = np.array(L)

for e in L:
    print(e)

for e in A:
    print(e)

L.append(4)
L = L + [5]


L2 = []
for e in L:
    L2.append(e+e)

A+A
2*A

2*L

L3 = []
for e in L:
    L3.append(e*e)

A**2
np.sqrt(A)
np.log(A)
np.exp(A)


a = np.array([1,2])
b = np.array([2,1])

dot = 0
for e,f in zip(a,b):
    dot += e*f

# elementwise multiplication
a*b

# dot product (inner product)
np.sum(a*b)
(a*b).sum()

np.dot(a,b)
a.dot(b)
b.dot(a)
np.inner(a,b)

# magnitude of a vector
amag = np.sqrt((a*a).sum())
amag = np.linalg.norm(a)

# angle between two vectors
cosangle = a.dot(b) / (np.linalg.norm(a)*np.linalg.norm(b))
angle = np.arccos(cosangle)


# matrix
L = [[1,2], [3,4]]
M = np.array(L)

L[0]
L[0][0]

M[0][0]
M[0,0]

M2 = np.matrix(L) # recommended: use np.array instead!!!

A = np.array(M2)
A

# transpose
A.T

# generating arrays and matrices
Z = np.zeros(10)
ZM = np.zeros((10,10))

OM = np.ones((10,10))

RM = np.random.random((10,10)) # uniformly distributed numbers between 0..1
GM = np.random.randn(10,10) # Gaussian distribution (note: no tuple!)

GM.mean()
GM.var()


# elementwise matrix multiplication:
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
A*B

# matrix multiplication:
A = np.array([[1,2],[3,4]])
B = np.array([[5,6,7],[7,8,9]])
A.dot(B)

# inverse:
Ainv = np.linalg.inv(A)
Ainv.dot(A)
A.dot(Ainv)

# determinant:
np.linalg.det(A)

# diagonal elements of a matrix:
np.diag(A)

# constructing a diagonal matrix:
np.diag([1,2])

# outer product:
a = np.array([1,2])
b = np.array([3,4])
np.outer(a,b)

# trace (sum of the diagnoal elements):
np.diag(A).sum()
np.trace(A)

# covariance:
X = np.random.randn(100,3)
cov = np.cov(X)
cov.shape
covT = np.cov(X.T)
covT.shape

# eigenvalues, eigenvectors:
np.linalg.eigh(covT) # eigh is for symmetric (A=A^T) and Hermitian (A=A^H) matrices
np.linalg.eig(covT) # eig is for general matrices

# solving a linear system of equations:
A = np.array([[1,2], [3,4]])
b = np.array([1,2])
x = np.linalg.inv(A).dot(b)

np.linalg.solve(A, b)
