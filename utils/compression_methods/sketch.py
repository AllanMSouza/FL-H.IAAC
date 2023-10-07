import numpy as np

np.random.seed(0)

def countSketchInMemroy(matrixA, s):
    m, n = matrixA.shape
    matrixC = np.zeros([m, s])
    hashedIndices = np.random.choice(s, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
    matrixA = matrixA * randSigns.reshape(1, n) # flip the signs of 50% columns of A
    for i in range(s):
        idx = (hashedIndices == i)
        matrixC[:, i] = np.sum(matrixA[:, idx], 1)
    return matrixC

matrixA = np.random.random((10, 10))
s = 5 # sketch size, can be tuned
matrixC = countSketchInMemroy(matrixA, 5)
rowNormsA = np.sqrt(np.sum(np.square(matrixA), 1))
print(rowNormsA)
rowNormsC = np.sqrt(np.sum(np.square(matrixC), 1))
print(rowNormsC)
print(matrixA)
print(matrixC)