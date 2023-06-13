import numpy as np
from sklearn.utils.extmath import randomized_svd

np.random.seed(0)
X = np.random.random((10, 10)).astype(np.float16)*10
print("X", X.nbytes)
U, Sigma, VT = randomized_svd(X,
                              n_components=4,
                              n_iter=5,
                              random_state=0)
U = U.astype(np.float16)
VT = VT.astype(np.float16)
print(U.nbytes)
print(U.shape)
print(Sigma.nbytes)
print(VT.nbytes)
print(X)
print(np.matmul(U*Sigma, VT).astype(np.float16))