import numpy as np

from scipy import linalg

from scipy import sparse

def bytes(sparse):

    if "numpy" in str(type(sparse)):
        return sparse.nbytes

    return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

m = np.float16
rng = np.random.default_rng()
np.random.seed(0)

n_rows, n_columns, density, sketch_n_rows = 10, 10, 0.01, 7

# ori = np.random.randint(0, 3, (n_rows, n_columns)).round(1).astype(m)
ori = np.random.random((n_rows, n_columns)).round(1).astype(m)
print(ori)
print("Tamanho matriz numpy original: ", ori.nbytes)

A = sparse.csr_matrix(ori)
print("Esparsa")
# print(A.toarray())

# A = sparse.rand(n_rows, n_columns, density=density, format='csc')
#
# B = sparse.rand(n_rows, n_columns, density=density, format='csr')
#
# C = sparse.rand(n_rows, n_columns, density=density, format='coo')
#
# D = rng.standard_normal((n_rows, n_columns))

print("Tamanho matriz esparsa original: ", bytes(A))
SA = linalg.clarkson_woodruff_transform(ori, sketch_n_rows)

# print("Tamanho matriz esparsa reduzida: ", sparse_nbyes(SA))
print("zi: ", SA.round(1))
print("Tamanho matriz numpy reduzida: ", bytes(SA))
SA = SA.astype(m) # fastest
SA_R = linalg.clarkson_woodruff_transform(SA, n_rows)
print("Matriz recuperada: ", SA_R.astype(m).round(1))

# SB = linalg.clarkson_woodruff_transform(B, sketch_n_rows).toarray() # fast
#
# SC = linalg.clarkson_woodruff_transform(C, sketch_n_rows).toarray() # slower
#
# SD = linalg.clarkson_woodruff_transform(D, sketch_n_rows) # slowest

print("Sketch: ", SA.shape, linalg.clarkson_woodruff_transform(SA, n_rows).shape)
print("Norma l2 da matriz original: ", np.sqrt(np.sum(np.square(SA))),  " bytes: ", SA.nbytes, " \nNormal l2 da matriz reconstruida: ", np.sqrt(np.sum(np.square(SA_R))), " bytes: ", SA_R.nbytes)
print(type(SA))
# print("Sketch: ", SB.shape)
# print("Sketch: ", SC.shape)
# print("Sketch: ", SD.shape)