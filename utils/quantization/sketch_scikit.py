from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import SGDClassifier
X = [[1,2,3,4], [5,6,7,8], [9, 10, 11, 12]]
ps = PolynomialCountSketch(degree=6, random_state=1)
X_features = ps.fit_transform(X)
print(X_features.shape)