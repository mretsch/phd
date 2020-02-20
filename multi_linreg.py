import numpy as np

X = np.array([[1, 5],
              [2, 3],
              [6, 4]])

Y = np.array([[50],
              [80],
              [40]])

k = X.transpose() @ X
k_inv = np.linalg.inv(X.transpose() @ X)
beta = k_inv @ X.transpose() @ Y