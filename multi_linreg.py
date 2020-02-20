import numpy as np
import statsmodels.api as sm

# ===== pen and paper ======
X = np.array([[1, 5],
              [2, 3],
              [6, 4]])

Y = np.array([[50],
              [80],
              [40]])

k     =               X.transpose() @ X
k_inv = np.linalg.inv(X.transpose() @ X)
beta = k_inv @ X.transpose() @ Y
pp_result = X @ beta
# prints
# array([[65.75757576],
#        [39.03030303],
#        [51.03030303]])

# ===== verify with package ========
model = sm.OLS(Y, X).fit()
sm_result = model.predict(X)
# prints
# array([65.75757576, 39.03030303, 51.03030303])


