import numpy as np
import matplotlib.pyplot as plt

import sub as FORTRAN

# Create dummy data
I = np.array([[0., 1.], [2., 0.2]])

# Call the fortran routine.
T = FORTRAN.first_test(image=I, threshold=0.3)