import numpy as np
from matplotlib import pyplot as plt
from gl_vars import *


pos_error_array = np.array([
        [1, 2.35, 0.08],
        [2, 2.36, 0.086],
        [3, 2.38, 0.086],
        [4, 2.42, 0.09],
        [5, 2.45, 0.091],
        [10, 2.76, 0.14]])

plt.grid()
plt.errorbar(pos_error_array[:, 0], pos_error_array[:, 1], yerr=pos_error_array[:, 2], fmt='-o')
plt.ylabel('точность позиционирования, м')
plt.xlabel('суммарная площадь преград, м2')
plt.show()
