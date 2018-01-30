#!/usr/bin/python3
from matplotlib import pyplot as plt
import numpy as np
f= 2400
d = range(1,1001)
noise = np.random.normal(0,7)
#noise = np.random.randint(0,10,100)
PL1 = 20-(20 * np.log10(d) + 20 * np.log10(f) - 147.55+noise)
#plt.semilogx(d,PL1,'ro')
#plt.show()


# example data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

# example error bar values that vary with x-position
error = 0.1 + 0.2 * x

# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
#ax0.errorbar(x, y, yerr=error, fmt='-o')
plt.errorbar(x, y, yerr=error, fmt='-o')
#ax0.set_title('variable, symmetric error')
plt.title('variable, symmetric error')
plt.show()