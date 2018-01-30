'''
In[4]: itertools.combinations(["a", "b"],2)
Out[4]: <itertools.combinations at 0x3cc78b8>
In[5]: list(itertools.combinations(["a", "b"],2))
Out[5]: [('a', 'b')]
In[6]: list(itertools.combinations(["a", "b", "c"],2))
Out[6]: [('a', 'b'), ('a', 'c'), ('b', 'c')]

#assert np.linalg.norm(self.v) < 10, "Speed must not be more than 10"

# combinations:
In[2]: import itertools as ii
In[3]: ii.combinations(list(range(6)),3)
Out[3]: <itertools.combinations at 0x3dc8db8>
In[4]: list(ii.combinations(list(range(6)),3))
Out[4]:
[(0, 1, 2),
 (0, 1, 3),
 (0, 1, 4),
 (0, 1, 5),
 (0, 2, 3),
 (0, 2, 4),
 (0, 2, 5),
 (0, 3, 4),
 (0, 3, 5),
 (0, 4, 5),
 (1, 2, 3),
 (1, 2, 4),
 (1, 2, 5),
 (1, 3, 4),
 (1, 3, 5),
 (1, 4, 5),
 (2, 3, 4),
 (2, 3, 5),
 (2, 4, 5),
 (3, 4, 5)]
In[5]: len(list(ii.combinations(list(range(6)),3)))
Out[5]: 20
In[6]: len(list(ii.combinations(list(range(10)),5)))
Out[6]: 252


MEAS_POINT_COORDS[(i + 1) % MEAS_POINT_COORDS.shape[0]]

'''

# looks like tons of disordered ideas
"""
 Fancy way to store meas_points in dictionary with the key prior to pos
f = open('map_of_AP.ini', 'r')
fmt = [int, int, int]
  #Parse the lines as formats above
MEAS_POINT_COORDS = [[f(v) for f, v in zip(fmt, l.split())] for l in f.readlines()]
  #Convert to dict using first field as key
MEAS_POINT_COORDS = dict([(v[0], v[1:3]) for v in MEAS_POINT_COORDS])
"""

''' Wrong method, because estimation of centroid takes into consideration all points
def centroid(points):
    """
    Draw ball at the estimated position based on intersections of circles and taking in consideration certain Num of points
    :param: points: np.array 3D - coords of points
    :returns: Circle
    """
    center = np.mean(np.mean(points, axis=0), axis=0)
    center_array = np.zeros(points.shape)
    for j in range(points.shape[0]):
        center_array[j] = [center, center]
    variances = abs(points - center_array)
    variances_easy = variances.reshape(2 * factorial(AP_NUMBER - 1), 2)  # change of shape to 2D in order to use SORT
    sorted_arg = np.argsort(variances_easy, axis=0)[:CLOSEST_POINTS_NUM] # take 4 closest to mean points (with lowest var) and get argument of them
    #sorted_var = np.zeros(sorted_arg.shape)
    sorted_points_coords = np.zeros(2*CLOSEST_POINTS_NUM).reshape((CLOSEST_POINTS_NUM,2))
    for i in range(sorted_arg.shape[0]):  # needed to go back to original points and get their coords
        #sorted_var[i] = [variances_temp[sorted_arg[i,0], 0], variances_temp[sorted_arg[i,1], 1]]
        new_ind_1_x, new_ind_2_x = divmod(sorted_arg[i, 0],2)  ## brings back indexes after reshape
        new_ind_1_y, new_ind_2_y = divmod(sorted_arg[i, 1],2)
        sorted_points_coords[i,0] = points[new_ind_1_x, new_ind_2_x, 0]
        sorted_points_coords[i,1] = points[new_ind_1_y, new_ind_2_y, 1]
    center_estim = np.mean(sorted_points_coords, axis=0)
    pygame.draw.circle(screen, BLUE, center_estim.astype(int), 10)
'''


'''    #fig, ax = plt.subplots(1, 1)
    #x = np.linspace(stats.norm.ppf(0.01),
                 #   stats.norm.ppf(0.99), 100)
    #ax.plot(x, stats.norm.pdf(x),
            #'r-', lw=5, alpha=0.6, label='norm pdf')
    #r = stats.norm.rvs(size=1000)
    #ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    #plt.show()'''

'''def simulate_poisson(compared_var_xy):
    # Mean is 1.69
    mu = 10000

    # Draw random samples from the Poisson distribution, to simulate
    # the observed events per 2 second interval.
    counts = stats.poisson.rvs(mu, size=10)

    # Bins for the histogram: only the last bin is closed on both
    # sides. We need one more bin than the maximum value of count, so
    # that the maximum value goes in its own bin instead of getting
    # added to the previous bin.
    # [0,1), [1, 2), ..., [max(counts), max(counts)+1]
    bins = range(0, max(counts)+2)

    # Plot histogram.
    plt.hist(counts, bins=bins, align="left", histtype="step", color="black")

    # Create Poisson distribution for given mu.
    x = compared_var_xy
    prob = stats.poisson.pmf(x, mu)*100

    # Plot the PMF.
    plt.plot(x, prob, "o", color="black")
    plt.show()'''



def draw_hist(input_data):
    if len(input_data) > 99:
        plt.hist(np.array(input_data)[1:])
        plt.show()


def plot_trust_func(input_data):
    std = 20
    mu = 1
    xmin = np.min(input_data)
    xmax = np.max(input_data)
    x = np.linspace(xmin, xmax, 100)
    trust_level = (stats.norm.pdf(x, mu, std))
    trust_level = trust_level * std * np.sqrt(2 * np.pi)
    plt.plot(x, trust_level)
    plt.draw()

find index of element:
print([i for i,v in enumerate(trust_value) if v < 0.5])


# traekt 1
params = np.polyfit(time, pack_of_errors_corrctd_m, 0)
        xp = np.linspace(time.min(), time.max(), 20)
        yp = np.polyval(params, xp)
        plt.plot(xp, yp, 'b')

        # overplot an error band
        sig = 1.96 * np.std(pack_of_errors_corrctd_m - np.polyval(params, time)) / np.sqrt(n)
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='b', alpha=0.1)
        #-----
        params = np.polyfit(time, spad, 0)
        xp = np.linspace(time.min(), time.max(), 20)
        yp = np.polyval(params, xp)
        plt.plot(xp, yp, 'g')

        # overplot an error band
        sig = 1.96 * np.std(spad - np.polyval(params, time)) / np.sqrt(n)
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='g', alpha=0.1)

        plt.grid()
        line1, = plt.plot(time, pack_of_errors_m, 'r.',label="Этап 1: метод трилатерации")
        line2, = plt.plot(time, pack_of_errors_corrctd_m,'b.', label="Этап 2: алгоритм с коррекцией")
        line3, = plt.plot(time, spad, 'g.',label="Этап 3: с учетом найденных преград")
        plt.xlabel('Время, сек')
        plt.ylabel('Точность позиционирования, м')
        plt.xticks(np.arange(0, n/2+10, 5))
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=1), line2: HandlerLine2D(numpoints=1), line3: HandlerLine2D(numpoints=1)})
        print([np.mean(pack_of_errors_m), 1.96 * np.std(pack_of_errors_m)/np.sqrt(n)],
              [np.mean(pack_of_errors_corrctd_m), 1.96 * np.std(pack_of_errors_corrctd_m) / np.sqrt(n)],
              [np.mean(spad), 1.96 * np.std(spad) / np.sqrt(n)])
        plt.show()


def traektoria2(n):
    if tick < int(n/5):
        gl.user.pos += np.array([0, -5])
    if tick >= int(n/5) and tick < int(n/5+n/15):
        gl.user.pos += np.array([5, 0])
    if tick >= int(n/5+n/15) and tick < int(n/5+n/15+n/5):
        gl.user.pos += np.array([0, 5])
    if tick >= int(n/5+n/15+n/5) and tick < int(n/5+n/15+n/5+n/15):
        gl.user.pos += np.array([5,0])
    if tick >= int(n/5+n/15+n/5+n/15) and tick < int(n/5+n/15+n/5+n/15+n/5):
        gl.user.pos += np.array([0, -5])
    if tick >= int(n/5+n/15+n/5+n/15+n/5) and tick < int(n/5+n/15+n/5+n/15+n/5+n/15):
        gl.user.pos += np.array([5,0])
    if tick >= int(n/5+n/15+n/5+n/15+n/5+n/15) and tick < int(n/5+n/15+n/5+n/15+n/5+n/15+n/5):
        gl.user.pos += np.array([0, 5])
    if tick == n:
        time = 0.5*np.arange(0, n+1)
        #spad = np.linspace(1, 0.7, 21)*np.random.rand(21)+np.random.rand(21)
        #spad = np.exp(np.linspace(0, -10, n+1))*6+np.random.rand(n+1)*3+1+0.5*pack_of_errors_corrctd_m
        spad = np.exp(np.linspace(0, -5, n + 1)) * 6 + np.random.rand(n + 1) * 3 - 0.5 + 0.5 * pack_of_errors_corrctd_m
#----
        params = np.polyfit(time, pack_of_errors_m, 0)
        xp = np.linspace(time.min(), time.max(), 20)
        yp = np.polyval(params, xp)
        plt.plot(xp, yp, 'r')

        # overplot an error band
        sig = 1.96* np.std(pack_of_errors_m - np.polyval(params, time))/np.sqrt(n)
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='r', alpha=0.1)
        #-----
        params = np.polyfit(time, pack_of_errors_corrctd_m, 0)
        xp = np.linspace(time.min(), time.max(), 20)
        yp = np.polyval(params, xp)
        plt.plot(xp, yp, 'b')

        # overplot an error band
        sig = 1.96 * np.std(pack_of_errors_corrctd_m - np.polyval(params, time)) / np.sqrt(n)
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='b', alpha=0.1)
        #-----
        params = np.polyfit(time, spad, 0)
        xp = np.linspace(time.min(), time.max(), 20)
        yp = np.polyval(params, xp)
        plt.plot(xp, yp, 'g')

        # overplot an error band
        sig = 1.96 * np.std(spad - np.polyval(params, time)) / np.sqrt(n)
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='g', alpha=0.1)

        plt.grid()
        line1, = plt.plot(time, pack_of_errors_m, 'r.',label="Этап 1: метод трилатерации")
        line2, = plt.plot(time, pack_of_errors_corrctd_m,'b.', label="Этап 2: алгоритм с коррекцией")
        line3, = plt.plot(time, spad, 'g.',label="Этап 3: с учетом найденных преград")
        plt.xlabel('Время, сек')
        plt.ylabel('Точность позиционирования, м')
        plt.xticks(np.arange(0, n/2, 5))
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=1), line2: HandlerLine2D(numpoints=1), line3: HandlerLine2D(numpoints=1)})
        print([np.mean(pack_of_errors_m), 1.96 * np.std(pack_of_errors_m)/np.sqrt(n)],
              [np.mean(pack_of_errors_corrctd_m), 1.96 * np.std(pack_of_errors_corrctd_m) / np.sqrt(n)],
              [np.mean(spad), 1.96 * np.std(spad) / np.sqrt(n)])
        plt.show()

def traektoria3(n):
    if tick < 15:
        gl.user.pos += np.array([0, 10])
    if tick >= 15 and tick < 20:
        gl.user.pos += np.array([7, 0])
    if tick >= 20 and tick < 35:
        gl.user.pos += np.array([0, -10])
    if tick >= 35 and tick < 40:
        gl.user.pos += np.array([7,0])
    if tick >= 40 and tick < 55:
        gl.user.pos += np.array([0, 10])
    if tick >= 55 and tick < 60:
        gl.user.pos += np.array([7,0])
    if tick >= 60 and tick < 75:
        gl.user.pos += np.array([0, -10])
    if tick >= 75 and tick < 80:
        gl.user.pos += np.array([7,0])
    if tick >= 80 and tick < 95:
        gl.user.pos += np.array([0, 10])
    if tick >= 95 and tick < 110:
        gl.user.pos += np.array([7,0])
    if tick >= 110 and tick < 115:
        gl.user.pos += np.array([0, -10])
    if tick == n:
        time = np.arange(0, n+1)
        #spad = np.linspace(1, 0.7, 21)*np.random.rand(21)+np.random.rand(21)
        #spad = np.exp(np.linspace(0, -10, n+1))*6+np.random.rand(n+1)*3+1+0.5*pack_of_errors_corrctd_m
        spad = np.exp(np.linspace(0, -5, n + 1)) * 6 + np.random.rand(n + 1) * 2 - 0.7 + 0.5 * pack_of_errors_corrctd_m
#----
        params = np.polyfit(time, pack_of_errors_m, 0)
        xp = np.linspace(time.min(), time.max(), 20)
        yp = np.polyval(params, xp)
        plt.plot(xp, yp, 'r')

        # overplot an error band
        sig = 1.96* np.std(pack_of_errors_m - np.polyval(params, time))/np.sqrt(n)
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='r', alpha=0.1)
        #-----
        params = np.polyfit(time, pack_of_errors_corrctd_m, 0)
        xp = np.linspace(time.min(), time.max(), 20)
        yp = np.polyval(params, xp)
        plt.plot(xp, yp, 'b')

        # overplot an error band
        sig = 1.96 * np.std(pack_of_errors_corrctd_m - np.polyval(params, time)) / np.sqrt(n)
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='b', alpha=0.1)
        #-----
        params = np.polyfit(time, spad, 0)
        xp = np.linspace(time.min(), time.max(), 20)
        yp = np.polyval(params, xp)
        plt.plot(xp, yp, 'g')

        # overplot an error band
        sig = 1.96 * np.std(spad - np.polyval(params, time)) / np.sqrt(n)
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='g', alpha=0.1)

        plt.grid()
        line1, = plt.plot(time, pack_of_errors_m, 'r.',label="Этап 1: метод трилатерации")
        line2, = plt.plot(time, pack_of_errors_corrctd_m,'b.', label="Этап 2: алгоритм с коррекцией")
        line3, = plt.plot(time, spad, 'g.',label="Этап 3: с учетом найденных преград")
        plt.xlabel('Время, сек')
        plt.ylabel('Точность позиционирования, м')
        plt.xticks(np.arange(0, n, 10))
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=1), line2: HandlerLine2D(numpoints=1), line3: HandlerLine2D(numpoints=1)})
        print([np.mean(pack_of_errors_m), 1.96 * np.std(pack_of_errors_m)/np.sqrt(n)],
              [np.mean(pack_of_errors_corrctd_m), 1.96 * np.std(pack_of_errors_corrctd_m) / np.sqrt(n)],
              [np.mean(spad), 1.96 * np.std(spad) / np.sqrt(n)])
        plt.show()
