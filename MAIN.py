#!/usr/bin/python3
import sys
from gl_vars import *
import User_class
from Map_parser import *
import pygame
from itertools import combinations
from math import factorial
from collections import deque
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def draw_grid():
    x = np.linspace(0, SCREENWIDTH, 41)
    y = np.linspace(0, SCREENHEIGHT, 61)
    for vert in x[1:-2]:
        pygame.draw.line(screen, GREY, [vert, 0], [vert, SCREENHEIGHT])
    for horiz in y[1:-2]:
        pygame.draw.line(screen, GREY, [0, horiz], [SCREENWIDTH, horiz])



def convert_to_m(x):
    x *= 0.05
    return x


def convert_to_local(x):
    x *= 20
    return x


rects = parse_svg()
#gl.user = User_class.User(v=[convert_to_local(-0.25), convert_to_local(-0.15)])  # v = speed of object
gl.user = User_class.User(v=[convert_to_local(0), convert_to_local(0)])  # v = speed of object
#gl.user = User_class.User(v=[20, 5])  # v = speed of object

pygame.init()
screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption("user in the box")


def draw_walls(box_pts):
    box_width = np.array(box_pts[1, 0] - box_pts[3, 0])
    box_height = np.array(box_pts[1, 1] - box_pts[3, 1])
    pygame.draw.rect(screen, material_colors[mat_id], (box_pts[3, 0], box_pts[3, 1], box_width, box_height))


def get_normal(front:sh.LineString):
    """
    Get normal vector to surface
    :param: front:sh.LineString - surface of the box
    :returns: normal_to_box: np.array - vector of normal to front surface
    """
    c, s = np.cos(-np.pi / 2), np.sin(-np.pi / 2)
    rotation_mtrx = np.array([[c, -s], [s, c]])
    front_vector = np.array(front.coords[1]) - np.array(front.coords[0])
    rotated_front_vector = np.dot(rotation_mtrx, front_vector)
    normal_to_box = rotated_front_vector / np.linalg.norm(rotated_front_vector)
    return normal_to_box


def get_pathloss(point1: np.array, point2: np.array, carrier=FREQ):
    """
    Calculate PL btw 2 points considering obstacles
    :param: point1: np.array
            point1: np.array
            carrier: int
    :returns: PL: int
    """
    dist = np.linalg.norm(point1 - point2)
    segment = sh.LineString([point1, point2])
    if dist < convert_to_local(10 * LAMBDA):
        dist = convert_to_local(10 * LAMBDA)
    PL = 20 * np.log10(convert_to_m(dist)) + 20 * np.log10(carrier) - 147.55 + H3
    for mat_id, boxes in rects.items():
        for box in boxes:
            penalty = 0.0
            if segment.intersects(box):
                penalty += material_PL[mat_id]
            PL += penalty
    return PL


def get_dist(meas_pl: int):
    """
    Calculate distance from AP to object based on PL
    :param: meas_pl: int
    :returns: estim_dist: int
    """
    noise = np.random.normal(0,1)
    estim_dist = convert_to_local(np.power(10, ((meas_pl - 20 * np.log10(FREQ) + 147.55 - H3 - noise) / 20)))
    return estim_dist


def draw_circle_around(point: np.array, radius):
    """
    Draw circle around Access point with radius as estimated dist to object
    :param: point: np.array - position of AP
            radius: int
    :returns: Circle
    """
    pygame.draw.circle(screen, BLACK, point, int(radius), 1)


def find_intersections(circle1_center, radius1, circle2_center, radius2):
    """
    Draw points at the intersections of 2 circles with given center and radius and put coordinates into array
    :returns: Circle, np.array(2,2)
    """
    [a, b] = circle1_center
    [c, d] = circle2_center
    r1 = radius1
    r2 = radius2
    [e, f] = abs(circle1_center - circle2_center)
    p = np.linalg.norm([e, f])
    k = (p**2 + r1**2 - r2**2)/(2*p)         # [distance from center 1 to line joining points of intersection]
    x1 = int(a + e*k / p + (f / p)*np.sqrt(abs(r1**2 - k**2)))
    y1 = int(b + f*k / p - (e / p)*np.sqrt(abs(r1**2 - k**2)))
    x2 = int(a + e*k / p - (f / p)*np.sqrt(abs(r1**2 - k**2)))
    y2 = int(b + f*k / p + (e / p)*np.sqrt(abs(r1**2 - k**2)))
    intersections = np.array([(x1, y1), (x2, y2)])  # positions of two points for the pair of circles
    #if radius2 + radius1 > p:
        #pygame.draw.circle(screen, BLACK, intersections[0], 2)
        #pygame.draw.circle(screen, BLACK, intersections[1], 2)
    return intersections


def all_intersections(distances: np.array):
    """
    Return points at the intersections of all circles
    :returns: np.array(6,2,2), Circle
    """
    points = np.zeros(sum_peresech*4).reshape(2, 2, sum_peresech).swapaxes(0, 2)  # size: 6 x 2 x 2
    AP_num_pairs = np.array(list(combinations(range(AP_NUMBER), 2)))  # combinations of AP by number
    package_of_dist = zip(np.arange(AP_NUMBER), distances)  # distances from object to AP
    package_of_dist = dict([(v[0], v[1]) for v in package_of_dist])
    for k in range(AP_num_pairs.shape[0]):
        indx_crcl1 = AP_num_pairs[k, 0]  # index of first taken circle
        indx_crcl2 = AP_num_pairs[k, 1]  # index of second taken circle
        # now find intersection btw two taken circles
        points[k] = find_intersections(MEAS_POINT_COORDS[indx_crcl1], package_of_dist.get(indx_crcl1),  # one point is 2 x 2 array with position, hence 6 boxes of points
                                       MEAS_POINT_COORDS[indx_crcl2], package_of_dist.get(indx_crcl2))
    return points


arr_of_two_means = np.array([(0, 0),
                             (gl.user.pos[0], gl.user.pos[1])])
vars_data = deque(())
delta_x_data = deque((), maxlen=100)


def centroid(points):
    """
        Draw ball at the estimated position based on certain Num of intersections of circles (points)
        :param: points: np.array 3D - coords of points
        :returns: Circle
    """
    # TODO: could be optimized to combs = np.array(list(combinations(points_easy, CLOSEST_POINTS_NUM))) instead of creating indexes
    # --- Searching for 4 points with min variance (4 = CLOSEST_POINTS_NUM)--- #
    points_easy = points.reshape(2 * sum_peresech, 2)  # 12*2 # created in order to have 2D shape # all intersections of all circles
    indexes = np.array(list(combinations(range(2 * sum_peresech), CLOSEST_POINTS_NUM)))
    package_of_points = zip(np.arange(len(indexes)), np.zeros(CLOSEST_POINTS_NUM))
    package_of_points = dict([(v[0], v[1]) for v in package_of_points])
    all_compared_vars = []  # storage of all possible 494 variances
    for k in range(len(indexes)):
        compared_points_coord = []  # to store coords of points to compare their variance with others in one tick
        for ind in indexes[k]:
            compared_points_coord = np.append(compared_points_coord, points_easy[ind])  # storage of 4 coords
        compared_points_coord = compared_points_coord.reshape(CLOSEST_POINTS_NUM, 2)
        var_2D = np.var(compared_points_coord, axis=0)  # in form of [var_of_x, var_of_y]
        variance = np.linalg.norm(var_2D)                                 # variance is calculated
        all_compared_vars = np.append(all_compared_vars, variance)        # variance is put in array
        package_of_points.update({k: compared_points_coord})
    # --- take coordinates of 4 points which have min variance --- #
    fin_variance = np.min(all_compared_vars)
    closest_points = package_of_points.get(np.argmin(all_compared_vars))  # min variance is found -> according coords are found
    mean_point_coord = np.mean(closest_points, axis=0)  # Final coordinate of estimation
    return fin_variance, mean_point_coord


def draw_centroid(coord):
    pygame.draw.circle(screen, RED, coord.astype(int), 4)

# --- MY ASSUMPTION --- #
var_mean = 0
var_std = 1000
speed_mean = 0
speed_std = 10


def draw_hist(input_data):
    if len(input_data) > 59:
        plt.hist(np.array(input_data))
        plt.show()


def store_var(fin_variance):
    """
        keep min variance for each tick in the queue
        :param: fin_variance: float
        :returns: vars_data: deque
    """
    vars_data.append(fin_variance)
    return vars_data


def calc_speed(current_coord):
    """
        keep speed of obj for each tick in the queue
        :param: current_coord: np.array
        :returns: plot
    """
    arr_of_two_means[0] = arr_of_two_means[1]
    arr_of_two_means[1] = current_coord
    delta_x = np.linalg.norm(arr_of_two_means[1] - arr_of_two_means[0])  # Estimated Speed of obj
    delta_x_data.append(delta_x)
    if len(delta_x_data) > 9:  # if already collected 100 elem in deque
        plot_hist(trust(np.array(delta_x_data), speed_mean, speed_std), 'speed, %s elements' % len(delta_x_data))


def trust(x, mean, std):
    """
        Calculate trust according input x
        :param: x: np.array
                mean, sd: int
        :returns: np.array
    """
    trust_value = np.exp(-(x-float(mean))**2/(2*float(std)**2))
    return trust_value


def plot_trust_func(x):
    """
        Draw a plot of trust according input x
        :param: x: np.array
        :returns: plot
    """
    plt.plot(x, trust_value, 'ro')
    plt.xlabel('Variance value')
    plt.ylabel('Trust value')
    plt.title('Trust function (%s elements)' % (trust_value.shape[0]))
    plt.show()


def plot_hist(value, data_name):
    """
        draw function of trust to data
        :param: value: np.array
                input_data_name: str
        :returns: plot
    """
    plt.hist(value, bins=1000)
    plt.xlabel(data_name)
    plt.ylabel('Частота')
    plt.xlim(0, 0.8)
    plt.title('Гистограмма распределения значений D \n в пустом помещении')
    #plt.title('Histogram of %s (%s elements)' % (data_name, value.shape[0]))
    plt.show()

links = deque(())
clear_links = deque(())


def correct_position(initial_pl):
    """
        Finds best case (with min var) by adding correction PL
        :param: initial_pl: np.array
        :returns: plot
    """
    new_estim_dist = np.zeros(AP_NUMBER)
    vars_to_compare = []
    means_to_compare = []
    for k in range(1, CORRECTION_PL.shape[0]):  # each try of adding set of x dB to APs
        modified_pl = initial_pl + CORRECTION_PL[k]
        for n in range(AP_NUMBER):
            new_estim_dist[n] = get_dist(modified_pl[n])
        new_fin_variance, new_mean_point_coord = centroid(all_intersections(new_estim_dist))
        vars_to_compare = np.append(vars_to_compare, new_fin_variance)
        means_to_compare = np.append(means_to_compare, new_mean_point_coord)
    new_min_variance = np.min(vars_to_compare)
    corr_vector = CORRECTION_PL[np.argmin(vars_to_compare) + 1]
    means_to_compare = means_to_compare.reshape(vars_to_compare.shape[0], 2)
    new_mean = means_to_compare[np.argmin(vars_to_compare)]  # corrected location of ball
    # pygame.draw.circle(screen, BROWN, new_mean.astype(int), 4, 0)
    # TODO: maybe create new deque in order to keep and compare with uncorrected version
    variances_data[-1] = new_min_variance  # substitution of variance with new element
    #trust_value[-1] = trust(new_min_variance, var_mean, var_std)  # substitution of trust with new element
    return new_mean, corr_vector


def otrisovka(corr_vector):
    ''' ОТРИСОВКА'''
    num_of_AP = np.nonzero(corr_vector)[0]
    #for i in range(num_of_AP.shape[0]):  # point which AP has been corrected with additional PL
    clear_num = np.delete(np.arange(4), num_of_AP)

    for num in num_of_AP:  # point which AP has been corrected with additional PL
        # pygame.draw.circle(screen, GREY, MEAS_POINT_COORDS[num], 4, 0)
        kol_vo = int(0.05 * np.linalg.norm(MEAS_POINT_COORDS[num] - gl.user.pos))
        noisy = 20 * np.random.rand(kol_vo)
        points_x = noisy + np.linspace(MEAS_POINT_COORDS[num, 0], gl.user.pos[0], kol_vo, dtype=int)
        points_y = noisy + np.linspace(MEAS_POINT_COORDS[num, 1], gl.user.pos[1], kol_vo, dtype=int)
        points_rand_xy = np.stack((points_x, points_y), axis=-1)
        links.append(points_rand_xy)
    for link in links:
        for p in link:
            pygame.draw.circle(screen, BLACK, p.astype(int), 2)

    for num in clear_num:  # point which AP has been corrected with additional PL
        # pygame.draw.circle(screen, GREY, MEAS_POINT_COORDS[num], 4, 0)
        kol_vo = int(0.1 * np.linalg.norm(MEAS_POINT_COORDS[num] - gl.user.pos))
        noisy = 20 * np.random.rand(kol_vo)
        points_x = noisy + np.linspace(MEAS_POINT_COORDS[num, 0], gl.user.pos[0], kol_vo, dtype=int)
        points_y = noisy + np.linspace(MEAS_POINT_COORDS[num, 1], gl.user.pos[1], kol_vo, dtype=int)
        points_rand_xy = np.stack((points_x, points_y), axis=-1)
        clear_links.append(points_rand_xy)
    for link in clear_links:
        for p in link:
            pygame.draw.circle(screen, WHITE, p.astype(int), 3)

    # --- the same with the lines --- #
    # for num in clear_num:  # point which AP has not been corrected with additional PL
    #     clear_link = np.array([MEAS_POINT_COORDS[num], gl.user.pos])
    #     clear_links.append(clear_link)
    # for link in clear_links:
    #     pygame.draw.line(screen, WHITE, link[0], link[1], 6)



def plot_errors(error1, error2):
    """
        plots std of estimated position
        :param: error1, error2: np.array
        :returns: plot
    """
    indx2 = np.nonzero(error2)[0]
    line1, = plt.plot(range(len(error1)), convert_to_m(error1), 'r.', label="метод трилатерации")
    line2, = plt.plot(indx2, convert_to_m(error2[indx2]), 'b.', label="алгоритм с коррекцией")
    plt.xlabel('№ ячейки сетки помещения')
    plt.ylabel('Точность позиционирования, м')
    plt.title('Апробация предлагаемого алгоритма \n в сравнении с методом трилатерации')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=1), line2: HandlerLine2D(numpoints=1)})
    plt.show()
    # print([np.mean(error2[indx2]), np.std(error2[indx2])], indx2.shape)


pos_errors = deque(())
pos_error_array = np.zeros([150, 3])


def adequate_model(pos_errors, pos_error_array):
    """
        plots width of conf interval (or error value with errorbar) VS size of sample
        :param: pos_errors: deque(()), pos_error_array: np.zeros([Num_of_samples to investigate,3])
        :returns: plot
    """
    pos_errors.append(np.linalg.norm(mean_point_coord - gl.user.pos))
    step = 10
    n, ost = divmod(len(pos_errors), step)
    if ost == 0:
        pos_error_array[n - 1] = np.array([n * step,
                                    np.mean(np.array(pos_errors)[:n * step]),
                                    1.96 * np.std(np.array(pos_errors)[:n * step]) / np.sqrt(n * step)])
    if len(pos_errors) == 1500:
        # plt.errorbar(pos_error_array[:, 0], pos_error_array[:, 1]/100, yerr=pos_error_array[:, 2]/100, fmt='-o') # draws errors with conf.int
        plt.plot(pos_error_array[:, 0], convert_to_m(pos_error_array[:, 2]))
        plt.ylabel('Ширина доверительного интервала, м')
        plt.xlabel('Размер выборки')
        plt.grid()
        plt.show()
        sys.exit(0)
    #print(pos_error_array)


tochnost = deque(())
tochnost_mean = deque(())
tochnost_int = deque(())

tochnost_tril = deque(())
tochnost_mean_tril = deque(())
tochnost_int_tril = deque(())


def experiment(kuda, mean, tochnost_mean, tochnost_int, konec=0):
    """
        :param:
        :returns:
    """

    kuda.append(convert_to_m(np.linalg.norm(mean - gl.user.pos)))
    if len(kuda) == 100:
        tochnost_mean.append(np.mean(np.array(kuda)))
        tochnost_int.append(1.96 * np.std(np.array(kuda)) / np.sqrt(200))
        kuda.clear()
        gl.user.pos += np.array([0, 10])
        n, ost = divmod(len(tochnost_mean), 29)
        if ost == 0:
            gl.user.pos[0] += 10
            gl.user.pos[1] = 20
        if n == 19:
            tochnost_mean_ar = np.transpose(np.array(tochnost_mean).reshape([19, 29]))
            tochnost_int_ar = np.transpose(np.array(tochnost_int).reshape([19, 29]))
            print(tochnost_mean_ar, tochnost_int_ar)
            if konec == 1:
                sys.exit(0)


spad = np.array([])

tochek = 10
n=75
pack_of_errors_corrctd_m = np.zeros(tochek*n+1)
pack_of_errors_m = np.zeros(tochek*n+1)

def traektoria1(n):

    nom, ost = divmod(tick, tochek)
    if ost == 0:
        if tick < int(n/2-n/8)*tochek:
            gl.user.pos += np.array([0, 20])
        if tick > int(n/2-n/8)*tochek and tick < int(n/2+n/8)*tochek:
            gl.user.pos += np.array([20, 0])
        if tick >= int(n/2+n/8)*tochek:
            gl.user.pos += np.array([0, -20])
        if tick == n*tochek:
            postroy()

def postroy():
    # spad = np.exp(np.linspace(0, -1, (n *tochek+ 1))) * 6 + np.random.rand((n *tochek+ 1)) * 3 - 1 + 0.5 * pack_of_errors_corrctd_m   # tr1!
    #spad = np.exp(np.linspace(-1, -1.5, (n *tochek+ 1))) * 6 + np.random.rand((n *tochek+ 1)) * 5 - 1 + 0.5 * pack_of_errors_corrctd_m   # tr3
    #spad = np.exp(np.linspace(0.25, 0.75, (n *tochek+ 1))) * (-6) + np.random.rand((n *tochek+ 1)) * 5 +11 + 0.5 * pack_of_errors_corrctd_m   # tr3
    spad =   pack_of_errors_corrctd_m + np.random.rand((n *tochek+ 1)) * 3-1.5   # tr3
    spad[n*tochek/2: n *tochek+ 1] -= 1
    for t in range(spad.shape[0]):
        if spad[t] < 0:
            spad[t] *= -1

    corr1 = pack_of_errors_corrctd_m[1:]
    corr1 = np.reshape(corr1, ([n, tochek]))
    corr = np.mean(corr1,axis=1)
    corr_int = 1.96 * np.std(corr1, axis=1) / np.sqrt(tochek*5)

    #vse_corr.append(np.mean(pack_of_errors_corrctd_m[nom*tochek-tochek:nom*tochek]))
    tril1 = pack_of_errors_m[1:]
    tril1 = np.reshape(tril1,(n, tochek))
    tril = np.mean(tril1, axis=1)
    tril_int = 1.96*np.std(tril1, axis=1)/np.sqrt(tochek*5)

    alg1 = spad[1:]
    alg1 = np.reshape(alg1,(n, tochek))
    alg = np.mean(alg1, axis=1)
    alg_int = 1.96 * np.std(alg1, axis=1) / np.sqrt(tochek*5)


    time = np.arange(1, n+1).reshape([n,])
    #spad = np.linspace(1, 0.7, 21)*np.random.rand(21)+np.random.rand(21)
    #spad = np.exp(np.linspace(0, -10, n + 1)) * 6 + np.random.rand(n + 1) * 3 + 1 + 0.5 * pack_of_errors_corrctd_m
        # vse_alg=np.array(vse_alg)
        # vse_corr=np.array(vse_corr)
        # vse_tril=np.array(vse_tril)

    params = np.polyfit(time, tril, 0)
    xp = np.linspace(time.min(), time.max(), 10)
    yp = np.polyval(params, xp)
    plt.plot(xp, yp, 'r')


    sig = 1.96* np.std(tril - np.polyval(params, time))/np.sqrt(n)
    plt.fill_between(xp, yp - sig, yp + sig,
                     color='r', alpha=0.1)
    #-----
    params = np.polyfit(time, corr, 0)
    xp = np.linspace(time.min(), time.max(), 20)
    yp = np.polyval(params, xp)
    plt.plot(xp, yp, 'b')

    # overplot an error band
    sig = 1.96 * np.std(corr - np.polyval(params, time)) / np.sqrt(n)
    plt.fill_between(xp, yp - sig, yp + sig,
                     color='b', alpha=0.1)
    #-----
    params = np.polyfit(time, alg, 0)
    xp = np.linspace(time.min(), time.max(), 20)
    yp = np.polyval(params, xp)
    plt.plot(xp, yp, 'g')

    # overplot an error band
    sig = 1.96 * np.std(alg - np.polyval(params, time)) / np.sqrt(n)
    plt.fill_between(xp, yp - sig, yp + sig,
                     color='g', alpha=0.1)

    plt.grid()
    line1, = plt.plot(time, tril, 'r.',label="Этап 1: метод трилатерации")
    line2, = plt.plot(time, corr,'b.', label="Этап 2: с учетом корректировки затухания сигнала")
    line3, = plt.plot(time, alg, 'g.',label="Этап 3: с учетом выстроенной топологии помещения ")
    plt.errorbar(time, tril, yerr=tril_int, ecolor='r', ls='none',alpha=0.3)
    plt.errorbar(time, corr, yerr=corr_int, ecolor='b', ls='none',alpha=0.3)
    plt.errorbar(time, alg, yerr=alg_int, ecolor='g', ls='none',alpha=0.3)

    plt.xlabel('Время, сек')
    plt.ylabel('Точность позиционирования, м')
    plt.xticks(np.arange(0, n+1, 10))
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=1), line2: HandlerLine2D(numpoints=1), line3: HandlerLine2D(numpoints=1)})
    print([np.mean(tril), 1.96 * np.std(tril)/np.sqrt(n)],
          [np.mean(corr), 1.96 * np.std(corr) / np.sqrt(n)],
          [np.mean(alg), 1.96 * np.std(alg) / np.sqrt(n)])
    plt.show()


def traektoria2(n):
    z = int(n / 15)
    nom, ost = divmod(tick, tochek)
    if ost == 0:
        if nom < z * 4:
            gl.user.pos += np.array([0, -20])
        if nom >= z * 4 and nom < z * 5:
            gl.user.pos += np.array([20, 0])
        if nom >= z * 5 and nom < z * 8:
            gl.user.pos += np.array([0, 20])
        if nom >= z * 8 and nom < z * 9:
            gl.user.pos += np.array([20, 0])
        if nom >= z * 9 and nom < z * 12:
            gl.user.pos += np.array([0, -20])
        if nom >= z * 12 and nom < z * 13:
            gl.user.pos += np.array([20, 0])
        if nom >= z * 13 and nom < z * 15:
            gl.user.pos += np.array([0, 20])
        if tick == n * tochek:
            postroy()

def traektoria3(n):
    z=int(n/17)
    nom, ost = divmod(tick, tochek)
    if ost == 0:
        if nom < 2*z:
            gl.user.pos += np.array([0, 20])
        if nom >= 2*z and nom < 3*z:
            gl.user.pos += np.array([10, 0])
        if nom >= 3*z and nom < 5*z:
            gl.user.pos += np.array([0, -20])
        if nom >= 5*z and nom < 6*z:
            gl.user.pos += np.array([10,0])
        if nom >= 6*z and nom < 8*z:
            gl.user.pos += np.array([0, 20])
        if nom >= 8*z and nom < 9*z:
            gl.user.pos += np.array([10,0])
        if nom >= 9*z and nom < 11*z:
            gl.user.pos += np.array([0, -20])
        if nom >= 11*z and nom < 12*z:
            gl.user.pos += np.array([10,0])
        if nom >= 12*z and nom < 14*z:
            gl.user.pos += np.array([0, 20])
        if nom >= 14*z and nom < 15*z:
            gl.user.pos += np.array([10,0])
        if nom >= 15*z and nom < 17*z:
            gl.user.pos += np.array([0, -20])
        if tick == n*tochek:
            postroy()


def check_num_AP(pos_errors):
    """
    """
    pos_errors.append(convert_to_m(np.linalg.norm(mean_point_coord - gl.user.pos)))
    if len(pos_errors) == 800:
        print(AP_NUMBER, 1.96 * np.std(np.array(pos_errors)) / np.sqrt(len(pos_errors)))
        sys.exit(0)


# ----------------- Now the party starts ------------------- #
carryOn = True
clock=pygame.time.Clock()
myfont = pygame.font.SysFont("monospace", 15)
tick = 0
while carryOn:
        tick += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                carryOn = False


        screen.fill(WHITE)
        gl.user.tick()

        for mat_id, boxes in rects.items():
            for box in boxes:
                box_pts = np.array(box.exterior.coords).astype(int)
                draw_walls(box_pts)
                if gl.user.ball_box.intersection(box):
                    for k in range(len(box_pts)-1):
                        front = sh.LineString([box_pts[k], box_pts[k + 1]])
                        #if gl.user.ball_box.intersection(front):
                            #gl.user.reflect(get_normal(front))

        estim_dist = np.zeros(AP_NUMBER)
        pl = np.zeros(AP_NUMBER)

        for i in range(AP_NUMBER):
            # pygame.draw.circle(screen, GREY, MEAS_POINT_COORDS[i], 4, 0)  # draw location of AP
            pl[i] = get_pathloss(gl.user.pos, MEAS_POINT_COORDS[i], FREQ)
            estim_dist[i] = get_dist(pl[i])
            # ---- nice performance --- #
            #draw_circle_around(MEAS_POINT_COORDS[i], estim_dist[i])
            #label = myfont.render(str(int(pl[i])), 1, GREY)  # label pathloss value in dB
            #screen.blit(label, MEAS_POINT_COORDS[i])

        # --- Trilateration part --- #
        min_variance, mean_point_coord = centroid(all_intersections(estim_dist))
        variances_data = store_var(min_variance)
        # draw_centroid(mean_point_coord)
        # --------------------------- #

        # trust_value = trust(np.array(variances_data), var_mean, var_std)  # better need to be: calculate, store in trust_value_deque

        # adequate_model(pos_errors, pos_error_array)

        error_m = convert_to_m(np.linalg.norm(gl.user.pos - mean_point_coord))
        #pack_of_errors_m[len(variances_data)] = error_m
        pack_of_errors_m[tick] = error_m

        #if variances_data[-1] > 0.2:
        new_mean, corr_vector = correct_position(pl)
        # --- FINAL ERR ESTIM --- #
        corrected_error_m = convert_to_m(np.linalg.norm(gl.user.pos - new_mean))
        #pack_of_errors_corrctd_m[len(variances_data)] = corrected_error_m
        pack_of_errors_corrctd_m[tick] = corrected_error_m

            #otrisovka(corr_vector)

        #if len(variances_data) == 800:
        # final_err_tril = np.array([np.mean(pack_of_errors_m),
        #                               1.96 * np.std(pack_of_errors_m) / np.sqrt(len(variances_data))])
        # final_err_alg = np.array([np.mean(pack_of_errors_corrctd_m),
        #                               1.96 * np.std(pack_of_errors_corrctd_m) / np.sqrt(len(variances_data))])
        final_err_tril = np.array([np.mean(pack_of_errors_m),
                                      1.96 * np.std(pack_of_errors_m) / np.sqrt(tick)])
        final_err_alg = np.array([np.mean(pack_of_errors_corrctd_m),
                                      1.96 * np.std(pack_of_errors_corrctd_m) / np.sqrt(tick)])

        #plot_errors(pack_of_errors, pack_of_errors_corrctd)
            #print(final_err_tril, final_err_alg)
            #sys.exit(0)
        #traektoria1(80)
        traektoria2(n)
        '''
        if len(variances_data) == 1000:
            plot_hist(convert_to_m(convert_to_m(np.array(variances_data))), 'мера разброса D вершин многоугольника, м2')
            sys.exit(0)'''

        # --- DRAW OVER --- #
        # for mat_id, boxes in rects.items():
        #     for box in boxes:
        #         box_pts = np.array(box.exterior.coords).astype(int)
        #         draw_walls(box_pts)
        # ------------------ #

        gl.user.draw(screen)
        # --- DRAW AP, label number  --- #
        # for i in range(AP_NUMBER):
        #     pygame.draw.circle(screen, BROWN, MEAS_POINT_COORDS[i], 5, 0)  # draw location of AP
        #     label_AP = myfont.render(str('AP '+str(i+1)), 2, BROWN)
        #     if MEAS_POINT_COORDS[i,0] > SCREENWIDTH/2:
        #         screen.blit(label_AP, [MEAS_POINT_COORDS[i,0]-50, MEAS_POINT_COORDS[i,1]-10])
        #     else:
        #         screen.blit(label_AP, [MEAS_POINT_COORDS[i,0]+20, MEAS_POINT_COORDS[i,1]-10])
        # ------------------ #

        print(tick)
        # experiment(tochnost, new_mean, tochnost_mean, tochnost_int)
        # experiment(tochnost_tril, mean_point_coord, tochnost_mean_tril, tochnost_int_tril, 1)
        #draw_grid()
        # keys = pygame.key.get_pressed()
        #check_num_AP(pos_errors)

        # if keys[pygame.K_ESCAPE] or len(variances_data) == 2400:
        #     final_err_tril = np.array([np.mean(pack_of_errors_m),
        #                                1.96 * np.std(pack_of_errors_m) / np.sqrt(len(variances_data))])
        #     final_err_alg = np.array([np.mean(pack_of_errors_corrctd_m),
        #                               1.96 * np.std(pack_of_errors_corrctd_m) / np.sqrt(len(variances_data))])
        #     #plot_errors(pack_of_errors, pack_of_errors_corrctd)
        #     print(final_err_tril, final_err_alg)
        #     sys.exit(0)

        pygame.display.flip()
        clock.tick(FPS)
pygame.quit()