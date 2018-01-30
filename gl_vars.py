import numpy as np
import itertools

GREEN = (0, 100, 0)
GREY = (100, 100, 100)
BROWN = (165, 42, 42)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
SVG_properties = (744, 1052)


SCREENWIDTH = int(400)
SCREENHEIGHT = int(600)
SCALE = (SCREENWIDTH/SVG_properties[0], SCREENHEIGHT/SVG_properties[1])

FPS = 30
N_LEFT = np.array([1, 0])
N_RIGHT = np.array([-1, 0])
N_CEILING = np.array([0, 1])
N_FLOOR = np.array([0, -1])

material_map = {'#0000ff':0, '#999999':1}
material_colors = [BLUE, GREY]
material_PL = [3, 15]
FREQ = 2.4e9
LAMBDA = 3e8/FREQ
H3 = 20  # Hervanta factor
#START_POINT = np.array([SCREENWIDTH*7/10, SCREENHEIGHT*8/10])
#START_POINT = np.array([SCREENWIDTH*1/2, SCREENHEIGHT*1/2])
#START_POINT = np.array([100,50]) # tr1
START_POINT = np.array([80,250]) #tr2
#START_POINT = np.array([50,50]) # tr3

# 3 AP
# MEAS_POINT_COORDS = np.array([(380,25), # coordinates of Access points
#                              (25, 300),
#                              (380, 575)]

# 4 AP
MEAS_POINT_COORDS = np.array([(25, 25), # coordinates of Access points
                              (25, 575),
                              (375, 25),
                              (375, 575)])

# 5 AP
# MEAS_POINT_COORDS = np.array([(380,200),
#                               (200, 25), # coordinates of Access points
#                               (25,200),
#                               (25, 575),
#                               (380, 575)])


AP_NUMBER = MEAS_POINT_COORDS.shape[0]
sum_peresech = np.sum(ind for ind in range(AP_NUMBER))

CLOSEST_POINTS_NUM = AP_NUMBER  # how much points to consider in order to find position of ball

CORRECTION_PL = np.array(list(itertools.product([0, -material_PL[0]], repeat=AP_NUMBER)))

class GL():
    def __init__(self):
        self.user = None


gl = GL()