from gl_vars import *
from Map_parser import *
import shapely.geometry as sh
import pygame


class User:
    def __init__(self, pos=START_POINT, v=np.ones(2), radius=4):
        self.r = radius
        self.pos = np.array(pos, dtype=float)
        self.v = np.array(v, dtype=float)
        self.bbox = np.array([[-self.r, -self.r],
                              [+self.r, +self.r],
                              ])


    def draw(self, screen):
        pygame.draw.circle(screen, GREEN, self.pos.astype(int), self.r, 0)

    def tick(self):
        self.pos += self.v
        bbox_offset = self.pos + self.v
        bounds = self.bbox + np.vstack([bbox_offset,bbox_offset])
    # FIXME> array manipulations!
        self.ball_box = sh.box(bounds[0,0], bounds[0,1], bounds[1,0], bounds[1,1])
        ball_box_pts = np.array(self.ball_box.exterior.coords)
        #for i in range(len(ball_box_pts) - 1):
        self.ball_fronts = [sh.LineString([ball_box_pts[i], ball_box_pts[i + 1]]) for i in range(len(ball_box_pts) - 1)]
            #self.ball_fronts = sh.LineString([ball_box_pts[i], ball_box_pts[i + 1]])

        if bounds[1, 0] > SCREENWIDTH-10:
            self.reflect(N_RIGHT)
        elif bounds[0, 0] < 10:
            self.reflect(N_LEFT)
        elif bounds[0, 1] < 10:
            self.reflect(N_CEILING)
        elif bounds[1, 1] > SCREENHEIGHT-10:
            self.reflect(N_FLOOR)

    def reflect(self, normal: np.array):
        """
        Reflect against normal
        :param normal: np.array - the normal of the surface for reflection
         :returns: none
        """
        self.v += normal * 2 * np.dot(-self.v, normal)


