import xml.etree.ElementTree as ET
import shapely.geometry as sh
from collections import defaultdict
from gl_vars import *
SVG_NS = "http://www.w3.org/2000/svg"


def parse_svg(fname="drawing_one_rect_1.svg"):
    map_svg = ET.parse(fname)
    map_root = map_svg.getroot()
    rects= defaultdict(list)
    for rect in map_svg.findall('.//{%s}rect' % SVG_NS):
        w, h, x_pos, y_pos = [float(rect.get(i)) for i in ['width', 'height','x','y']]
        id = rect.get('id')
        style = rect.get('style')
        style_list = str(style).split(';')
        fill = style_list[0].split(':')
        #print('{0} {1} {2} {3} {4} {5}'.format(x_pos, y_pos, w, h, id, fill[1]))
        mat = material_map[fill[1]]
        box = sh.box(x_pos*SCALE[0], y_pos*SCALE[1],
                     (x_pos + w)*SCALE[0], (y_pos+h)*SCALE[1])
        rects[mat].append(box)
    return rects

