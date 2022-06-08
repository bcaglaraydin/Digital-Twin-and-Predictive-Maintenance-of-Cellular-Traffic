from shapely.geometry import Polygon, Point
import random


def new_point_generator(place: str) -> str:
    if place == "ITU":
        polygon = Polygon([(41.106905, 29.015169), (41.110811, 29.036618),
                           (41.100261, 29.036788), (41.100295, 29.021511)])
    elif place == "Taksim Square":
        polygon = Polygon([(41.036526, 28.984618), (41.037637, 28.984688),
                           (41.037485, 28.987479), (41.036929, 28.987115)])

    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return str(p.x) + "-" + str(p.y)
