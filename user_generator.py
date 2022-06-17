from shapely.geometry import Polygon, Point
import random
import json
from random import randint


def new_users_json(count: int, random_weight: bool, place: str) -> str:
    if place == "ITU":
        polygon = Polygon([(41.106905, 29.015169), (41.110811, 29.036618),
                           (41.100261, 29.036788), (41.100295, 29.021511)])
    elif place == "Taksim Square":
        polygon = Polygon([(41.036526, 28.984618), (41.037637, 28.984688),
                           (41.037485, 28.987479), (41.036929, 28.987115)])
    users = []
    for index in range(count):
        p = get_random_point_in_polygon(polygon)
        w = randint(1, 100) if random_weight else 50
        user = {'userID': index,
                'x': p.x,
                'y': p.y,
                'weight': w,
                'color': 0
                }
        users.append(user)

    return json.dumps(users)


def get_random_point_in_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p


# print(new_users_json(1, True, "ITU"))
