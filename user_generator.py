"""Module for generating users"""
import json
import random
from random import randint

from shapely.geometry import Point, Polygon
from move_point_generator import get_place_polygon


def new_users_json(count: int, random_weight: bool, place: str) -> str:
    """Return a bunch of user coordinates with weights

    Args:
        count (int): Number of users to generate
        random_weight (bool): Whether to randomize the weights of users (weight defaults to 50)
        place (str): The location users should be in

    Returns:
        str: Returns a json string with user data
    """
    polygon = get_place_polygon(place)
    users = []
    for index in range(count):
        p = _get_random_point_in_polygon(polygon)
        w = randint(1, 100) if random_weight else 50
        user = {"userID": index, "x": p.x, "y": p.y, "weight": w, "color": 0}
        users.append(user)

    return json.dumps(users)


def _get_random_point_in_polygon(poly: Polygon) -> Point:
    """Return a random point in given polygon"""
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p
