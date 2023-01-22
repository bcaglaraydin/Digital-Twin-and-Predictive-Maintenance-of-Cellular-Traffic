"""Module for generating target moving points for users"""
import random

from shapely.geometry import Point, Polygon


def get_place_polygon(place: str) -> Polygon:
    """Return a polygon having the coordinates of the corners of the given location"""
    # TODO : Enum
    if place == "ITU":
        polygon = Polygon(
            [
                (41.106905, 29.015169),
                (41.110811, 29.036618),
                (41.100261, 29.036788),
                (41.100295, 29.021511),
            ]
        )
    elif place == "Taksim Square":
        polygon = Polygon(
            [
                (41.036526, 28.984618),
                (41.037637, 28.984688),
                (41.037485, 28.987479),
                (41.036929, 28.987115),
            ]
        )
    else:
        raise ValueError("Location is ambiguous!")
    return polygon


def new_point_generator(place: str) -> str:
    """Create a coordiante in given location

    Args:
        place (str): The location the point should be in

    Returns:
        str: Return coordinates as "x-y" string
    """
    polygon = get_place_polygon(place)

    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return str(p.x) + "-" + str(p.y)
