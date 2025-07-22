import math
from typing import Tuple, List

import shapely.errors
from shapely import LineString
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


def calculate_rectangular_width(coords: List[Tuple[int, int]]) -> float:
    """
    Calculates an approximation of the smaller dimension (width) of a quadrilateral.

    It works by averaging the distances of vertices to their opposite edges for
    both pairs of opposite edges, and then returns the smaller of these two
    calculated dimensions.

    Args:
        coords: A list of four (x, y) coordinates representing the quadrilateral.

    Returns:
        A float representing the smaller dimension (width) of the shape.
    """
    num_vertices = len(coords)
    i = 0

    edges = [LineString((coords[j], coords[(j + 1) % num_vertices])) for j in range(num_vertices)]

    avg1 = (Point(coords[i]).distance(edges[2]) + Point(coords[(i + 1) % num_vertices]).distance(edges[2])) / 2
    avg2 = (Point(coords[(i + 2) % num_vertices]).distance(edges[0]) + Point(coords[(i + 3) % num_vertices]).distance(edges[0])) / 2
    first_set_min_distance = min(avg1, avg2)

    avg3 = (Point(coords[(i + 1) % num_vertices]).distance(edges[3]) + Point(coords[(i + 2) % num_vertices]).distance(edges[3])) / 2
    avg4 = (Point(coords[(i + 3) % num_vertices]).distance(edges[1]) + Point(coords[(i + 4) % num_vertices]).distance(edges[1])) / 2
    second_set_min_distance = min(avg3, avg4)

    return min(first_set_min_distance, second_set_min_distance)


def calculate_width(coords: List[Tuple[int, int]]) -> List[float]:
    """
    Recursively calculates the widths of a complex polygon by decomposing it
    into quadrilaterals.

    Args:
        coords: A list of (x, y) coordinates for the polygon.

    Returns:
        A list of floats, where each float is the calculated width of a
        decomposed rectangular part of the original polygon.
    """
    coord_count = len(coords)
    width = []

    if coord_count > 4:
        for i in range(coord_count):
            rest_coords = coords.copy()
            indices = [i, (i + 1) % coord_count, (i + 2) % coord_count, (i + 3) % coord_count]
            partial_coords = [coords[index] for index in indices]

            rest_coords.remove(partial_coords[1])
            rest_coords.remove(partial_coords[2])
            partial_poly = Polygon(partial_coords)

            try:
                if partial_poly.is_valid and partial_poly.within(Polygon(coords)):
                    width.append(calculate_rectangular_width(partial_coords))
                    width += calculate_width(rest_coords)
                    break
            except shapely.errors.GEOSException:
                continue

        return width

    else:
        return [calculate_rectangular_width(coords)]


def distance_outside_shape(polygon: Polygon, target_point: Tuple[float, float]) -> int:
    """
    Calculates the distance from a point to the exterior of a polygon.

    This is a useful function for reward shaping in reinforcement learning.

    Args:
        polygon: The Shapely Polygon to check against.
        target_point: The (x, y) coordinates of the point to check.

    Returns:
        0 if the point is inside the polygon, otherwise a negative integer
        representing the distance to the polygon's exterior.
    """
    target_point = Point(target_point)

    if polygon.contains(target_point):
        return 0

    nearest_point, _ = nearest_points(polygon.exterior, target_point)

    return -round(target_point.distance(nearest_point))
