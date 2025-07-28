import math
from typing import Tuple, List

import numpy as np
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

    edges = [LineString((coords[j], coords[(j + 1) % num_vertices])) for j in
             range(num_vertices)]

    avg1 = (Point(coords[i]).distance(edges[2]) + Point(
        coords[(i + 1) % num_vertices]).distance(edges[2])) / 2
    avg2 = (Point(coords[(i + 2) % num_vertices]).distance(edges[0]) + Point(
        coords[(i + 3) % num_vertices]).distance(edges[0])) / 2
    first_set_min_distance = min(avg1, avg2)

    avg3 = (Point(coords[(i + 1) % num_vertices]).distance(edges[3]) + Point(
        coords[(i + 2) % num_vertices]).distance(edges[3])) / 2
    avg4 = (Point(coords[(i + 3) % num_vertices]).distance(edges[1]) + Point(
        coords[(i + 4) % num_vertices]).distance(edges[1])) / 2
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
            indices = [i, (i + 1) % coord_count, (i + 2) % coord_count,
                       (i + 3) % coord_count]
            partial_coords = [coords[index] for index in indices]

            rest_coords.remove(partial_coords[1])
            rest_coords.remove(partial_coords[2])
            partial_poly = Polygon(partial_coords)

            try:
                if partial_poly.is_valid and partial_poly.within(
                        Polygon(coords)):
                    width.append(calculate_rectangular_width(partial_coords))
                    width += calculate_width(rest_coords)
                    break
            except shapely.errors.GEOSException:
                continue

        return width

    else:
        return [calculate_rectangular_width(coords)]


def calculate_polygon_angles(coords: List[Tuple[int, int]]) -> List[float]:
    """
    Calculates the interior angles of a polygon at each vertex.

    Args:
        coords: A list of (x, y) coordinates for the polygon.

    Returns:
        A list of floats representing the angle in degrees at each vertex.
    """
    angles = []
    num_vertices = len(coords)
    if num_vertices < 3:
        return [0.0] * num_vertices

    for i in range(num_vertices):
        p_prev = coords[i - 1]
        p_curr = coords[i]
        p_next = coords[(i + 1) % num_vertices]

        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag_v1 * mag_v2 == 0:
            angles.append(180.0)
            continue

        angle_rad = math.acos(
            np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0))
        angles.append(math.degrees(angle_rad))

    return angles


def distance_to_polygon(polygon: Polygon,
                        target_point: Tuple[float, float]) -> Tuple[
    float, Tuple[float, float]]:
    """
     Calculates the distance from a point to a polygon and finds the nearest point on its boundary.

    Args:
        polygon: The Shapely Polygon to check against.
        target_point: The (x, y) coordinates of the point to check.

    Returns:
        A tuple containing:
         - A non-negative float representing the distance. It is 0 if the point
           is inside or on the boundary of the polygon.
        - The coordinates of the nearest point on the polygon's exterior.
          If the target_point is inside, it returns the target_point itself.
    """
    point_geom = Point(target_point)

    if polygon.contains(point_geom):
        return 0, target_point

    nearest_point_on_boundary, _ = nearest_points(polygon.boundary, point_geom)
    distance = point_geom.distance(nearest_point_on_boundary)

    return distance, (nearest_point_on_boundary.x, nearest_point_on_boundary.y)