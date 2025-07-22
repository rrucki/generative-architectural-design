import random
from itertools import combinations
from typing import List, Tuple, Optional

from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points

from src.urban_planner.config import WINDOW_WIDTH, WINDOW_HEIGHT, GENERATION_PARAMS


def _project_point_to_line(point: Tuple[float, float], line: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Helper function to project a point onto a line segment.

    Args:
        point: The (x, y) coordinates of the point to project.
        line: A list of two (x, y) tuples representing the line segment.

    Returns:
        The (x, y) coordinates of the projected point on the line.
    """
    line_geom = LineString(line)
    projected_point = nearest_points(Point(point), line_geom)[1]
    return projected_point.x, projected_point.y


def _generate_initial_points() -> List[Tuple[int, int]]:
    """
    Generates a set of random points within the window boundaries.

    Returns:
        A list of (x, y) integer coordinates.
    """
    return [
        (random.randint(0, WINDOW_WIDTH), random.randint(50, WINDOW_HEIGHT))
        for _ in range(GENERATION_PARAMS["NUM_POINTS"])
    ]


def _validate_plot_polygon(points: List[Tuple[int, int]]) -> Optional[Polygon]:
    """
    Validates if a set of points forms a suitable polygon for the outer plot boundary.

    The validation checks for:
    1. Minimum distance between any two points.
    2. The polygon being "simple" (i.e., its edges do not self-intersect).
    3. The polygon having a sufficient area relative to the window size.

    Args:
        points: A list of (x, y) coordinates.

    Returns:
        A Shapely Polygon object if valid, otherwise None.
    """
    min_dist = GENERATION_PARAMS["MIN_DISTANCE_BETWEEN_POINTS"]
    if any(Point(p1).distance(Point(p2)) < min_dist for p1, p2 in combinations(points, 2)):
        return None

    polygon = Polygon(points)
    min_area = (WINDOW_WIDTH * (WINDOW_HEIGHT - 50)) * GENERATION_PARAMS["MIN_POLYGON_AREA_RATIO"]

    if polygon.is_simple and polygon.area >= min_area:
        return polygon
    return None


def _create_asymmetrically_modified_polygon(plot_polygon: Polygon, offset: int) -> Optional[Polygon]:
    """
    Creates an intermediate polygon by moving one random edge inward.

    This function is the core of the complex generation logic. It performs the following steps:
    1. Selects a random edge of the input polygon.
    2. Calculates an inward normal vector.
    3. Moves the edge's two vertices inward along the normal by a random amount.
    4. Projects the new vertices onto their adjacent edges to maintain a clean, connected shape.

    Args:
        plot_polygon: The initial outer plot polygon.
        offset: A base value used to calculate the move distance.

    Returns:
        A new, modified Shapely Polygon if successful, otherwise None.
    """
    points = list(plot_polygon.exterior.coords)[:-1]
    num_points = len(points)

    edge_index = random.randint(0, num_points - 1)
    p1 = points[edge_index]
    p2 = points[(edge_index + 1) % num_points]

    edge_length = Point(p1).distance(Point(p2))
    if edge_length < GENERATION_PARAMS["MIN_EDGE_LENGTH"]:
        return None

    random_offset_factor = random.uniform(
        GENERATION_PARAMS["EDGE_MOVE_FACTOR_MIN"],
        GENERATION_PARAMS["EDGE_MOVE_FACTOR_MAX"]
    )
    move_distance = offset * random_offset_factor

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    normal_vector = (-dy / edge_length, dx / edge_length)

    new_p1 = (p1[0] + normal_vector[0] * move_distance, p1[1] + normal_vector[1] * move_distance)
    new_p2 = (p2[0] + normal_vector[0] * move_distance, p2[1] + normal_vector[1] * move_distance)

    prev_point = points[edge_index - 1]
    next_point = points[(edge_index + 2) % num_points]
    projected_p1 = _project_point_to_line(new_p1, [prev_point, p1])
    projected_p2 = _project_point_to_line(new_p2, [p2, next_point])

    if projected_p1 == p1 and projected_p2 == p2:
        return None

    modified_points = points.copy()
    modified_points[edge_index] = projected_p1
    modified_points[(edge_index + 1) % num_points] = projected_p2

    return Polygon(modified_points)


def generate_random_outline(offset: int) -> Tuple[List[Tuple[int, int]], Polygon]:
    """
    Generates a valid random plot outline and a permitted building area within it.

    This function orchestrates the generation process using a complex, two-step logic:
    1. An "intermediate" polygon is created by asymmetrically modifying the initial plot.
    2. This intermediate polygon is then buffered inward to create the final, smaller permitted outline.

    This process is repeated until a valid outline is generated or the attempt limit is reached.

    Args:
        offset: The base offset value used for modifications and buffering.

    Returns:
        A tuple containing:
        - A list of (x, y) coordinates for the outer plot boundary.
        - A Shapely Polygon object for the inner, permitted building area.

    Raises:
        RuntimeError: If a valid outline cannot be generated within the attempt limit.
    """
    for _ in range(GENERATION_PARAMS["MAX_GENERATION_ATTEMPTS"]):
        initial_points = _generate_initial_points()
        plot_polygon = _validate_plot_polygon(initial_points)
        if not plot_polygon:
            continue

        modified_polygon = _create_asymmetrically_modified_polygon(plot_polygon, offset)
        if not modified_polygon:
            continue

        permitted_outline = modified_polygon.buffer(-offset)

        min_area = (WINDOW_WIDTH * (WINDOW_HEIGHT - 50)) * GENERATION_PARAMS["MIN_POLYGON_AREA_RATIO"]
        if permitted_outline.is_simple and permitted_outline.area >= min_area and isinstance(permitted_outline, Polygon):
            plot_coords_int = [(round(x), round(y)) for x, y in plot_polygon.exterior.coords[:-1]]
            permitted_coords_int = [(round(x), round(y)) for x, y in permitted_outline.exterior.coords]

            if len(permitted_coords_int) >= 4:
                 return plot_coords_int, Polygon(permitted_coords_int)

    raise RuntimeError(f"Failed to generate a valid outline after {GENERATION_PARAMS['MAX_GENERATION_ATTEMPTS']} attempts.")


def generate_desired_center(plot_outline: Polygon, offset: int) -> Point:
    """
    Generates a random point within a buffered inner area of the plot.

    This is useful for finding a valid starting point or target location for an agent
    that is guaranteed to be well within the plot's boundaries.

    Args:
        plot_outline: The polygon defining the area to generate a point within.
        offset: A base value used to calculate the buffer distance.

    Returns:
        A Shapely Point object representing the generated center.
    """
    buffer_dist = offset * GENERATION_PARAMS["CENTER_BUFFER_FACTOR"]
    inner_area = plot_outline.buffer(buffer_dist, join_style="mitre")

    if inner_area.is_empty:
        return plot_outline.centroid

    min_x, min_y, max_x, max_y = inner_area.bounds

    for _ in range(GENERATION_PARAMS["MAX_GENERATION_ATTEMPTS"]):
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if inner_area.contains(point):
            return point

    return inner_area.representative_point()