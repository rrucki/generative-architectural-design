from typing import Tuple, List, Dict
from shapely.geometry import Polygon

from urban_planner.config import TARGET_ANGLE_DEGREES, \
    ANGLE_TOLERANCE_DEGREES
from urban_planner.types import Stage

from urban_planner.utils.geometry_utils import distance_to_polygon, \
    calculate_rectangular_width, calculate_polygon_angles


class OutlineAssessment:
    """
    A stateful calculator that assesses how well a building outline meets the required criteria.

    This class tracks the state of a building's outline relative to a permitted plot area.
    It calculates key metrics such as:
    - The distance of each outline vertex from the permitted area.
    - The difference between the outline's current area and the target area range.
    - A checklist of completed objectives.
    - A total reward score based on these metrics.

    It is designed to be updated with new outline coordinates at each step of the simulation.
    """
    def __init__(self, permitted_plot: Polygon,
                 initial_outline_coords: List[Tuple[int, int]],
                 area_range: Tuple[int, int],
                 width_range: Tuple[int, int]):
        """
        Initializes the OutlineAssessment.

        Args:
            permitted_plot: A Shapely Polygon representing the area where the building is allowed.
            initial_outline_coords: The starting coordinates of the building's outline.
            area_range: A tuple (min_area, max_area) defining the target area for the building.
            width_range: A tuple (min_width, max_width) defining the target width for the building.
        """
        self.permitted_plot = permitted_plot
        self.area_range = area_range
        self.width_range = width_range

        self.current_outline_coords: List[Tuple[int, int]] = []
        self.current_area: int = 0
        self.outside_shape_distances: List[float] = []
        self.nearest_boundary_points: List[Tuple[float, float]] = []
        self.area_difference: int = 0
        self.width_difference: float = 0.0
        self.angle_differences: List[float] = []

        self.current_width: float = 0.0
        self.current_angles: List[float] = []
        self.is_simple: bool = True

        self.checklist: Dict[Stage, bool] = {stage: False for stage in Stage}

    def update(self, new_outline_coords: List[Tuple[int, int]]):
        """
        Updates the assessment state based on new outline coordinates.

        This is the main method to call at each simulation step. It recalculates
        all internal metrics based on the provided coordinates.

        Args:
            new_outline_coords: The new list of (x, y) coordinates for the building outline.
        """
        self.current_outline_coords = new_outline_coords

        if len(self.current_outline_coords) >= 3:
            self.current_area = int(Polygon(self.current_outline_coords).area)
        else:
            self.current_area = 0

        self._check_distances()
        self._check_area_difference()
        self._check_width()
        self._check_angles()
        self._check_self_intersection()
        self._update_checklist()

    def _check_width(self) -> None:
        """Calculates the outline's width and its difference from the target range."""
        if len(self.current_outline_coords) < 4:
            self.current_width = 0
            self.width_difference = self.width_range[1]
            return

        self.current_width = calculate_rectangular_width(
            self.current_outline_coords)
        min_w, max_w = self.width_range
        if self.current_width < min_w:
            self.width_difference = min_w - self.current_width
        elif self.current_width > max_w:
            self.width_difference = self.current_width - max_w
        else:
            self.width_difference = 0

    def _check_angles(self) -> None:
        """Calculates the outline's angles and their strict excess beyond tolerance."""
        self.current_angles = calculate_polygon_angles(
            self.current_outline_coords)
        self.angle_differences = [
            max(0.0, abs(angle - TARGET_ANGLE_DEGREES) - ANGLE_TOLERANCE_DEGREES)
            for angle in self.current_angles
        ]

    def _check_distances(self) -> None:
        """Calculates the distance of each outline vertex to the permitted plot boundary."""
        self.outside_shape_distances = []
        self.nearest_boundary_points = []
        for coord in self.current_outline_coords:
            distance, nearest_point = distance_to_polygon(self.permitted_plot, coord)
            self.outside_shape_distances.append(distance)
            self.nearest_boundary_points.append(nearest_point)

    def _check_area_difference(self) -> None:
        """Calculates the difference between the current area and the target area range."""
        min_area, max_area = self.area_range
        if self.current_area < min_area:
            self.area_difference = self.current_area - min_area
        elif self.current_area > max_area:
            self.area_difference = self.current_area - max_area
        else:
            self.area_difference = 0

    def _check_self_intersection(self) -> None:
        """Checks whether non-adjacent quadrilateral edges intersect (bowtie topology)."""
        if len(self.current_outline_coords) < 4:
            self.is_simple = False
            return

        p0, p1, p2, p3 = self.current_outline_coords[:4]

        def ccw(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> bool:
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        def intersect(a: Tuple[int, int], b: Tuple[int, int],
                      c: Tuple[int, int], d: Tuple[int, int]) -> bool:
            return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

        self.is_simple = not (intersect(p0, p1, p2, p3) or intersect(p1, p2, p3, p0))

    def _update_checklist(self) -> None:
        """Updates the completion checklist based on the current state."""
        self.checklist[Stage.PLACEMENT] = all(
            dist == 0 for dist in self.outside_shape_distances)
        self.checklist[Stage.AREA] = self.area_difference == 0
        self.checklist[Stage.WIDTH] = self.width_difference == 0
        self.checklist[Stage.ANGLE] = (
            all(diff == 0 for diff in self.angle_differences) and self.is_simple
        )
