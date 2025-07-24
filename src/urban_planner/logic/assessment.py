from typing import Tuple, List
from shapely.geometry import Polygon

from urban_planner.utils.geometry_utils import distance_outside_shape


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
                 area_range: Tuple[int, int]):
        """
        Initializes the OutlineAssessment.

        Args:
            permitted_plot: A Shapely Polygon representing the area where the building is allowed.
            initial_outline_coords: The starting coordinates of the building's outline.
            area_range: A tuple (min_area, max_area) defining the target area for the building.
        """
        self.permitted_plot = permitted_plot
        self.area_range = area_range

        self.current_outline_coords: List[Tuple[int, int]] = []
        self.current_area: int = 0
        self.outside_shape_distances: List[int] = []
        self.nearest_boundary_points: List[Tuple[float, float]] = []
        self.area_difference: int = 0
        self.checklist: List[bool] = [False, False]


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
        self._update_checklist()

    def _check_distances(self) -> None:
        """Calculates the distance of each outline vertex to the permitted plot boundary."""
        self.outside_shape_distances = []
        self.nearest_boundary_points = []
        for coord in self.current_outline_coords:
            distance, nearest_point = distance_outside_shape(self.permitted_plot, coord)
            self.outside_shape_distances.append(abs(distance))
            self.nearest_boundary_points.append(nearest_point)

    def _check_area_difference(self) -> None:
        """Calculates the difference between the current area and the target area range."""
        min_area, max_area = self.area_range
        if self.current_area < min_area:
            self.area_difference = min_area - self.current_area
        elif self.current_area > max_area:
            self.area_difference = self.current_area - max_area
        else:
            self.area_difference = 0

    def _update_checklist(self) -> None:
        """Updates the completion checklist based on the current state."""
        all_nodes_in_place = all(
            dist == 0 for dist in self.outside_shape_distances)

        self.checklist[0] = all_nodes_in_place
        self.checklist[1] = (self.area_difference == 0)
