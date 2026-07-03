import random
from dataclasses import dataclass, field
from typing import List, Tuple

from shapely import Polygon

from urban_planner.config import WINDOW_WIDTH, WINDOW_HEIGHT, MIN_TARGET_WIDTH, \
    TARGET_WIDTH_MIN_FACTOR, TARGET_WIDTH_MAX_FACTOR
from urban_planner.utils import generate_random_outline


@dataclass
class Outline:
    """A data class representing the building's outline."""
    coords: List[Tuple[int, int]]
    polygon: Polygon = field(init=False)
    area: int = field(init=False)

    def __post_init__(self):
        self.update_geometry()

    def update_geometry(self):
        if len(self.coords) >= 3:
            self.polygon = Polygon(self.coords)
            self.area = int(self.polygon.area)
        else:
            self.polygon = Polygon()
            self.area = 0

    @classmethod
    def from_default_square(cls, size: int = 100) -> 'Outline':
        """
        Creates a default square outline centered in the window.

        Args:
            size: The half-width and half-height of the square. The total
                  sidelength will be 2 * size.

        Returns:
            An Outline instance representing the default square.
        """
        center_x, center_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        initial_coords = [
            (center_x - size, center_y - size),
            (center_x + size, center_y - size),
            (center_x + size, center_y + size),
            (center_x - size, center_y + size)
        ]
        return cls(coords=initial_coords)


@dataclass
class Plot:
    """A data class representing the plot of land, using generation logic."""
    coords: List[Tuple[int, int]]
    permitted_outline: Polygon
    area_range: Tuple[int, int]
    width_range: Tuple[int, int]

    @classmethod
    def from_random_generation(cls) -> 'Plot':
        """
        Creates a new Plot instance with a procedurally generated shape.

        This factory method uses the `generate_random_outline` utility to create
        a complex plot shape and a permitted building area within it. It then
        calculates a target area range based on a random ratio of the
        total plot area.

        Returns:
            A Plot instance with randomized geometric properties.
        """
        offset = random.randint(4, 8) * 10
        coords, permitted_outline = generate_random_outline(offset)

        polygon = Polygon(coords)
        covered_area_ratio = random.uniform(0.1, 0.2)
        max_area = polygon.area * covered_area_ratio
        area_range = (int(max_area * 0.75), int(max_area))

        min_x, min_y, max_x, max_y = permitted_outline.bounds
        plot_width = max_x - min_x
        plot_height = max_y - min_y
        max_achievable_width = min(plot_width, plot_height)

        lower_bound_for_random = max(MIN_TARGET_WIDTH,
                                     int(max_achievable_width * TARGET_WIDTH_MIN_FACTOR))
        upper_bound_for_random = int(
            max_achievable_width * TARGET_WIDTH_MAX_FACTOR)

        if lower_bound_for_random >= upper_bound_for_random:
            width_range = (
                lower_bound_for_random, lower_bound_for_random + 50)
        else:
            v1 = random.randint(lower_bound_for_random, upper_bound_for_random)
            v2 = random.randint(lower_bound_for_random, upper_bound_for_random)
            width_range = (min(v1, v2), max(v1, v2))

        return cls(
            coords=coords,
            permitted_outline=permitted_outline,
            area_range=area_range,
            width_range=width_range
        )
