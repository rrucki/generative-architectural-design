from enum import Enum


class Stage(int, Enum):
    """Defines the stages of the curriculum learning task."""
    PLACEMENT = 0
    AREA = 1
    WIDTH = 2
    ANGLE = 3


class ObsKeys(str, Enum):
    """Defines the keys for the observation space dictionary."""
    AGENT = "agent"
    AREA_DIFFERENCE = "area_difference"
    VERTEX_VECTORS = "vertex_vectors"
    WIDTH_DIFFERENCE = "width_difference"
    ANGLE_DIFFERENCES = "angle_differences"
    BUILDING_AREA = "building_area"
    OUTSIDE_SHAPE = "outside_shape"
    STAGE = "stage"
    NEAREST_BOUNDARY_POINTS = "nearest_boundary_points"
    REMAINING_MOVES = "remaining_moves"
