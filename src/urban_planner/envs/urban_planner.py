from enum import Enum
from typing import Optional, Dict, Any, Tuple, Type

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.urban_planner.config import *
from src.urban_planner.domain.geometries import Plot, Outline
from src.urban_planner.logic.assessment import OutlineAssessment
from src.urban_planner.presentation.renderer import Renderer



class ObsKeys(str, Enum):
    AGENT = "agent"
    TARGET = "target"
    AREA_DIFFERENCE = "area_difference"
    BUILDING_AREA = "building_area"
    OUTSIDE_SHAPE = "outside_shape"
    REMAINING_MOVES = "remaining_moves"
    TARGET_AREA = "target_area"


class UrbanPlannerEnv(gym.Env):
    """
    A Gymnasium environment for the urban planning problem.

    The agent's goal is to adjust a building's outline (a quadrilateral)
    to meet two main criteria:
    1. The entire outline must be within the permitted area of the plot.
    2. The outline's area must be within a specified range.

    The observation space includes information about the agent's position,
    the target, area differences, and remaining moves.
    The action space allows for the direct setting of the four vertices
    of the outline.
    """
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": RENDER_FPS}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 renderer_class: Type[Renderer] = Renderer,
                 assessment_class: Type[OutlineAssessment] = OutlineAssessment):
        super().__init__()
        self.render_mode = render_mode
        self.assessment_class = assessment_class

        self.plot: Optional[Plot] = None
        self.outline: Optional[Outline] = None
        self.steps_taken: int = 0
        self.assessment: Optional[OutlineAssessment] = None
        self.last_reward: float = 0.0
        self.renderer = renderer_class(
            WINDOW_WIDTH, WINDOW_HEIGHT, self.metadata["render_fps"]
        ) if render_mode == "human" else None

        if self.renderer:
            self.renderer.init()

        self.observation_space = spaces.Dict({
            ObsKeys.AGENT.value: spaces.Box(0, WINDOW_WIDTH, (4, 2), dtype=np.int64),
            ObsKeys.AREA_DIFFERENCE.value: spaces.Box(0, MAX_AREA, (1,), dtype=np.int64),
            ObsKeys.BUILDING_AREA.value: spaces.Box(0, MAX_AREA, (1,), dtype=np.int64),
            ObsKeys.OUTSIDE_SHAPE.value: spaces.Box(0, WINDOW_WIDTH, (4,), dtype=np.int64),
            ObsKeys.REMAINING_MOVES.value: spaces.Box(0, MODEL_COMPATIBILITY_STEPS, (1,), dtype=np.int64),
            ObsKeys.TARGET.value: spaces.Box(0, WINDOW_WIDTH, (4, 2), dtype=np.int64),
            ObsKeys.TARGET_AREA.value: spaces.Box(0, MAX_AREA, (1,), dtype=np.int64)
        })

        self.action_space = spaces.MultiDiscrete([
            WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1,
            WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1,
            WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1,
            WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1,
        ])

    def _get_obs(self) -> Dict[str, np.ndarray]:
        target_coords = list(self.plot.permitted_outline.exterior.coords[:-1])
        while len(target_coords) < 4:
            target_coords.append(target_coords[-1])

        remaining_moves_for_obs = max(0, MODEL_COMPATIBILITY_STEPS - self.steps_taken)

        return {
            ObsKeys.AGENT.value: np.array(self.outline.coords, dtype=np.int64),
            ObsKeys.TARGET.value: np.array(target_coords[:4], dtype=np.int64),
            ObsKeys.AREA_DIFFERENCE.value: np.array([self.assessment.area_difference], dtype=np.int64),
            ObsKeys.BUILDING_AREA.value: np.array([self.assessment.current_area], dtype=np.int64),
            ObsKeys.OUTSIDE_SHAPE.value: np.array(self.assessment.outside_shape_distances, dtype=np.int64),
            ObsKeys.REMAINING_MOVES.value: np.array([remaining_moves_for_obs], dtype=np.int64),
            ObsKeys.TARGET_AREA.value: np.array([self.plot.area_range[0]], dtype=np.int64)
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "distance": self.assessment.outside_shape_distances,
            "area_difference": self.assessment.area_difference,
            "is_solution": all(self.assessment.checklist),
            "steps": self.steps_taken
        }

    def _calculate_reward(self) -> float:
        """Calculates and returns the final reward for the current step."""
        base_reward = float(self.assessment.total_reward())
        penalty = self.steps_taken * PENALTY_PER_MOVE
        return base_reward - penalty

    def _check_termination(self) -> bool:
        """Checks if the termination conditions for the episode are met."""
        solution_found = all(self.assessment.checklist)
        max_steps_reached = self.steps_taken >= INTERNAL_MAX_STEPS

        if solution_found:
            print(f"Solution found in {self.steps_taken} steps!")

        return solution_found or max_steps_reached

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.steps_taken = 0

        self.plot = Plot.from_random_generation()
        self.outline = Outline.from_default_square()
        self.assessment = self.assessment_class(
            self.plot.permitted_outline,
            self.outline.coords,
            self.plot.area_range
        )

        self.last_reward = self._calculate_reward()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[
        Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.steps_taken += 1

        new_coords = action.reshape(4, 2).tolist()

        self.outline.coords = new_coords
        self.outline.update_geometry()
        self.assessment.update(new_coords)

        reward = self._calculate_reward()
        self.last_reward = reward

        terminated = self._check_termination()
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        """Renders the current state of the environment."""
        if self.render_mode == "human" and self.renderer:
            info_lines = []
            if self.assessment:
                info_lines = [
                    f"Area Range: {self.plot.area_range} | Building Area: {self.outline.area}",
                    f"Reward: {self.last_reward:.2f} | Remaining Moves: {INTERNAL_MAX_STEPS - self.steps_taken}",
                    f"Checklist: {self.assessment.checklist}"
                ]

            self.renderer.render_frame(
                plot=self.plot,
                outline=self.outline,
                info_lines=info_lines
            )


    def close(self):
        if self.renderer:
            self.renderer.close()