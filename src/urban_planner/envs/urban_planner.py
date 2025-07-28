import logging
import multiprocessing
import time

from typing import Optional, Dict, Any, Tuple, Type, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from urban_planner.config import *
from urban_planner.domain.geometries import Plot, Outline
from urban_planner.logic.assessment import OutlineAssessment
from urban_planner.presentation.renderer import Renderer

logger = logging.getLogger(__name__)

class UrbanPlannerEnv(gym.Env):
    """
    A Gymnasium environment for the urban planning problem.
    """
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": RENDER_FPS}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 generation_queue: Optional[multiprocessing.Queue] = None,
                 renderer_class: Type[Renderer] = Renderer,
                 assessment_class: Type[OutlineAssessment] = OutlineAssessment):
        super().__init__()
        self.render_mode = render_mode
        self.assessment_class = assessment_class
        self.generation_queue = generation_queue
        self.plot: Optional[Plot] = None
        self.outline: Optional[Outline] = None
        self.steps_taken: int = 0
        self.assessment: Optional[OutlineAssessment] = None
        self.last_reward: float = 0.0

        self._renderer_class = renderer_class
        self.renderer: Optional[Renderer] = None

        self.observation_space = spaces.Dict({
            ObsKeys.AGENT.value: spaces.Box(0, WINDOW_WIDTH, (4, 2), dtype=np.int64),
            ObsKeys.NEAREST_BOUNDARY_POINTS.value: spaces.Box(0, WINDOW_WIDTH, (4, 2), dtype=np.int64),
            ObsKeys.OUTSIDE_SHAPE.value: spaces.Box(0, WINDOW_WIDTH, (4,), dtype=np.int64),
            ObsKeys.BUILDING_AREA.value: spaces.Box(0, MAX_AREA, (1,), dtype=np.int64),
            ObsKeys.AREA_DIFFERENCE.value: spaces.Box(0, MAX_AREA, (1,), dtype=np.int64),
            ObsKeys.REMAINING_MOVES.value: spaces.Box(0, MODEL_COMPATIBILITY_STEPS, (1,), dtype=np.int64),
        })

        self.action_space = spaces.MultiDiscrete([
            WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1,
            WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1,
            WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1,
            WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1,
        ])

    def _init_renderer(self):
        """Initializes the renderer instance if it doesn't exist."""
        if self.renderer is None and self.render_mode == "human":
            self.renderer = self._renderer_class(
                WINDOW_WIDTH, WINDOW_HEIGHT, self.metadata["render_fps"]
            )
            self.renderer.init()

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.steps_taken = 0

        if self.generation_queue:
            start_time = time.monotonic()
            self.plot = self.generation_queue.get()
            wait_time = time.monotonic() - start_time

            if wait_time > 0.1:
                logger.warning(
                    f"Queue was empty. Waited {wait_time:.2f}s for a new environment. "
                    f"Consider increasing generator performance or queue size."
                )
        else:
            self.plot = Plot.from_random_generation()

        self.outline = Outline.from_default_square()
        self.assessment = self.assessment_class(
            self.plot.permitted_outline,
            self.outline.coords,
            self.plot.area_range
        )
        self.assessment.update(self.outline.coords)
        self.last_reward = self._calculate_reward(terminated=False)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.steps_taken += 1
        new_coords = action.reshape(4, 2).tolist()
        self.outline.coords = new_coords
        self.outline.update_geometry()
        self.assessment.update(new_coords)
        terminated = self._check_termination()
        reward = self._calculate_reward(terminated)
        self.last_reward = reward
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        self._init_renderer()
        if self.renderer:
            info_lines = self._get_info_for_render()
            self.renderer.render_frame(
                plot=self.plot,
                outline=self.outline,
                info_lines=info_lines
            )

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        remaining_moves_for_obs = max(0, MODEL_COMPATIBILITY_STEPS - self.steps_taken)
        return {
            ObsKeys.AGENT.value: np.array(self.outline.coords, dtype=np.int64),
            ObsKeys.NEAREST_BOUNDARY_POINTS.value: np.array(self.assessment.nearest_boundary_points, dtype=np.int64),
            ObsKeys.OUTSIDE_SHAPE.value: np.array(self.assessment.outside_shape_distances, dtype=np.int64),
            ObsKeys.BUILDING_AREA.value: np.array([self.assessment.current_area], dtype=np.int64),
            ObsKeys.AREA_DIFFERENCE.value: np.array([self.assessment.area_difference], dtype=np.int64),
            ObsKeys.REMAINING_MOVES.value: np.array([remaining_moves_for_obs], dtype=np.int64),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "distance": self.assessment.outside_shape_distances,
            "area_difference": self.assessment.area_difference,
            "is_solution": all(self.assessment.checklist),
            "steps": self.steps_taken
        }

    def _calculate_reward(self, terminated: bool) -> float:
        is_solution = all(self.assessment.checklist)
        if terminated and is_solution:
            remaining_steps = INTERNAL_MAX_STEPS - self.steps_taken
            early_bonus = remaining_steps * EARLY_SOLUTION_MULTIPLIER
            return SOLUTION_BONUS + early_bonus
        reward = 0.0
        total_distance_outside = sum(self.assessment.outside_shape_distances)
        reward -= total_distance_outside * DISTANCE_PENALTY_MULTIPLIER
        reward -= self.assessment.area_difference * AREA_DIFFERENCE_PENALTY_MULTIPLIER
        return reward

    def _check_termination(self) -> bool:
        solution_found = all(self.assessment.checklist)
        max_steps_reached = self.steps_taken >= INTERNAL_MAX_STEPS
        if solution_found:
            print(f"Solution found in {self.steps_taken} steps!")
        return solution_found or max_steps_reached

    def _get_info_for_render(self) -> List[str]:
        if not self.assessment or not self.plot:
            return []
        area_range_str = f"({self.plot.area_range[0]}, {self.plot.area_range[1]})"
        remaining_steps = INTERNAL_MAX_STEPS - self.steps_taken
        return [
            f"Area Range: {area_range_str} | Building Area: {self.outline.area}",
            f"Reward: {self.last_reward:.2f} | Remaining Steps: {remaining_steps}",
            f"Checklist: {self.assessment.checklist}"
        ]
