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
from urban_planner.logic.reward_calculator import RewardCalculator
from urban_planner.presentation.renderer import Renderer
from urban_planner.types import ObsKeys, Stage

logger = logging.getLogger(__name__)

class UrbanPlannerEnv(gym.Env):
    """
    A Gymnasium environment for the urban planning problem.
    """
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": RENDER_FPS}

    def __init__(self, render_mode: Optional[str] = None,
                 generation_queue: Optional[multiprocessing.Queue] = None,
                 stage: int = 4):
        super().__init__()
        self.render_mode = render_mode
        self.generation_queue = generation_queue
        self.stage = stage
        if self.stage not in [1, 2, 3, 4]:
            raise ValueError(
                f"Invalid stage: {self.stage}. Must be 1, 2, 3, or 4.")
        self.plot: Optional[Plot] = None
        self.outline: Optional[Outline] = None
        self.steps_taken: int = 0
        self.assessment: Optional[OutlineAssessment] = None
        self.reward_calculator: RewardCalculator = RewardCalculator(stage)
        self.last_reward: float = 0.0
        self._renderer_class = Renderer
        self.renderer: Optional[Renderer] = None

        self.placement_solved_in_episode: bool = False
        self.area_solved_in_episode: bool = False
        self.width_solved_in_episode: bool = False

        self.observation_space = spaces.Dict({
            ObsKeys.VERTEX_VECTORS.value: spaces.Box(-WINDOW_WIDTH,
                                                     WINDOW_WIDTH, (4, 2),
                                                     dtype=np.int64),
            ObsKeys.AREA_DIFFERENCE.value: spaces.Box(-MAX_AREA, MAX_AREA, (1,),
                                                      dtype=np.int64),
            ObsKeys.WIDTH_DIFFERENCE.value: spaces.Box(-WINDOW_WIDTH, WINDOW_WIDTH, (1,),
                                                       dtype=np.int64),
            ObsKeys.ANGLE_DIFFERENCES.value: spaces.Box(0, 180, (4,),
                                                        dtype=np.int64),
            ObsKeys.REMAINING_MOVES.value: spaces.Box(0,
                                                      INTERNAL_MAX_STEPS,
                                                      (1,), dtype=np.int64),
        })

        num_vertices = 4
        move_range = 2 * MAX_MOVE_DISTANCE + 1
        self.action_space = spaces.MultiDiscrete([
            num_vertices,
            move_range,
            move_range
        ])

    def _init_renderer(self):
        """Initializes the renderer instance if it doesn't exist."""
        if self.renderer is None and self.render_mode == "human":
            self.renderer = self._renderer_class(
                WINDOW_WIDTH, WINDOW_HEIGHT, self.metadata["render_fps"]
            )
            self.renderer.init()

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
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

        self.outline = Outline(coords=[
            (0, 0), (WINDOW_WIDTH, 0),
            (WINDOW_WIDTH, WINDOW_HEIGHT), (0, WINDOW_HEIGHT)
        ])

        self.assessment = OutlineAssessment(
            self.plot.permitted_outline,
            self.outline.coords,
            self.plot.area_range,
            self.plot.width_range
        )
        self.assessment.update(self.outline.coords)

        self.placement_solved_in_episode = False
        self.area_solved_in_episode = False
        self.width_solved_in_episode = False

        self.reward_calculator.reset()
        self.reward_calculator.initial_differences[Stage.PLACEMENT] = sum(
            self.assessment.outside_shape_distances)
        self.reward_calculator.initial_differences[Stage.AREA] = abs(
            self.assessment.area_difference)
        self.reward_calculator.initial_differences[Stage.WIDTH] = self.assessment.width_difference
        self.reward_calculator.initial_differences[Stage.ANGLE] = sum(
            self.assessment.angle_differences)

        self.last_reward = 0.0
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[
        Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.steps_taken += 1

        vertex_index, dx_action, dy_action = action
        dx = dx_action - MAX_MOVE_DISTANCE
        dy = dy_action - MAX_MOVE_DISTANCE

        new_coords_list = [list(coord) for coord in self.outline.coords]

        new_coords_list[vertex_index][0] += dx
        new_coords_list[vertex_index][1] += dy
        new_coords_list[vertex_index][0] = np.clip(
            new_coords_list[vertex_index][0], 0, WINDOW_WIDTH)
        new_coords_list[vertex_index][1] = np.clip(
            new_coords_list[vertex_index][1], 0, WINDOW_HEIGHT)

        final_coords = [(int(coord[0]), int(coord[1])) for coord in
                        new_coords_list]

        self.outline.coords = final_coords
        self.outline.update_geometry()
        self.assessment.update(final_coords)

        self._update_curriculum_state()

        terminated = self._check_termination()
        reward = self._calculate_reward(terminated)
        self.last_reward = reward

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _calculate_reward(self, terminated: bool) -> float:
        solved_flags = {
            Stage.PLACEMENT: self.placement_solved_in_episode,
            Stage.AREA: self.area_solved_in_episode,
            Stage.WIDTH: self.width_solved_in_episode
        }
        return self.reward_calculator.calculate(self.assessment, solved_flags,
                                                terminated)

    def _update_curriculum_state(self):
        """Manages the state flags for the curriculum learning stages."""
        if self.stage >= 2 and self.assessment.checklist[
            Stage.PLACEMENT] and not self.placement_solved_in_episode:
            self.placement_solved_in_episode = True

        if self.stage >= 3 and self.assessment.checklist[
            Stage.AREA] and not self.area_solved_in_episode:
            self.area_solved_in_episode = True

        if self.stage >= 4 and self.assessment.checklist[
            Stage.WIDTH] and not self.width_solved_in_episode:
            self.width_solved_in_episode = True

        if not self.assessment.checklist[Stage.PLACEMENT]:
            self.placement_solved_in_episode = False
            self.area_solved_in_episode = False
            self.width_solved_in_episode = False
        elif not self.assessment.checklist[Stage.AREA]:
            self.area_solved_in_episode = False
            self.width_solved_in_episode = False
        elif not self.assessment.checklist[Stage.WIDTH]:
            self.width_solved_in_episode = False

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
        """
        Constructs the observation dictionary for the agent.
        The agent always sees all available information.
        """
        agent_coords = np.array(self.outline.coords, dtype=np.int64)
        target_coords = np.array(self.assessment.nearest_boundary_points,
                                 dtype=np.int64)

        obs = {
            ObsKeys.VERTEX_VECTORS.value: target_coords - agent_coords,
            ObsKeys.AREA_DIFFERENCE.value: np.array(
                [self.assessment.area_difference], dtype=np.int64),
            ObsKeys.WIDTH_DIFFERENCE.value: np.array(
                [self.assessment.width_difference], dtype=np.int64),
            ObsKeys.ANGLE_DIFFERENCES.value: np.array(
                self.assessment.angle_differences, dtype=np.int64),
            ObsKeys.REMAINING_MOVES.value: np.array(
                [max(0, INTERNAL_MAX_STEPS - self.steps_taken)],
                dtype=np.int64),
        }

        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "distance": sum(self.assessment.outside_shape_distances),
            "area_difference": self.assessment.area_difference,
            "is_solution": all(self.assessment.checklist.values()),
            "steps": self.steps_taken
        }

    def _check_termination(self) -> bool:
        solution_found = all(
            self.assessment.checklist[Stage(i)] for i in range(self.stage))
        if solution_found:
            logger.info(f"Solution found in {self.steps_taken} steps!")
        max_steps_reached = self.steps_taken >= INTERNAL_MAX_STEPS

        return solution_found or max_steps_reached

    def _get_info_for_render(self) -> List[str]:
        if not self.assessment or not self.plot:
            return []
        area_range_str = f"({self.plot.area_range[0]}, {self.plot.area_range[1]})"
        width_range_str = f"({self.plot.width_range[0]}-{self.plot.width_range[1]})"
        angle_target_str = f"{TARGET_ANGLE_DEGREES}° (±{ANGLE_TOLERANCE_DEGREES}°)"

        angles_str = ", ".join(
            [f"{int(a)}" for a in self.assessment.current_angles])

        remaining_steps = INTERNAL_MAX_STEPS - self.steps_taken
        checklist_str = (
            f"Placement: {self.assessment.checklist[Stage.PLACEMENT]}, "
            f"Area: {self.assessment.checklist[Stage.AREA]}, "
            f"Width: {self.assessment.checklist[Stage.WIDTH]}, "
            f"Angles: {self.assessment.checklist[Stage.ANGLE]}")

        return [
            f"Area: {self.outline.area} (Target: {area_range_str})",
            f"Width: {self.assessment.current_width:.1f} (Target: {width_range_str})",
            f"Angles: [{angles_str}] (Target: {angle_target_str})",
            f"Reward: {self.last_reward:.2f} | Remaining Steps: {remaining_steps}",
            f"Checklist: {checklist_str}"
        ]