from typing import List, Dict

from urban_planner.config import *

from urban_planner.logic.assessment import OutlineAssessment
from urban_planner.types import Stage


class RewardCalculator:
    """
    A stateful class responsible for calculating rewards based on environment state.
    This class implements a reward-shaping strategy for curriculum learning.
    It calculates rewards for all goals (placement, area, etc.) in every step,
    but weighs them based on the current training stage. Goals for the current
    and past stages receive full weight, while future goals receive a smaller,
    non-zero weight to provide a "hint" to the agent.
    """

    def __init__(self, stage: int):
        self.stage = stage
        self.last_proportional_rewards: Dict[Stage, float] = {}
        self.last_checklist: Dict[Stage, bool] = {}
        self.initial_differences: Dict[Stage, float] = {}
        self.last_solved_flags: Dict[Stage, bool] = {}
        self.last_vertices_in_place: List[bool] = []
        self.last_angles_ok: List[bool] = []
        self.reset()

    def reset(self):
        """Resets the calculator's state for a new episode."""
        self.last_proportional_rewards = {s: 0.0 for s in Stage}
        self.last_checklist = {s: False for s in Stage}
        self.last_solved_flags = {s: False for s in Stage}
        self.initial_differences = {s: 0.0 for s in Stage}
        self.last_vertices_in_place = [False] * 4
        self.last_angles_ok = [False] * 4

    def calculate(self, assessment: OutlineAssessment,
                  solved_flags: Dict[Stage, bool], terminated: bool) -> float:
        """
        Calculates the total weighted reward for the current step based on the training stage.

        Args:
            assessment: The current assessment object with all metrics.
            solved_flags: (No longer used directly for reward calculation in this new approach).
            terminated: A boolean indicating if the episode has terminated.

        Returns:
            The total weighted reward for the step.
        """
        rewards = {
            Stage.PLACEMENT: self._calculate_placement_reward(assessment),
            Stage.AREA: self._calculate_area_reward(assessment),
            Stage.WIDTH: self._calculate_width_reward(assessment),
            Stage.ANGLE: self._calculate_angle_reward(assessment),
        }

        total_reward = 0.0
        for objective_stage, reward_value in rewards.items():
            if objective_stage.value <= self.stage - 1:
                total_reward += reward_value
            else:
                total_reward += reward_value * LOW_PRIORITY_REWARD_FACTOR

        is_stage_solution = all(
            assessment.checklist[Stage(i)] for i in range(self.stage))
        if terminated and is_stage_solution:
            total_reward += SOLUTION_BONUS * self.stage

        self.last_checklist = assessment.checklist.copy()
        self.last_vertices_in_place = [
            dist == 0 for dist in assessment.outside_shape_distances
        ]
        self.last_angles_ok = [
            diff == 0 for diff in assessment.angle_differences
        ]

        return total_reward

    def _calculate_placement_reward(self,
                                    assessment: OutlineAssessment) -> float:
        """Calculates reward for placement (Stage 1)."""
        reward = 0.0

        if self.initial_differences[Stage.PLACEMENT] > 0:
            current_total_distance = sum(assessment.outside_shape_distances)
            progress_made = self.initial_differences[
                                Stage.PLACEMENT] - current_total_distance
            progress_ratio = progress_made / (self.initial_differences[
                Stage.PLACEMENT] + REWARD_EPSILON)
            proportional_reward = progress_ratio * PROGRESS_REWARD_POOL
            reward += (proportional_reward - self.last_proportional_rewards[
                Stage.PLACEMENT])
            self.last_proportional_rewards[
                Stage.PLACEMENT] = proportional_reward

        current_vertices_in_place = [dist == 0 for dist in
                                     assessment.outside_shape_distances]
        for i in range(len(current_vertices_in_place)):
            if current_vertices_in_place[i] and not \
            self.last_vertices_in_place[i]:
                reward += PLACEMENT_BONUS_PER_VERTEX
            elif not current_vertices_in_place[i] and \
                    self.last_vertices_in_place[i]:
                reward -= PLACEMENT_BONUS_PER_VERTEX

        return reward

    def _calculate_area_reward(self, assessment: OutlineAssessment) -> float:
        """Calculates reward for area (Stage 2)."""
        reward = 0.0

        if self.initial_differences[Stage.AREA] > 0:
            current_abs_diff = abs(assessment.area_difference)
            progress_made = self.initial_differences[
                                Stage.AREA] - current_abs_diff
            progress_ratio = progress_made / (self.initial_differences[
                Stage.AREA] + REWARD_EPSILON)
            proportional_reward = progress_ratio * AREA_PROGRESS_REWARD_POOL
            reward += (proportional_reward -
                       self.last_proportional_rewards[Stage.AREA])
            self.last_proportional_rewards[Stage.AREA] = proportional_reward

        is_area_solved = assessment.checklist[Stage.AREA]
        was_area_solved = self.last_checklist.get(Stage.AREA, False)
        if is_area_solved and not was_area_solved:
            reward += AREA_SOLUTION_BONUS
        elif not is_area_solved and was_area_solved:
            reward -= AREA_SOLUTION_BONUS

        return reward

    def _calculate_width_reward(self, assessment: OutlineAssessment) -> float:
        """Calculates reward for width (Stage 3)."""
        reward = 0.0

        if self.initial_differences[Stage.WIDTH] > 0:
            progress_made = self.initial_differences[
                                Stage.WIDTH] - assessment.width_difference
            progress_ratio = progress_made / (self.initial_differences[
                Stage.WIDTH] + REWARD_EPSILON)
            proportional_reward = progress_ratio * WIDTH_PROGRESS_REWARD_POOL
            reward += (proportional_reward -
                       self.last_proportional_rewards[Stage.WIDTH])
            self.last_proportional_rewards[Stage.WIDTH] = proportional_reward

        is_width_solved = assessment.checklist[Stage.WIDTH]
        was_width_solved = self.last_checklist.get(Stage.WIDTH, False)
        if is_width_solved and not was_width_solved:
            reward += WIDTH_SOLUTION_BONUS
        elif not is_width_solved and was_width_solved:
            reward -= WIDTH_SOLUTION_BONUS

        return reward

    def _calculate_angle_reward(self, assessment: OutlineAssessment) -> float:
        """Calculates reward for angles (Stage 4)."""
        reward = 0.0

        if self.initial_differences[Stage.ANGLE] > 0:
            current_total_angle_diff = sum(assessment.angle_differences)
            progress_made = self.initial_differences[
                                Stage.ANGLE] - current_total_angle_diff
            progress_ratio = progress_made / (self.initial_differences[
                Stage.ANGLE] + REWARD_EPSILON)
            proportional_reward = progress_ratio * ANGLE_PROGRESS_REWARD_POOL
            reward += (proportional_reward -
                       self.last_proportional_rewards[Stage.ANGLE])
            self.last_proportional_rewards[Stage.ANGLE] = proportional_reward

        current_angles_ok = [diff == 0 for diff in
                             assessment.angle_differences]
        for i in range(len(current_angles_ok)):
            if current_angles_ok[i] and not self.last_angles_ok[i]:
                reward += ANGLE_SOLUTION_BONUS_PER_CORNER
            elif not current_angles_ok[i] and self.last_angles_ok[i]:
                reward -= ANGLE_SOLUTION_BONUS_PER_CORNER

        return reward
