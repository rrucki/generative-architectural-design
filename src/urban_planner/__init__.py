from gymnasium.envs.registration import register

register(
     id="UrbanPlanner-v0",
     entry_point="urban_planner.envs.urban_planner:UrbanPlannerEnv",
)
