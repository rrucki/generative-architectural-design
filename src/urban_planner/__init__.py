from gymnasium.envs.registration import register

register(
     id="UrbanPlanner-v0",
     entry_point="src.urban_planner.envs.urban_planner:UrbanPlannerEnv",
)
