from gym.envs.registration import register

register(
    id='AntGoal-v0',
    entry_point='dbc.envs.ant:AntGoalEnv',
    max_episode_steps=50,
    )
