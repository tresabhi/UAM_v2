from multi_uav_uam import UamUavEnvPZ
import matplotlib.pyplot as plt

parallel_env = UamUavEnvPZ("Austin, Texas, USA", 8, 5, sleep_time=0.01)
observations, infos = parallel_env.reset(seed=42)


# To have interactive plotting active
plt.ion()
fig, ax = parallel_env.render_init()


while parallel_env.agents:
    # this is where you would insert your policy
    actions = {
        agent: parallel_env.action_space(agent).sample()
        for agent in parallel_env.agents
    }
    parallel_env.render(fig, ax)
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
