import gym
import numpy as np
import matplotlib.pyplot as htmp

show = False

# env = gym.make("MountainCar-v0")

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')

# our states limitators (-0.07, -0.06, ... , 0.007)
vels = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 15)
# our actions
actions = np.array((0, 1, 2))
# rewards and transition defined by environment
learn_rate = 0.1
disc_fact = 1

# initialise q-values
q_val = np.random.rand(15, 3)  # state x action

timesteps = 10000

while timesteps>100:
    # reset simulation
    timesteps = 0
    observation = env.reset()
    state = np.digitize(observation[1], vels)  # which state are we based on the observation
    done = False

    while not done:
        if show: env.render()

        # choose action based on policy (max q)
        action = np.argmax(q_val[state])
        if show: print(q_val[state])
        if show: print(action)

        # new state
        observation, reward, done, info = env.step(action)
        new_state = np.digitize(observation[1], vels)  # which state are we based on the observation
        timesteps+=1

        # update q-value
        q_val[state][action] = (1-learn_rate)* q_val[state][action] + \
                learn_rate * (reward + disc_fact * np.max(q_val[new_state]))

        # update state
        state = new_state

        # if show: print(observation)
        # if show: print(reward)
        # if show: print(done)
        # if show: print(info)

    print("Episode finished after ", timesteps, "timesteps.")


# SHOWS STATE-VALUES
which_state = np.argmax(q_val, axis=1)
htmp.imshow((which_state, which_state), cmap='hot', interpolation='nearest')
htmp.colorbar()
htmp.show()

htmp.imshow((np.max(q_val, axis=1), np.max(q_val, axis=1)), cmap='hot', interpolation='nearest')
htmp.colorbar()
htmp.show()
