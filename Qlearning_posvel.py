import gym
import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

show = False
iters=[]

# env = gym.make("MountainCar-v0")

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')
epsilon=0.01
n_vels=20
n_poss=20
epochs=500
# our states limitators (-0.07, -0.06, ... , 0.007)
vels = np.linspace(env.observation_space.low[1], env.observation_space.high[1], n_vels)
poss=np.linspace(env.observation_space.low[0], env.observation_space.high[0], n_poss)
# our actions
actions = np.array((0, 1, 2))
# rewards and transition defined by environment
learn_rate = 0.1
disc_fact = 0.9

# set q-values
# q_val = np.random.rand(n_vels,n_poss, 3)  # state x action
q_val=np.zeros([n_vels,n_poss, 3])
for _ in range(epochs):
    # reset simulation
    observation = env.reset()

    state = np.digitize(observation[1], vels), np.digitize(observation[0], poss) # which state are we based on the observation

    done = False
    timesteps = 0

    while not done:
        # if show: env.render()
        # Choose an action
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(env.action_space.n)
        else:

            # choose action based on policy (max q)
            action = np.argmax(q_val[state])
        if show: print(q_val[state])
        if show: print(action)

        # new state
        observation, reward, done, info = env.step(action)
        # /\state = np.digitize(observation[1], vels), np.digitize(observation[0],
        #                                                        poss)  # which state are we based on the observation

        new_state = np.digitize(observation[1], vels), np.digitize(observation[0],
                                                               poss) # which state are we based on the observation
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
    iters.append(timesteps)
    # print("Episode finished after ", timesteps, "timesteps.")
optimal_policy = np.argmax(q_val, axis=2)

print('3 examples')
#see 3 examples
for i in range(0,3):
    observation = env.reset()
    done = False
    timesteps=0
    while not done:

            env.render()
            time.sleep(0.0015)
            timesteps+=1
            # print(timesteps)

            state = np.digitize(observation[1], vels), np.digitize(observation[0],
                                                                   poss)  # which state are we based on the observation

            action = np.argmax(q_val[state])
            if timesteps<500:
                observation, reward, done, info = env.step(action)

            else:
                observation = env.reset()
                timesteps = 0

    print('timesteps: ' + str(timesteps))
env.close()


#VISUALIZATIONS


# state_values = pd.DataFrame(columns=np.round(poss,2), index=np.round(vels,2),data=np.argmax(q_val, axis=2))
# plt.figure(figsize=(10, 7))
# sns.heatmap(state_values, linewidths=0, cbar_kws={'label': 'Actions'},cmap="YlGnBu")
# fig1=plt.gcf()
# plt.xlabel("Position")
# plt.ylabel("Velocity")
# plt.show()
# fig1.savefig('myplot3')
# mymax=4000
# for ind,val in enumerate(iters):
#     if val>mymax:
#         iters[ind]=mymax

# plt.plot(iters)
# plt.yticks(np.arange(min(iters), max(iters)+1, 500.0))

# fig1=plt.gcf()
# plt.xlabel("Episodes")
# plt.ylabel("Iterations to goal")
# plt.show()
# fig1.savefig('Iters2 ep500,pos10,vels20,gama090,alpha01,epsi001.png')
