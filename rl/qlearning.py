import random
import gym
import numpy as np

hyperparams = {'lr': 0.8,
               'discount': 0.9,
               'epochs': 15000,
               'eps':{'min': 0.01,
                      'max': 1.0,
                      'decay': 0.96
                      }
               }

def explore(eps):
    return random.uniform(0,1) < eps

verbose = False
env = gym.make('Taxi-v3').env
if verbose:
    env.render()

q_table = np.zeros((env.observation_space.n, env.action_space.n))
epsilon = hyperparams['eps']['max']
rewards = []

for epoch in range(1, hyperparams['epochs']+1):
    done = False
    obs = env.reset()

    reward = 0
    total_rewards = 0

    while not done:
        # select exploration or exploitation
        if explore(epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[obs])

        # make the step on the selected action
        next_obs, reward, done, _ = env.step(action)

        # return the maximum value from the next observation
        next_max_obs = np.max(q_table[next_obs])

        # bellman equation
        q_table[obs,action] = (q_table[obs,action] + hyperparams['lr'] *
                            (reward + hyperparams['discount'] * next_max_obs - q_table[obs,action]))

        obs = next_obs
        total_rewards += reward

    # reduce the exploration rate by the decay factor with every epoch
    epsilon = max(hyperparams['eps']['min'], epsilon * hyperparams['eps']['decay'])
    rewards.append(total_rewards)

if verbose:
    print("Q table: {}".format(q_table))

def run(episodes):
    for episode in range(1, episodes+1):
        obs = env.reset()
        done = False
        print("-"*20)
        print("Episode {}".format(episode))
        print("-"*20)

        while not done:
            action = np.argmax(q_table[obs,:])

            next_obs, _, done, _ = env.step(action)
            env.render()

            obs = next_obs

run(2)
print("Total rewards: {}".format(sum(rewards)))
env.close()
