# OpenAI Cartpole implementations.
# By Tom Jacobs
#
# Two methods:
# 1. Random: It just tries random parameters, and picks the first one that gets a 200 score.
# 2. Mutation: It starts with random parameters, and adds a 50% mutation on the best parameters found, each time.
#
# Runs on Python 3.
# Originally based on https://github.com/kvfrans/openai-cartpole
# You can easily submit it to the OpenAI Gym scoreboard by entering your OpenAI key and switching on submit below.

# Method to use?
method = 2

# Submit it?
submit = True
api_key = ''

import gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, parameters):
    observation = env.reset()

    # Run 200 steps and see what our total reward is
    total_reward = 0
    for t in range(200):

        # Show us what's going on. Comment this line out to run super fast. The monitor will still render some random ones though for video recording, even if render is off.
#        env.render()

        # Pick action
        action = 0 if np.matmul(parameters, observation) < 0 else 1

        # Step
        observation, reward, done, info = env.step(action)
        total_reward += reward

        # Done?
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    return total_reward

def train(submit):
    # Start cartpole
    env = gym.make('CartPole-v0')
    if submit:
        env = gym.wrappers.Monitor(env, 'cartpole', force=True)

    # Keep results
    results = []
    counter = 0

    # For method 1. Run lots of episodes with random params, and find the best_parameters.
    best_parameters = None
    best_reward = 0

    # Additional for method 2
    episodes_per_update = 5
    mutation_amount = 0.5
    best_parameters = np.random.rand(4) * 2 - 1

    # Run
    for t in range(100):
        counter += 1

        # Pick random parameters and run
        if method == 1:
            new_parameters = np.random.rand(4) * 2 - 1
            reward = run_episode(env, new_parameters)

        # Method 2 is to use the best parameters, with 10% random mutation
        elif method == 2:
            new_parameters = best_parameters + (np.random.rand(4) * 2 - 1) * mutation_amount
            reward = 0
            for e in range(episodes_per_update):
                 run = run_episode(env, new_parameters)
                 reward += run
            reward /= episodes_per_update

        # One more result
        results.append(reward)

        # Did this one do better?
        if reward > best_reward:
            best_reward = reward
            best_parameters = new_parameters
            print("Better parameters found.")

            # And did we win the world?
            if reward == 200:
                print("Win! Episode {}".format(t))
                break # Can't do better than 200 reward, so quit trying

    # Run 100 runs with the best found params
    print("Found best_parameters, running 100 more episodes with them.")
    for t in range(100):
        reward = run_episode(env, best_parameters)
        results.append(reward)
        print( "Episode " + str(t) )

    # Done
    return results

# Run
results = train(submit=submit)

print(results)
