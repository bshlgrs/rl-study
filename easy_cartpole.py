import gym
import numpy as np


def random_policy():
    return np.random.normal(0, 1, 4)


def get_score(env, policy, render=False):
    NUM_TRIALS = 15
    total_reward = 0.0

    for i in range(NUM_TRIALS):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = int(np.dot(obs, policy) < 0)
            obs, r, done, _ = env.step(action)
            total_reward += r

    return total_reward / NUM_TRIALS


env = gym.make('CartPole-v0')
best_policy = None
best_score = 0

for i in range(500):
    policy = random_policy()
    score = get_score(env, policy)

    if score > best_score:
        best_score = score
        best_policy = policy

get_score(env, best_policy, True)
print(best_policy)
