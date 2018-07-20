import gym
import sys
sys.path.append('C:/Users/xuwei1/Documents/baselines')

from baselines import deepq
import time

def test0():
    env = gym.make("OptimizeGauss-v0")
    act = deepq.load("model/gauss.pkl")
    episode = 0
    for i in range(1000):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
            # print(episode_rew)

    print(env.gauss(obs))

if __name__ == '__main__':
    test0()