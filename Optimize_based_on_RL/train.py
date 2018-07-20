import gym
import sys
sys.path.append('C:/Users/xuwei1/Documents/baselines')
from baselines import deepq
import time

def main():
    env = gym.make('OptimizeGauss-v0')
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.5,
        exploration_final_eps=0,
        print_freq=10,
        batch_size=32,
        
    )
    print("Saving model to gauss_model.pkl")
    act.save("model/gauss.pkl")

def test():
    env = gym.make('OptimizeGauss-v0')
    print(env.reset())
    for _ in range(10):
        state, reward, done, _ = env.step(env.action_space.sample())
        value = env.gauss(state)
        print((value,state, reward, done))

if __name__ == '__main__':
    main()