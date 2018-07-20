import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class optimize_gauss(gym.Env):
	def __init__(self):
		# gauss distribution parameters
		self.mu_1 = 1
		self.mu_2 = 1
		self.sigma_1 = 3
		self.sigma_2 = 3

		self.viewer = None
		self.state = None
		# record current value
		self.cur_value = None
		self.converge = 1e-5
		self.count = 0 # count times that satisfy convergence
		self.max_count = 5 # max times that satisfy convergence

		# step size
		self.step_size = 0.01

		# (x,y) range  need to be reconsidered
		self.range = 100
		# action space [-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]
		self.action_space = spaces.Box(low = np.array([-1,-1]), high = np.array([1,1]), dtype = np.int)

		# obervation_space from [-range,-range] to [range, range]
		high = np.array([self.range,self.range])
		low = np.array([-self.range,-self.range])
		self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

		# self.viewer = None

		# self.state init
		self.reset()
		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.state = self.observation_space.sample()
		self.cur_value = self.gauss(self.state)
		return self.state

	def step(self,action):
		self.state = self.state + self.step_size * action
		value = self.gauss(self.state)
		reward = value - self.cur_value
		self.cur_value = value
		if reward<=self.converge:
			self.count += 1
			if self.count > self.max_count:
				done = True
		else:
			done = Flase
		return self.state, reward, done, {}

	def gauss(self, state):
		x, y = self.state
		self.cur_value = (1/(np.sqrt(2*np.pi)*self.sigma_1)) * np.exp(-np.square(x - self.mu_1) / (2 * np.square(self.sigma_1))) \
			+ (1/(np.sqrt(2*np.pi)*self.sigma_2)) * np.exp(-np.square(y - self.mu_2) / (2 * np.square(self.sigma_2)))
		return self.cur_value


def test():
	op = optimize_gauss()
	print(op.reset())

if __name__ == '__main__':
	test()