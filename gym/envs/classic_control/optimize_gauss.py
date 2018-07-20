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
		self.converge = 1e-7
		self.count = 0 # count times that satisfy convergence
		self.max_count = 5 # max times that satisfy convergence

		# step size , decay?
		self.step_size = 0.1

		# (x,y) range  need to be reconsidered
		self.range = 5
		# action space [-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]
		# self.action_space = spaces.Box(low = np.array([-1,-1]), high = np.array([1,1]), dtype = np.int)
		self.action_space = spaces.Discrete(9)
		self.action_dict = {0 : np.array([-1,-1]),
							1 : np.array([-1,0]),
							2 : np.array([-1,1]),
							3 : np.array([0,-1]),
							4 : np.array([0,0]),
							5 : np.array([0,1]),
							6 : np.array([1,-1]),
							7 : np.array([1,0]),
							8 : np.array([1,1]),}
		# obervation_space from [-range,-range] to [range, range]
		high = np.array([self.range,self.range])
		low = np.array([-self.range,-self.range])
		self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)

		# self.viewer = None

		# self.state init
		self.seed()
		self.reset()


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		self.state = self.range * np.random.rand(2)
		self.cur_value = self.gauss(self.state)
		self.count = 0
		return self.state

	def step(self,action):
		action = self.action_dict[action]
		state = self.state + self.step_size * action
		value = self.gauss(state)
		reward = value - self.cur_value
		self.state = state
		self.cur_value = value
		if reward<=self.converge:
			self.count += 1
			done = False
			if self.count > self.max_count:
				done = True
		else:
			done = False
		return self.state, 100 * reward, done, {}

	def gauss(self, state):
		x, y = state
		cur_value = (1/(np.sqrt(2*np.pi)*self.sigma_1)) * np.exp(-np.square(x - self.mu_1) / (2 * np.square(self.sigma_1))) \
			+ (1/(np.sqrt(2*np.pi)*self.sigma_2)) * np.exp(-np.square(y - self.mu_2) / (2 * np.square(self.sigma_2)))
		return cur_value


def test():
	op = optimize_gauss()
	print(op.reset())

if __name__ == '__main__':
	test()