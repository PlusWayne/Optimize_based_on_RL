import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


mu_1 = 1
mu_2 = -1
sigma_1 = 1
sigma_2 = 1

def gauss(x, y, mu_1=1, sigma_1=3, mu_2=1, sigma_2=3):
	z = (1/(np.sqrt(2*np.pi)*sigma_1)) * np.exp(-np.square(x - mu_1) / (2 * np.square(sigma_1))) \
		+ (1/(np.sqrt(2*np.pi)*sigma_2)) * np.exp(-np.square(y - mu_2) / (2 * np.square(sigma_2)))

	return z

def render(fun,x_low = -10, x_high = 10, y_low = -10, y_high = 10, space = 0.01):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x_range = np.arange(x_low,x_high+space,space)
	y_range = np.arange(y_low,y_high+space,space)
	x_range, y_range = np.meshgrid(x_range,y_range)
	z_range = np.array(fun(x_range,y_range))
	ax.plot_surface(x_range,y_range,z_range, cmap=plt.cm.coolwarm)
	plt.show()

def main():
	# a = gauss(np.array([1,1]),np.array([-1,-1]))
	render(gauss)

if __name__ == '__main__':
	main()