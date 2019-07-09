import os
import numpy as np
from numpy import genfromtxt
import argparse

g_path = './gt/'
i_path = './gan/poses/'
groundtruth = ['216.csv', '221.csv', '226.csv', '231.csv', '236.csv', '241.csv', '246.csv', '251.csv', '256.csv', '261.csv']
inputs = ['1.csv', '2.csv', '3.csv', '4.csv', '5.csv', '6.csv', '7.csv', '8.csv', '9.csv', '10.csv']

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--al', type=str, help='the algorithm is using')
	args = vars(parser.parse_args())

	al = args['al']

	g_paths = [os.path.join(g_path, g) for g in groundtruth]
	g_poses = [genfromtxt(p, delimiter=' ') for p in g_paths]
	g = [np.reshape(gp, (17*3,)) for gp in g_poses]
	g = np.concatenate(g).ravel()

	i_paths = [os.path.join(i_path, i) for i in inputs]
	i_poses = [genfromtxt(p, delimiter=' ') for p in i_paths]
	i = [np.reshape(ip, (17*3),) for ip in i_poses]
	i = np.concatenate(i).ravel()

	error = ((g-i)**2).mean()
	print('the error is ', error)
	file = open(al + '.txt','w')
	file.write(str(error))
