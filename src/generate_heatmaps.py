import os
import numpy as np
from shutil import rmtree
import scipy.io
from glob import glob
from utils.calculate_3Dheatmap import calculate_3Dheatmap


dim_heatmap = 64
sigma = 0.2
input_path = '../dataset/Human3.6m/3d_poses/'
output_path = '../dataset/Human3.6m/heatmaps/'
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


if __name__ == '__main__':
	if os.path.exists(output_path):
		rmtree(output_path)

	os.makedirs(output_path)
	for s in subs:
		os.makedirs(os.path.join(output_path, s))

	print('Converting...')

	for idx, s in enumerate(subs):
		print('generating', s + ':')
		acts = os.listdir(os.path.join(input_path, s))
		for act in acts:
			print('generating', s + '-' + act)
			os.makedirs(os.path.join(output_path, s, act))
			file_list = glob(os.path.join(input_path, s, act) + '/*.csv')
			for f in file_list:
				filename = f.split('/')[-1][:-4]
				pts = np.genfromtxt(f, delimiter = ' ')
				heatmap = calculate_3Dheatmap(pts, dim_heatmap, sigma)
				filename = output_path + s +'/' + act + '/' + filename + '.mat'
				scipy.io.savemat(filename, {'heatmap': heatmap})

	print('Done!')

