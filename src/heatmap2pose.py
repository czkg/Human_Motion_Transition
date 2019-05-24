import os
import numpy as np
import argparse
from glob import glob
import tqdm


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, help='the input path')
	parser.add_argument('--output_path', type=str, help='the output path')
	parser.add_argument('--dim_heatmap', type=int, default=64, help='the dimension of the heatmap')
	parser.add_argument('--n_joints', type=int, default=17, delp='the number of joints')
	args = vars(parser.parse_args())

	input_path = args['input_path']
	output_path = args['output_path']
	dim_heatmap = args['dim_heatmap']
	n_joints = args['n_joints']

	dim_xy = dim_heatmap ** 2
	dim = size_xy + dim_heatmap

	l = np.linspace(-1, 1, dim_heatmap)
	file_list = glob(input_path + '/*.mat')
	num = len(file_list)
	for i in tqdm.trange(num):
		f = file_list[i]
		filename = f.split('/')[-1]
		data = scipy.io.loadmat(f)['heatmap'][0]
		data = np.split(data, n_joints)
		data_xy = [d[:size_xy] for d in data]
		data_z = [d[size_xy:] for d in data]

		xy_max = [np.argmax(dxy) for dxy in data_xy]
		x_max = [xym // dim_heatmap for xym in xy_max]
		y_max = [xym % dim_heatmap for xym in xy_max]
		z_max = [np.argmax(dz) for dz in data_z]
		x = [l[xm] for xm in x_max]
		y = [l[ym] for ym in y_max]
		z = [l[zm] for zm in z_max]

		pose = np.stack((x, y, z), axis = 0)
		pose = pose.T

		name = os.path.join(output_path, filename)
		scipy.io.savemat(name, {'pose': pose})
