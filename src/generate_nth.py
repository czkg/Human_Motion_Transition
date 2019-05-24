import numpy as np
import os
from shutil import rmtree
from shutil import copyfile
from glob import glob

step = 5
input_x_path = '../dataset/Human3.6m/heatmaps/'
input_z_path = '../dataset/Human3.6m/latent'
output_x_path = '../dataset/Human3.6m/heatmaps_nth/'
output_z_path = '../dataset/Human3.6m/latent_nth/'
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

if __name__ == '__main__':
	if os.path.exists(output_x_path):
		rmtree(output_x_path)
	if os.path.exists(output_z_path):
		rmtree(output_z_path)

	os.makedirs(output_x_path)
	os.makedirs(output_z_path)

	for s in subs:
		os.makedirs(os.path.join(output_x_path, s))
		os.makedirs(os.path.join(output_z_path, s))

	for s in subs:
		print('generating', s + ':')
		acts = os.listdir(os.path.join(input_x_path, s))
		for act in acts:
			print('generating', s + '-' +  act)
			os.makedirs(os.path.join(output_x_path, s, act))
			os.makedirs(os.path.join(output_z_path, s, act))
			file_list_x = glob(os.path.join(input_x_path, s, act) + '/*.mat')
			file_list_z = glob(os.path.join(input_z_path, s, act) + '/*.mat')

			id_list = [f.split('/')[-1][:-4] for f in file_list_x]
			id_list = list(map(int, id_list))
			id_list.sort()
			# get key frames
			key_list = id_list[0::step]
			key_list = list(map(str, key_list))
			file_list_x = [os.path.join(input_x_path, s, act) + '/' + k + '.mat' for k in key_list]
			new_file_list_x = [os.path.join(output_x_path, s, act) + '/' + k + '.mat' for k in key_list]
			file_list_z = [os.path.join(input_z_path, s, act) + '/' + k + '.mat' for k in key_list]
			new_file_list_z = [os.path.join(output_z_path, s, act) + '/' + k + '.mat' for k in key_list]
			for idx, _ in enumerate(file_list_x):
				copyfile(file_list_x[idx], new_file_list_x[idx])
			for idx, _ in enumerate(file_list_z):
				copyfile(file_list_z[idx], new_file_list_z[idx])

	print('Done!')


			

