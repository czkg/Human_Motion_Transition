import os
import scipy.io
from glob import glob
from shutil import rmtree
import hdf5storage

x_path = '../dataset/Human3.6m/heatmaps_nth/'
z_path = '../dataset/Human3.6m/latent_nth/'
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
out_path = '../dataset/Human3.6m/x_and_z/'

if __name__ == '__main__':
	if os.path.exists(out_path):
		rmtree(out_path)

	os.makedirs(out_path)
	os.makedirs(os.path.join(out_path, 'x'))
	os.makedirs(os.path.join(out_path, 'z'))

	print('Processing...')
	#read x and z
	print('read x and z ...')
	for s in subs:
		x = []
		z = []
		print('processing', s + ':')
		acts = os.listdir(os.path.join(x_path, s))
		for act in acts:
			print('processing', s + '-' + act)
			file_list_x = glob(os.path.join(x_path, s, act) + '/*.mat')
			file_list_z = glob(os.path.join(z_path, s, act) + '/*.mat')
			for f_x in file_list_x:
				data = scipy.io.loadmat(f_x)['heatmap'][0]
				x.append(data)
			for f_z in file_list_z:
				data = scipy.io.loadmat(f_z)['latent'][0]
				z.append(data)
		#save x and z
		print('save ', s)
		hdf5storage.savemat(out_path + 'x/' + s + '.mat', {'x': x})
		hdf5storage.savemat(out_path + 'z/' + s + '.mat', {'z': z})

	print('Done!')