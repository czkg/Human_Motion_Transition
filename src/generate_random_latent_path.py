import os
import numpy as np
import scipy.io
from glob import glob
import json
from shutil import rmtree
from random import randrange
from utils.utils import slerp
import time

"""
This module is used to generate paths given random start and end frames
"""

input_path = '../dataset/Human3.6m/latent_nth/'
input_unwrap_path = '../dataset/Human3.6m/latent_nth_unwrap/'
output_path = '../dataset/Human3.6m/latent_path_random/'

subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
steps = 10

"""
source: pose under specific subject/action
target: any pose sampled from the whole dataset
thus we need to go through every action for each subject
"""

def interpolate_poses_euclidean(start, end):
	sequence = np.array([slerp(start, end, t) for t in np.linspace(0, 1, steps)])
	return sequence

if __name__ == '__main__':
	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)

	print('Generating...')

	target_files = glob(input_unwrap_path + '*.mat')
	target_sample_size = len(target_files)
	s_sample_size = 300
	t_sample_size = 10 * 300

	for s in subs:
		print('generating', s + ':')
		acts = os.listdir(os.path.join(input_path, s))
		for a in acts:
			# default sample size
			source_files = glob(os.path.join(input_path, s, a) + '/*.mat')
			source_sample_size = min(s_sample_size, len(source_files))
			for i in range(t_sample_size):
				source_sample_id = randrange(source_sample_size)
				start_file = source_files[source_sample_id]
				start_latent = scipy.io.loadmat(start_file)['latent'][0]
				target_sample_id = randrange(target_sample_size)
				end_file = target_files[target_sample_id]
				end_latent = scipy.io.loadmat(end_file)['latent'][0]
				seq = interpolate_poses_euclidean(start_latent, end_latent)
				save_name = s+'_'+a+'_'+start_file[start_file.rfind('/')+1:-4]+'_to_'+end_file[end_file.rfind('/')+1:-4]+'.mat'
				save_path = os.path.join(output_path, save_name)
				if os.path.exists(save_path):
					current_time = time.asctime()
					save_path = save_path[:-4] + '_' + current_time + '.mat'
				scipy.io.savemat(save_path, {'data': seq})



	

