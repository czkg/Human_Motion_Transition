import numpy as np
import os
from shutil import rmtree
from glob import glob
import json

"""
This module is used to generate a json file including all the paths composed of pose id
"""

# the length of the path is set to 15
length = 15
step = 5
num_act = 30
num_sub = 7

input_path = '../dataset/Human3.6m/3d_poses/'
output_file = '../dataset/Human3.6m/paths.json'
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
sub_names = [input_path + s for s in subs]

if __name__ == '__main__':
	if not os.path.exists(input_path):
		raise Exception('Input not exists!')

	if os.path.isfile(output_file):
		os.remove(output_file)

	dictionary = {}
	for idx, s in enumerate(subs):
		print('generating', s + ':')
		act_list = glob(input_path + s + '/*/')
		dictionary[s] = {}
		for act in act_list:
			action = act.split('/')[-2]
			print('generating', s + '-' + action)
			file_list = glob(input_path + s + '/' + action + '/*.csv')
			num = len(file_list)
			file_list = [f.split('/')[-1][:-4] for f in file_list]
			file_list = list(map(int, file_list))
			file_list.sort()
			# get key frames
			key_list = file_list[0::step]

			# length = 15 here
			num = len(key_list) - length + 1
			paths = np.empty((num, length), dtype = int)
			for i in range(num):
				paths[i] = key_list[i: i + length]

			dictionary[s][action] = paths.tolist()

	with open(output_file, 'w') as fp:
		json.dump(dictionary, fp, sort_keys=True, indent=4, separators=(',', ': '))			


				

