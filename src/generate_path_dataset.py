import os
import numpy as np
import argparse
from glob import glob
from shutil import rmtree
import scipy.io
from random import shuffle
import json

input_path = '../dataset/Human3.6m/latent_nth'
output_file = '../dataset/Human3.6m/new_paths.json'
log_file = '../dataset/Human3.6m/log.json'
length = 10

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--input_path', type=str, help='the input path')
	# parser.add_argument('--output_path', type=str, help='the output path')
	# args = vars(parser.parse_args())

	# input_path = args['input_path']
	# output_path = args['output_path']

	if not os.path.exists(input_path):
		raise Exception('Input not exists!')

	if os.path.isfile(output_file):
		os.remove(output_file)


	subs = os.listdir(input_path)
	dictionary = {}
	log = {}

	for s in subs:
		print('processing', s + ':')
		acts = os.listdir(os.path.join(input_path, s))
		num = len(acts)
		act_idx = list(range(num))
		shuffle(act_idx)
		data = []
		dictionary[s] = {}

		for idx in act_idx:
			act = acts[idx]
			print('processing', s + '-' + act)
			current_path = os.path.join(input_path, s, act)
			file_list = glob(current_path + '/*.mat')

			id_list = [f.split('/')[-1][:-4] for f in file_list]
			id_list = list(map(int, id_list))
			id_list.sort()

			id_list_rev = id_list.copy()
			id_list_rev.reverse()
			new_list = id_list + id_list_rev[1:]
			new_list = list(map(str, new_list))
			new_list = [act + '_' + ii for ii in new_list]

			data = data + new_list

		n = len(data) - length + 1
		paths = []
		for i in range(n):
			paths.append(data[i: i + length])
		dictionary[s] = paths

		#log
		dd = [acts[tt] for tt in act_idx]
		log[s] = dd

	with open(output_file, 'w') as fp:
		json.dump(dictionary, fp, sort_keys=True, indent=4, separators=(',', ': '))

	with open(log_file, 'w') as fp:
		json.dump(log, fp, sort_keys=True, indent=4, separators=(',', ': '))




