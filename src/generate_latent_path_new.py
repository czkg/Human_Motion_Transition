import os
import numpy as np
from shutil import rmtree
import scipy.io
from glob import glob
import json
import scipy.io


"""
This module is used to generate paths composed of latent poses based on the json file
"""

input_path = '../dataset/Human3.6m/latent_nth/'
reference = '../dataset/Human3.6m/new_paths.json'
output_path = '../dataset/Human3.6m/latent_path_new/'

if __name__ == '__main__':
	if os.path.exists(output_path):
		rmtree(output_path)

	os.makedirs(output_path)

	print('Generating...')

	with open(reference) as ref:
		data = json.load(ref)
	subs = [s for s in data.keys()]
	for s in subs:
		print('generating', s + ':')
		for i in range(len(data[s])):
			path_ids = data[s][i]
			name = output_path + s + '_' + str(i+1) + '.mat'
			paths = []
			for k in range(len(path_ids)):
				pid = path_ids[k]
				act = pid.split('_')[0]
				iid = pid.split('_')[1]

				filename = input_path + s + '/' + act + '/' + iid + '.mat'
				latent = scipy.io.loadmat(filename)['latent'][0]
				paths.append(latent)
			scipy.io.savemat(name, {'data': paths})

	print('Done!')
