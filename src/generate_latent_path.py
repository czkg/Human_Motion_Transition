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

input_path = '../dataset/Human3.6m/latent/'
reference = '../dataset/Human3.6m/paths.json'
output_path = '../dataset/Human3.6m/latent_path/'

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
		acts = [a for a in data[s]]
		for a in acts:
			print('generating', s + '-' + a)
			for i in range(len(data[s][a])):
				path_ids = data[s][a][i]
				name = output_path + s + '_' + a + '_' + str(i+1) + '.mat'
				paths = []
				for k in range(len(path_ids)):
					filename = input_path + s + '/' + a + '/' + str(path_ids[k]) + '.mat'
					latent = scipy.io.loadmat(filename)['latent'][0]
					paths.append(latent)
				scipy.io.savemat(name, {'data': paths})

	print('Done!')
