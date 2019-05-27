import os
import numpy as np
from options.test_options import TestOptions
from models import create_model
from data import create_dataset
from shutil import rmtree
import scipy.io
import torch
from glob import glob

"""
This module is used to generate latent variable for each pose using pretrained VAE
"""

input_path = '../dataset/Human3.6m/heatmaps/'
output_path = '../dataset/Human3.6m/latent/'
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

if __name__ == '__main__':
	if os.path.exists(output_path):
		rmtree(output_path)

	os.makedirs(output_path)
	for s in subs:
		os.makedirs(os.path.join(output_path, s))

	print('Generating...')

	opt = TestOptions().parse()           # get training options
	opt.num_threads = 1                    # test code only support num_thread = 1
	opt.batch_size = 1                     # test code only support batch_size = 1
	opt.serial_batches = True              # no shuffle

	#create dataset
	#dataset = create_dataset(opt)          # create a dataset given opt.dataset_mode and other options
	model = create_model(opt)
	model.setup(opt)                       # regular setup: load and print networks; create schedulers
	model.eval()
	print('Loading model %s' % opt.model)


	for idx, s in enumerate(subs):
		print('generating', s + ':')
		acts = os.listdir(os.path.join(input_path, s))
		for act in acts:
			print('generating', s + '-' + act)
			os.makedirs(os.path.join(output_path, s, act))
			file_list = glob(os.path.join(input_path, s, act) + '/*.mat')
			for f in file_list:
				filename = f.split('/')[-1]
				heatmap = scipy.io.loadmat(f)['heatmap'][0]
				heatmap = torch.tensor(heatmap)
				model.set_input(heatmap)
				z,_ = model.inference()
				z = z.data.cpu().numpy()
				filename = output_path + s + '/' + act + '/' + filename
				scipy.io.savemat(filename, {'latent': z})

	print('Done!')