import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.calculate_3Dheatmap import calculate_3Dheatmap
from utils.utils import slerp
from utils.utils import blend

import scipy.io
from glob import glob
import tqdm
from shutil import rmtree
import torch
import numpy as np
import time
import pickle

eps = 2e-1

def load_fvae(opt, file_names):
	"""Load data for fvae module
	Parameters:
		opt: options
		file_names: ndarray of input file names
	Return:
		data: tensor of fvae input data
	"""
	fvae_data = []
	for f_name in file_names:
		with open(f_name, 'rb') as f:
			data = pickle.load(f, encoding='latin1')
		data = calculate_3Dheatmap(data, opt.dim_heatmap, opt.sigma)
		fvae_data.append(data)
	fvae_data = np.asarray(fvae_data)
	return torch.from_numpy(fvae_data)

def load_rtn(file_names):
	"""Load data for rtn module
	Parameters:
		opt: options
		file_names: ndarray of input data
	Return:
		data: tensor of rtn input data
	"""
	rtn_data = []
	for f_name in file_names:
		with open(f_name, 'rb') as f:
			data = pickle.load(f, encoding='latin1')
		rtn_data.append(data)
	rtn_data = np.asarray(rtn_data)
	return torch.from_numpy(rtn_data)



def run(opt):
	start_time = time.time()
	# options
	dataset = create_dataset(opt)        # create a dataset given opt.dataset_mode and other options
	dataset_size = len(dataset)        # get dataset size

	# create model
	model = create_model(opt)
	model.setup(opt)
	model.eval()
	print('Loading model %s' % opt.model)

	output_path = opt.output_path

	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)
	print("Testing...")

	for i, data in enumerate(dataset):   # inner loop within one epoch
		model.set_input(data)
		out, gt, file_name = model.inference()
		out = out[0].data.cpu().numpy()
		gt = gt[0].data.cpu().numpy()
		file_name = file_name[0]
		
		out_path = os.path.join(output_path, file_name)

		out_data = {'data': out, 'gt': gt}
		with open(out_path, 'wb') as f:
			pickle.dump(out_data, f, protocol=pickle.HIGHEST_PROTOCOL)

	end_time = time.time()
	print('Time:', end_time - start_time)
	print('Done!')


def run_gui(opt, data):
	# create model
	model = create_model(opt)
	model.setup(opt)
	model.eval()
	print('Loading model %s' % opt.model)

	output_path = opt.output_path


	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)
	print("Evaluating...")

	model.set_input_gui(data)
	output, _, _ = model.inference()
	output = output.data.cpu().numpy()

	return output

if __name__ == '__main__':
	opt = TestOptions().parse()
	run(opt)