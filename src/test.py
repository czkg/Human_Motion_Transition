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

def postprocess(output, gt, opt):
	"""postprocess
	Parameters:
		output (array): output array of shape [len_sequence, x_dim]
		gt (array): ground truth array of shape [len_sequence, x_dim]
	"""
	if opt.model == 'rtncl':
		root = np.zeros((output.shape[0], 3))
		output = np.concatenate((root, output), axis=1)
		output = output.reshape(output.shape[0], -1, 3)

		return output
	elif opt.model == 'rtn2':
		transition  = blend(output, gt[-1])
		return transition
	elif opt.model == 'vae2':
		# dataset = opt.dataset_mode
		# minmax_path = getattr(opt, dataset + '_minmax_path')
		# minmax = np.load(minmax_path)
		# mmin = minmax[0]
		# mmax = minmax[1]

		# x = mmin + (mmax - mmin) * x
		root = np.zeros(3)
		output = np.concatenate((root, output), axis=0)
		output = output.reshape(-1, 3)
		output = output[np.newaxis, ...]

		return output
	elif opt.model == 'vaedmp':
		return output
	elif opt.model == 'rtn':
		return output


def run(opt):
	start_time = time.time()
	# options
	#opt = TestOptions().parse()
	# opt.num_threads = 1    # test code only support num_threads = 1
	# opt.batch_size = 1     # test code only support batch_size = 1
	# opt.serial_batches = True  # no shuffle
	dataset = create_dataset(opt)        # create a dataset given opt.dataset_mode and other options
	dataset_size = len(dataset)        # get dataset size

	# create model
	model = create_model(opt)
	model.setup(opt)
	model.eval()
	print('Loading model %s' % opt.model)

	# dataset = create_dataset(opt)        # create a dataset given opt.dataset_mode and other options
	# dataset_size = len(dataset)        # get dataset size

	# input_path = opt.input_path
	output_path = opt.output_path
	# model_name = opt.model
	# path_length = opt.path_length
	# z0 = opt.z0
	# z1 = opt.z1
	# a_mode = opt.a_mode

	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)
	print("Testing...")

	for i, data in enumerate(dataset):   # inner loop within one epoch
		model.set_input(data)
		_, out, _, _, file_name = model.inference()
		out = out[0].data.cpu().numpy()
		gt = data['data'][0].data.cpu().numpy()
		file_name = file_name[0]

		out = postprocess(out, gt, opt)
		
		out_path = os.path.join(output_path, file_name)

		out_data = {'data': out, 'gt': gt}
		with open(out_path, 'wb') as f:
			pickle.dump(out_data, f, protocol=pickle.HIGHEST_PROTOCOL)

	# test data
	# if model_name == 'vaedmp' or model_name == 'vae2':
	# 	file_list = glob(input_path + '/*.npy')
	# 	num = len(file_list)
	# 	for f in file_list:
	# 		f_name = f.split('/')[-1]
	# 		data_in = np.load(f)
	# 		data_in = data_in[np.newaxis, ...]
	# 		data = torch.tensor(data_in).float()
	# 		model.set_input(data)
	# 		_, out = model.inference()
	# 		out = out.data.cpu().numpy()
	# 		filepath = os.path.join(output_path, f_name)
	# 		np.save(filepath, out)

	# elif model_name == 'path_gan':
	# 	z0 = scipy.io.loadmat(z0)['latent'][0]
	# 	z1 = scipy.io.loadmat(z1)['latent'][0]
	# 	size = len(z0)
	# 	a = np.zeros((1, size * path_length))
	# 	a[0][:size] = z0
	# 	a[0][-size:] = z1

	# 	if a_mode == 'zeros':
	# 		b = a
	# 	elif a_mode == 'ones':
	# 		a[0][size:-size] = 1.0
	# 		b = a
	# 	elif a_mode == 'linear':
	# 		ba = np.array([slerp(z0, z1, t) for t in np.linspace(0, 1, path_length)])
	# 		mid = ba[1:-1]
	# 		a[0][size:-size] = np.ravel(mid)
	# 		b = a
	# 	else:
	# 		raise NotImplementedError('mode [%s] is not implemented' % a_mode)
	# 	a = torch.tensor(a)
	# 	b = torch.tensor(b)
	# 	a = a.float()
	# 	b = b.float()
	# 	model.set_input({'A': a, 'B': b})
	# 	out = model.inference().data.cpu().numpy()[0]
	# 	out = np.split(out, path_length)
	# 	for i in range(path_length):
	# 		filepath = output_path + '/' + str(i+1) + '.mat'
	# 		scipy.io.savemat(filepath, {'latent': out[i]})

	# else:
	# 	raise('unrecognized mode')

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