import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import scipy.io
from glob import glob
import tqdm
from shutil import rmtree
import torch
import numpy as np
from utils.utils import slerp
import time


if __name__ =='__main__':
	start_time = time.time()
	# options
	opt = TestOptions().parse()
	opt.num_threads = 1    # test code only support num_threads = 1
	opt.batch_size = 1     # test code only support batch_size = 1
	opt.serial_batches = True  # no shuffle

	# create model
	model = create_model(opt)
	model.setup(opt)
	model.eval()
	print('Loading model %s' % opt.model)

	input_path = opt.input_path
	output_path = opt.output_path
	model_name = opt.model
	path_length = opt.path_length
	z0 = opt.z0
	z1 = opt.z1
	a_mode = opt.a_mode

	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)

	# test data
	if model_name == 'vae':
		file_list = glob(input_path + '/*.mat')
		num = len(file_list)
		for i in tqdm.trange(num):
			f = file_list[i]
			filename = f.split('/')[-1]
			x = scipy.io.loadmat(f)['heatmap'][0]
			x = torch.tensor(x)
			model.set_input(x)
			_, out = model.inference()
			out = out.data.cpu().numpy()
			filepath = os.path.join(output_path, filename)
			scipy.io.savemat(filepath, {'heatmap': out})

	elif model_name == 'path_gan':
		z0 = scipy.io.loadmat(z0)['latent'][0]
		z1 = scipy.io.loadmat(z1)['latent'][0]
		size = len(z0)
		a = np.zeros((1, size * path_length))
		a[0][:size] = z0
		a[0][-size:] = z1

		if a_mode == 'zeros':
			b = a
		elif a_mode == 'ones':
			a[0][size:-size] = 1.0
			b = a
		elif a_mode == 'linear':
			ba = np.array([slerp(z0, z1, t) for t in np.linspace(0, 1, path_length)])
			mid = ba[1:-1]
			a[0][size:-size] = np.ravel(mid)
			b = a
		else:
			raise NotImplementedError('mode [%s] is not implemented' % a_mode)
		a = torch.tensor(a)
		b = torch.tensor(b)
		a = a.float()
		b = b.float()
		model.set_input({'A': a, 'B': b})
		out = model.inference().data.cpu().numpy()[0]
		out = np.split(out, path_length)
		for i in range(path_length):
			filepath = output_path + '/' + str(i+1) + '.mat'
			scipy.io.savemat(filepath, {'latent': out[i]})

	else:
		raise('unrecognized mode')

	end_time = time.time()
	print('Time:', end_time - start_time)
	print('Done!')