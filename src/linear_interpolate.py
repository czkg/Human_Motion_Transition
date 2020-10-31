import os
import numpy as np
from glob import glob
from options.test_options import TestOptions
import torch
import scipy.io
from shutil import rmtree
from models import create_model
import time

p0 = '../dataset/Human3.6m/latent_nth/S5/Smoking/131.mat'
p1 = '../dataset/Human3.6m/latent_nth/S5/Smoking/176.mat'
output_path = '../res/linear/heatmaps/'

def slerp(p0, p1, t):
    """
    Spherical linear interpolation
    """
    omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)),
                             np.squeeze(p1/np.linalg.norm(p1))))
    so = np.sin(omega)
    return np.sin(1.0 - t) * omega / so * p0 + np.sin(t * omega) / so * p1


if __name__ == '__main__':
	start_time = time.time()
	opt = TestOptions().parse()
	opt.num_threads = 1
	opt.batch_size = 1
	opt.serial_batches = True
	opt.is_decoder = True

	#create model
	model = create_model(opt)
	model.setup(opt)
	model.eval()
	print('Loading model %s' % opt.model)

	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)
	d0 = scipy.io.loadmat(p0)['latent'][0]
	d1 = scipy.io.loadmat(p1)['latent'][0]

	data = np.array([slerp(d0, d1, t) for t in np.linspace(0, 1, 10)])
	end_time = time.time()

	for i in range(10):
		z = torch.tensor(data[i])
		x = model.decoder(z)
		x = x.data.cpu().numpy()
		filename = str(i + 1) + '.mat'
		filename = os.path.join(output_path, filename)
		scipy.io.savemat(filename, {'heatmap': x})

	print('Done!')
	print('Time:', end_time - start_time)