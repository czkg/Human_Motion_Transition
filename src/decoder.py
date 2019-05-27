import os
import numpy as np
from glob import glob
from options.test_options import TestOptions
import torch
import scipy.io


if __name__ == '__main__':
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

	input_path = opt.input_path
	output_path = opt.output_path

	file_list = glob(input_path + '/*.mat')
	for f in file_list:
		filename = f.split('/')[-1]
		z = scipy.io.loadmat(f)['latent'][0]
		z = torch.tensor(z)
		x = model.decoder(z)
		x = x.data.cpu().numpy()
		filename = os.path.join(output_path, filename)
		scipy.io.savemat(filename, {'heatmap': x})

	print('Done!')

