import numpy as np
import networkx as nx
import os
from glob import glob
import scipy.io
import torch
from torch.autograd import Variable
from utils.riemannian_metric import RiemannianMetric
from utils.riemannian_tree import RiemannianTree
from options.metric_options import MetricOptions
from models import create_model
from data import create_dataset

subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
filename = 'graph.gpickle'

if __name__ == '__main__':
	opt = MetricOptions().parse()
	opt.num_threads = 1
	opt.serial_batches = True
	input_path = opt.dataroot
	s = opt.current_s

	if s not in subs:
		raise Exception('unrecodnized s!')

	input_path = os.path.join(input_path, s)
	#read inputs
	inputs = []
	acts = os.listdir(input_path)
	print('read inputs from %s ...' % s)
	for act in acts:
		file_list = glob(os.path.join(input_path, act) + '/*.mat')
		for f in file_list:
			data = scipy.io.loadmat(f)['heatmap'][0]
			inputs.append(data)
	inputs = np.asarray(inputs)
	#opt.batch_size = inputs.shape[0]

	#create vae model
	model = create_model(opt)
	print('loading model %s' % opt.model)
	model.setup(opt)
	model.eval()

	print('decode latent code')
	model.set_input(torch.tensor(inputs, requires_grad=True))
	z, x = model.valid()
	# step = inputs.shape[0] // opt.batch_size

	# current_input = inputs[: opt.batch_size, :]
	# current_input = torch.tensor(current_input, requires_grad=True)
	# model.set_input(current_input)
	# z, x = model.valid()
	# last = 0
	# for i in range(step):
	# 	current_input = inputs[i * opt.batch_size : (i + 1) * opt.batch_size, :]
	# 	model.set_input(torch.tensor(current_input))
	# 	current_z, current_x = model.valid()
	# 	x = torch.cat((x, current_x), 0)
	# 	z = torch.cat((z, current_z), 0)
	# 	last = i
	# current_input = inputs[(last + 1) * opt.batch_size :, :]
	# model.set_input(torch.tensor(current_input))
	# current_z, current_x = model.valid()
	# x = torch.cat((x, current_x), 0)
	# z = torch.cat((z, current_z), 0)


	#how many steps do we need for each pair of z neighbors
	n_neighbors = 4
	RMetric = RiemannianMetric(x, z, opt)
	RMetric.create_jacob()
	RTree = RiemannianTree(RMetric)
	#create graph
	print('create riemannian graph ...')
	graph = RTree.create(z, n_neighbors)

	path = opt.store_path
	if not os.path.exists(path):
		os.makedirs(path)
	path = os.path.join(path, filename)
	print('save graph ...')
	nx.write_gpickle(graph, path)
	print('Done!')
