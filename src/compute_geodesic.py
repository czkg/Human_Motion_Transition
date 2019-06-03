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
	opt.is_decoder = True
	input_path = opt.input_path
	s = opt.current_s
	z0 = opt.z0
	z1 = opt.z1

	#read z0 and z1
	act_z0 = z0.split('/')[-2]
	act_z1 = z1.split('/')[-2]
	name_z0 = z0.split('/')[-1][:-4]
	name_z1 = z1.split('/')[-1][:-4]


	if s not in subs:
		raise Exception('unrecodnized s!')

	input_path = os.path.join(input_path, s)
	#read inputs
	z = []
	z_name = []
	z0 = []
	z1 = []
	idx = 0
	acts = os.listdir(input_path)
	print('read inputs from %s ...' % s)
	for act in acts:
		file_list = glob(os.path.join(input_path, act) + '/*.mat')
		for f in file_list:
			data = scipy.io.loadmat(f)['latent'][0]
			z.append(data)
			z_name.append(f.split('/')[-2] + '/' + f.split('/')[-1])
			name = f.split('/')[-1][:-4]
			if act == act_z0 and name == name_z0:
				z0.append(idx)
			if act == act_z1 and name == name_z1:
				z1.append(idx) 
			idx = idx + 1
	z = np.asarray(z)
	#opt.batch_size = inputs.shape[0]
	if len(z0) > 1 or len(z1) > 1:
		assert('multiple z0 or z1')

	#create vae model
	model = create_model(opt)
	print('loading model %s' % opt.model)
	model.setup(opt)
	model.eval()

	print('decode latent code')
	z = torch.tensor(z, requires_grad=True)
	x = model.decoder_with_grad(z)
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
	n_neighbors = 10
	RMetric = RiemannianMetric(x, z, opt)
	RMetric.create_jacob()
	RTree = RiemannianTree(RMetric)
	#create graph
	print('create riemannian graph ...')
	G = RTree.create(z, n_neighbors)

	# path = opt.store_path
	# if not os.path.exists(path):
	# 	os.makedirs(path)
	# path = os.path.join(path, filename)
	# print('save graph ...')
	# nx.write_gpickle(G, path)
	# print('Done!')
	path = nx.shortest_path(G, source = z0[0], target = z1[0], weight = 'weight')
	print(path,' !!!')
	print([z_name[idx] for idx in path], ' !!')


