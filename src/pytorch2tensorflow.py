import torch
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import InputLayer, Dense, LeakyReLU
import os
from options.test_options import TestOptions
from models import create_model
import keras.backend as K
import numpy as np

load_path = '../results/vae/20_net_VAE.pth'
save_path = '../dataset/Human3.6m/vae_weights_bias.h5'

if __name__ == '__main__':
	opt = TestOptions().parse()       # get training options
	#model = create_model(opt)
	#model.setup(opt)

	batch_size = 1
	dim_heatmap = opt.dim_heatmap
	num_joints = opt.num_joints
	x_dim = (dim_heatmap ** 2 + dim_heatmap) * num_joints
	pca_dim = opt.pca_dim
	z_dim = opt.z_dim
	#x = torch.randn(batch_size, size, requires_grad=True)
	#m = model.get_model()

	state_dict = torch.load(load_path)
	#print(state_dict.keys())

	model = Sequential()
	#model.add(InputLayer(input_shape = (z_dim,), name = 'input'))
	model.add(Dense(pca_dim, input_dim = z_dim, name = 'fc5'))
	model.add(LeakyReLU(alpha = 0.01))
	model.add(Dense(pca_dim, name = 'fc6'))
	model.add(LeakyReLU(alpha = 0.01))
	model.add(Dense(pca_dim, name = 'fc7'))
	model.add(LeakyReLU(alpha = 0.01))
	model.add(Dense(x_dim, activation = 'sigmoid', name = 'fc8'))


	for layer in model.layers:
		if layer.name.startswith('fc'):
			name = layer.name
			weight_name = name + '.weight'
			bias_name = name + '.bias'
			val = []
			val.append(np.transpose(state_dict[weight_name].data.cpu().numpy()))
			val.append(state_dict[bias_name].data.cpu().numpy())
			layer.set_weights(val)

	model.save_weights(save_path)


	#for i, weights in enumerate(weights_list):



