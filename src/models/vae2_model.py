import torch
from .base_model import BaseModel
from . import networks
from utils.visualizer import Visualizer


class VAE2Model(BaseModel):
	""" This class implements the VAE model.
	"""

	@staticmethod
	def modify_commandline_options(parser, is_train = True):
		"""Add new options
		"""
		return parser


	def __init__(self, opt):
		""" Initialize the vae class.

		Parameters:
			opt (Option class)-- stores all the experiment flags, needs to be a subclass of BaseOptions
		"""
		BaseModel.__init__(self, opt)
		self.loss_names = ['VAE2']
		self.model_names = ['VAE2']
		self.x_dim = opt.x_dim
		self.pca_dim = opt.pca_dim
		self.z_dim = opt.z_dim
		if opt.is_decoder:
			self.is_decoder = True
		else:
			self.is_decoder = False
		self.netVAE2 = networks.VAE2(self.x_dim, self.z_dim, self.pca_dim, self.is_decoder)
		self.netVAE2 = networks.init_net(self.netVAE2, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)
		if self.isTrain:
			#define loss functions
			self.criterionVAE = networks.VAE2Loss().to(self.device)
			#initialize optimizers
			#self.optimizerVAE2 = torch.optim.SGD(self.netVAE2.parameters(), lr = opt.lr)
			self.optimizerVAE2 = torch.optim.Adam(self.netVAE2.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999), eps = 1e-6)
			self.optimizers.append(self.optimizerVAE2)

		# self.vis = Visualizer(opt) 


	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		self.input = input.to(self.device).float()

	def get_model(self):
		return self.netVAE2


	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.output, self.zd, _ = self.netVAE2(self.input)

	def inference(self):
		with torch.no_grad():
			out,zd,z = self.netVAE2(self.input)

		return z,out

	def decoder(self, z):
		z = z.to(self.device)
		if not self.is_decoder:
			assert('should be in decoder mode')
		with torch.no_grad():
			out = self.netVAE2(z)
			return out


	def decoder_with_grad(self, z):
		z = z.to(self.device)
		if not self.is_decoder:
			assert('should be in decoder mode')
		out = self.netVAE2(z)
		return out

	def update(self):
		self.set_requires_grad(self.netVAE2, True)  # enable backprop
		self.optimizerVAE2.zero_grad()              # set gradients to zero

		self.loss_VAE2 = self.criterionVAE(self.zd, self.input, self.output)
		self.loss_VAE2.backward()

		self.optimizerVAE2.step()

	# def visual(self):
	# 	self.vis.plot_heatmap_xy(self.output[0], self.input[0])

	def get_current_out_in(self):
		return self.output[0],self.input[0]


	def optimize_parameters(self):
		self.forward()
		self.update()