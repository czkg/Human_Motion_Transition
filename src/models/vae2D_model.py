import torch
from .base_model import BaseModel
from . import networks
from utils.visualizer import Visualizer


class VAE2DModel(BaseModel):
	""" This class implements the VAE2D model.
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
		self.loss_names = ['VAE2D']
		self.model_names = ['VAE2D']
		self.x_dim = opt.x_dim
		self.pca_dim = opt.pca_dim
		self.z_dim = opt.z_dim
		if opt.is_decoder:
			self.is_decoder = True
		else:
			self.is_decoder = False
		self.netVAE2D = networks.VAE2D(self.x_dim, self.z_dim, self.pca_dim, self.is_decoder)
		self.netVAE2D = networks.init_net(self.netVAE2D, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)
		if self.isTrain:
			#define loss functions
			self.criterionVAE2D = networks.VAE2DLoss().to(self.device)
			#initialize optimizers
			#self.optimizerVAE2 = torch.optim.SGD(self.netVAE2.parameters(), lr = opt.lr)
			self.optimizerVAE2D = torch.optim.Adam(self.netVAE2D.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999), eps = 1e-6)
			self.optimizers.append(self.optimizerVAE2D)

		# self.vis = Visualizer(opt) 


	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		self.input = input.to(self.device).float()

	def get_model(self):
		return self.netVAE2D


	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.output, self.z, self.mu, self.logvar = self.netVAE2D(self.input)

	def inference(self):
		with torch.no_grad():
			out,z,_,_ = self.netVAE2D(self.input)

		return z,out

	def decoder(self, z):
		z = z.to(self.device)
		if not self.is_decoder:
			assert('should be in decoder mode')
		with torch.no_grad():
			out = self.netVAE2D(z)
			return out


	def decoder_with_grad(self, z):
		z = z.to(self.device)
		if not self.is_decoder:
			assert('should be in decoder mode')
		out = self.netVAE2D(z)
		return out

	def update(self):
		self.set_requires_grad(self.netVAE2D, True)  # enable backprop
		self.optimizerVAE2D.zero_grad()              # set gradients to zero

		self.loss_VAE2D = self.criterionVAE2D(self.mu, self.logvar, self.input, self.output)
		self.loss_VAE2D.backward()

		self.optimizerVAE2D.step()

	# def visual(self):
	# 	self.vis.plot_heatmap_xy(self.output[0], self.input[0])

	def get_current_out_in(self):
		return self.output[0],self.input[0]


	def optimize_parameters(self):
		self.forward()
		self.update()