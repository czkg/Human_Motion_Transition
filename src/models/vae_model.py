import torch
from .base_model import BaseModel
from . import networks

num_joints = 17


class VAEModel(BaseModel):
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
		self.loss_names = ['VAE']
		self.model_names = ['VAE']
		x_dim = opt.dim_heatmap ** 2 * num_joints + opt.dim_heatmap * num_joints
		self.netVAE = networks.VAE(x_dim, opt.z_dim, opt.pca_dim)
		self.netVAE = networks.init_net(self.netVAE, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)
		if self.isTrain:
			#define loss functions
			self.criterionVAE = networks.VAELoss().to(self.device)
			#initialize optimizers
			self.optimizer = torch.optim.Adam(self.netVAE.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))


	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		self.input = input.to(self.device)


	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.output, self.mu, self.logvar = self.netVAE(self.input)


	def update(self):
		self.set_requires_grad(self.netVAE, True)  # enable backprop
		self.optimizer.zero_grad()              # set gradients to zero

		self.loss_VAE = self.criterionVAE(self.mu, self.logvar, self.input, self.output)
		self.loss_VAE.backward()

		self.optimizer.step()


	def optimize_parameters(self):
		self.forward()
		self.update()