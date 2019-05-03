import torch
from .base_model import BaseModel
from . import networks


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
		self.vae = networks.VAE(opt.dim_heatmap, opt.z_dim, opt.pca_dim).to(self.device)
		if self.isTrain:
			#define loss functions
			self.loss = networks.VAELoss().to(self.device)
			#initialize optimizers
			self.optimizer = torch.optim.Adam(self.vae.parameters(), lr = opt.lr, betas = opt.beta1)


	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		self.input = input['heatmap'].to(self.device)


	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.output, self.mu, self.logvar = self.vae(self.input)


	def update(self):
		self.set_requires_grad(self.vae, True)  # enable backprop
		self.optimizer.zero_grad()              # set gradients to zero
		self.loss(self.mu, self.logvar, self.input, self.output)
		self.loss.backward()
		self.optimizer.step()


	def optimize_parameters(self):
		self.forward()
		self.update()