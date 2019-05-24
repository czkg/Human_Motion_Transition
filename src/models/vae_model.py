import torch
from .base_model import BaseModel
from . import networks
from utils.visualizer import Visualizer


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
		self.n_joints = opt.num_joints
		self.dim_heatmap = opt.dim_heatmap
		self.x_dim = self.dim_heatmap ** 2 * self.n_joints + self.dim_heatmap * self.n_joints
		self.pca_dim = opt.pca_dim
		self.z_dim = opt.z_dim
		self.netVAE = networks.VAE(self.x_dim, self.z_dim, self.pca_dim)
		self.netVAE = networks.init_net(self.netVAE, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)
		if self.isTrain:
			#define loss functions
			self.criterionVAE = networks.VAELoss().to(self.device)
			#initialize optimizers
			#self.optimizerVAE = torch.optim.SGD(self.netVAE.parameters(), lr = opt.lr)
			self.optimizerVAE = torch.optim.Adam(self.netVAE.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999), eps = 1e-6)
			self.optimizers.append(self.optimizerVAE)

		# self.vis = Visualizer(opt) 


	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		self.input = input.to(self.device)


	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.output, self.mu, self.logvar, _ = self.netVAE(self.input)

	def test(self):
		with torch.no_grad():
			out,_,_,z = self.netVAE(self.input)

			return z, out

	def valid(self):
		out,_,_,z = self.netVAE(self.input)
		return z, out

	def update(self):
		self.set_requires_grad(self.netVAE, True)  # enable backprop
		self.optimizerVAE.zero_grad()              # set gradients to zero

		self.loss_VAE = self.criterionVAE(self.mu, self.logvar, self.input, self.output)
		self.loss_VAE.backward()

		self.optimizerVAE.step()

	# def visual(self):
	# 	self.vis.plot_heatmap_xy(self.output[0], self.input[0])

	def get_current_out_in(self):
		return self.output[0],self.input[0]


	def optimize_parameters(self):
		self.forward()
		self.update()