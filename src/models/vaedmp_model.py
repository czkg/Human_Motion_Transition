import torch
from .base_model import BaseModel
from . import networks
from utils.visualizer import Visualizer


class VAEDMPModel(BaseModel):
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
		self.loss_names = ['VAEDMP', 'VAEDMPForce', 'VAEDMP2', 'VAEDMPZ']
		#self.loss_names = ['VAEDMP']
		self.model_names = ['VAEDMP']

		self.file_name = None

		# dimensions
		self.n_joints = opt.num_joints
		self.dim_heatmap = opt.dim_heatmap
		self.x_dim = self.dim_heatmap ** 2 * self.n_joints + self.dim_heatmap * self.n_joints
		self.hidden_dim = opt.hidden_dim
		self.noise_dim = opt.noise_dim
		self.transform_dim = opt.transform_dim
		self.z_dim = opt.z_dim
		self.u_dim = opt.u_dim

		if opt.is_decoder:
			self.is_decoder = True
		else:
			self.is_decoder = False
		self.netVAEDMP = networks.VAEDMP(self.x_dim, self.u_dim, self.z_dim, self.hidden_dim, self.transform_dim, self.noise_dim, self.is_decoder, self.device)
		self.netVAEDMP = networks.init_net(self.netVAEDMP, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)
		if self.isTrain:
			#define loss functions
			self.criterionVAEDMP = networks.VAEDMPLoss().to(self.device)
			self.criterionVAEDMPForce = networks.VAEDMPForceLoss().to(self.device)
			self.criterionVAEDMP2 = networks.VAEDMPLoss().to(self.device)
			self.criterionVAEDMPZ = networks.VAEDMPZLoss().to(self.device)
			#initialize optimizers
			#self.optimizerVAE = torch.optim.SGD(self.netVAE.parameters(), lr = opt.lr)
			self.optimizerVAEDMP = torch.optim.Adam(self.netVAEDMP.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999), eps = 1e-6)
			self.optimizers.append(self.optimizerVAEDMP)

		# self.vis = Visualizer(opt) 


	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		self.input = input['data'].to(self.device).float()
		self.file_name = input['info']

	def set_input_gui(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.
		Parameters:
			input : tensor of input data.
		"""
		self.input = input

	def get_model(self):
		return self.netVAEDMP


	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.output, self.z, self.sx, self.sz, self.wd, self.fs = self.netVAEDMP(self.input)
		#self.output, self.z, self.wd = self.netVAEDMP(self.input)

	def inference(self):
		with torch.no_grad():
			xs,zs,_,_,_,_ = self.netVAEDMP(self.input)

		return zs, xs, self.file_name

	def decoder(self, z):
		z = z.to(self.device)
		if not self.is_decoder:
			assert('should be in decoder mode')
		with torch.no_grad():
			out = self.netVAEDMP(z)
			return out


	def decoder_with_grad(self, z):
		z = z.to(self.device)
		if not self.is_decoder:
			assert('should be in decoder mode')
		out = self.netVAEDMP(z)
		return out

	def update(self):
		self.set_requires_grad(self.netVAEDMP, True)  # enable backprop
		self.optimizerVAEDMP.zero_grad()              # set gradients to zero

		self.loss_VAEDMP = self.criterionVAEDMP(self.wd, self.output, self.input)
		self.loss_VAEDMPForce = self.criterionVAEDMPForce(self.fs)
		self.loss_VAEDMP2 = self.criterionVAEDMP2(self.wd, self.sx, self.input)
		self.loss_VAEDMPZ = self.criterionVAEDMPZ(self.sz, self.z)

		self.loss = self.loss_VAEDMP + self.loss_VAEDMPForce + self.loss_VAEDMP2 + 100*self.loss_VAEDMPZ
		self.loss.backward()

		self.optimizerVAEDMP.step()

	# def visual(self):
	# 	self.vis.plot_heatmap_xy(self.output[0], self.input[0])

	# def get_current_out_in(self):
	# 	return self.output[0],self.input[0]

	def get_current_out_in(self):
		return self.output[0][10],self.input[0][10]

	def optimize_parameters(self):
		self.forward()
		self.update()