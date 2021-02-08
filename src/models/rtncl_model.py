import torch
from .base_model import BaseModel
from . import networks
from utils.visualizer import Visualizer


class RTNCLModel(BaseModel):
	""" This class implements the RTNCL model.
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
		self.loss_names = ['RTNCL']
		self.model_names = ['RTNCL']
		self.n_joints = opt.num_joints
		self.x_dim = opt.x_dim
		self.hidden_dim = opt.hidden_dim
		self.t_dim = 2*opt.x_dim
		self.o_dim = opt.x_dim
		if opt.is_decoder:
			self.is_decoder = True
		else:
			self.is_decoder = False
		self.netRTNCL = networks.RTNCL(self.x_dim, self.t_dim, self.hidden_dim, self.is_decoder)
		self.netRTNCL = networks.init_net(self.netRTNCL, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)
		if self.isTrain:
			#define loss functions
			self.criterionRTNCL = networks.RTNCLLoss().to(self.device)
			#initialize optimizers
			#self.optimizerVAE2 = torch.optim.SGD(self.netVAE2.parameters(), lr = opt.lr)
			self.optimizerRTNCL = torch.optim.Adam(self.netRTNCL.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999), eps = 1e-6)
			self.optimizers.append(self.optimizerRTNCL)

		# self.vis = Visualizer(opt) 


	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		self.input = input.to(self.device).float()
		self.target = input.to(self.device).float()

		past_idx = range(10)
		self.input = self.input[:, past_idx, ...]
		self.target = self.target[:, 1:, ...]

		self.t = self.target[:, -2:, ...]

	def get_model(self):
		return self.netRTNCL


	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.hs, self.output = self.netRTNCL(self.input, self.t)

	def inference(self):
		with torch.no_grad():
			hs, xs = self.netRTNCL(self.input)

		return hs, xs

	def update(self):
		self.set_requires_grad(self.netRTNCL, True)  # enable backprop
		self.optimizerRTNCL.zero_grad()              # set gradients to zero

		self.loss_RTNCL = self.criterionRTNCL(self.output, self.target)
		self.loss_RTNCL.backward()

		self.optimizerRTNCL.step()

	# def visual(self):
	# 	self.vis.plot_heatmap_xy(self.output[0], self.input[0])

	def get_current_out_in(self):
		return self.output[0],self.input[0]


	def optimize_parameters(self):
		self.forward()
		self.update()