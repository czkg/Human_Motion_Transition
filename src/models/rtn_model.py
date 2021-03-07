import torch
from .base_model import BaseModel
from . import networks
from utils.visualizer import Visualizer


class RTNModel(BaseModel):
	""" This class implements the RTN model.
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
		self.loss_names = ['RTNRec', 'RTNMono']
		self.model_names = ['RTN']

		self.file_name = None

		self.n_joints = opt.num_joints
		self.x_dim = opt.x_dim
		self.hidden_dim = opt.hidden_dim
		self.z_dim = opt.z_dim

		self.transition_len = opt.transition_len
		self.past_len = opt.past_len
		self.target_len = opt.target_len
		if opt.is_decoder:
			self.is_decoder = True
		else:
			self.is_decoder = False
		self.netRTN = networks.RTN(self.x_dim, self.z_dim, self.hidden_dim, self.is_decoder)
		self.netRTN = networks.init_net(self.netRTN, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)
		if self.isTrain:
			#define loss functions
			self.criterionRTNRec = networks.RTNRecLoss().to(self.device)
			self.criterionRTNMono = networks.RTNMonoLoss().to(self.device)
			#initialize optimizers
			#self.optimizerVAE2 = torch.optim.SGD(self.netVAE2.parameters(), lr = opt.lr)
			self.optimizerRTN = torch.optim.Adam(self.netRTN.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999), eps = 1e-6)
			self.optimizers.append(self.optimizerRTN)

		# self.vis = Visualizer(opt) 


	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.
		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		self.input = input['data'].to(self.device).float()
		self.gt = input['data'].to(self.device).float()

		self.gt = self.gt[:, -self.transition_len:, ...]

		self.file_name = input['info']

	def set_input_gui(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.
		Parameters:
			input : tensor of input data.
		"""
		self.input = input

	def get_model(self):
		return self.netRTN

	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.output, self.hs, self.f = self.netRTN(self.input, self.isTrain)

	def inference(self):
		with torch.no_grad():
			xs, hs, _ = self.netRTN(self.input, self.isTrain)

		return xs, hs, self.file_name

	def update(self):
		self.set_requires_grad(self.netRTN, True)  # enable backprop
		self.optimizerRTN.zero_grad()              # set gradients to zero

		self.loss_RTNRec = self.criterionRTNRec(self.output, self.gt)
		self.loss_RTNMono = self.criterionRTNMono(self.f)
		self.loss = self.loss_RTNRec + self.loss_RTNMono
		self.loss.backward()

		self.optimizerRTN.step()

	# def visual(self):
	# 	self.vis.plot_heatmap_xy(self.output[0], self.input[0])

	def get_current_out_in(self):
		return self.output[0],self.input[0]


	def optimize_parameters(self):
		self.forward()
		self.update()