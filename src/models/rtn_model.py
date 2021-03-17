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
		self.loss_names = ['RTNRec', 'RTNMono', 'RTNXRec']
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

		# VAEDMP
		dim_heatmap = 64
		#vaedmp_x_dim = dim_heatmap ** 2 * self.n_joints + dim_heatmap * self.n_joints
		vaedmp_x_dim = (self.n_joints + 1) * 4
		vaedmp_z_dim = 32
		vaedmp_u_dim = 32
		vaedmp_hidden_dim = 128
		vaedmp_noise_dim = 32
		vaedmp_transform_dim = 64
		vaedmp_init_type = 'kaiming'
		vaedmp_init_gain = 0.8
		vaedmp_epoch = 100
		self.netVAEDMP = networks.VAEDMP(vaedmp_x_dim, vaedmp_u_dim, vaedmp_z_dim, vaedmp_hidden_dim, vaedmp_transform_dim, vaedmp_noise_dim, True, self.device)
		self.netVAEDMP = networks.load_net(self.netVAEDMP, opt.checkpoints_dir, 'vaedmp', vaedmp_epoch, gpu_ids = opt.gpu_ids)
		if self.isTrain:
			#define loss functions
			self.criterionRTNRec = networks.RTNRecLoss().to(self.device)
			self.criterionRTNMono = networks.RTNMonoLoss().to(self.device)
			self.criterionRTNXRec = networks.RTNXRecLoss().to(self.device)
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
		self.x_gt = input['gt'].to(self.device).float()

		self.gt = self.gt[:, -self.transition_len:, ...]
		self.x_gt = self.x_gt[:, -self.transition_len:, ...]

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
		with torch.no_grad():
			self.x = self.netVAEDMP(self.output)

	def inference(self):
		with torch.no_grad():
			xs, hs, _ = self.netRTN(self.input, self.isTrain)
		return xs, hs, self.file_name

	def update(self):
		self.set_requires_grad(self.netRTN, True)  # enable backprop
		self.optimizerRTN.zero_grad()              # set gradients to zero

		self.loss_RTNRec = self.criterionRTNRec(self.output, self.gt)
		self.loss_RTNMono = self.criterionRTNMono(self.f)
		self.loss_RTNXRec = self.criterionRTNXRec(self.x, self.x_gt)
		self.loss = self.loss_RTNRec + self.loss_RTNMono + self.loss_RTNXRec
		self.loss.backward()

		self.optimizerRTN.step()

	# def visual(self):
	# 	self.vis.plot_heatmap_xy(self.output[0], self.input[0])

	def get_current_out_in(self):
		return self.x[0][10],self.x_gt[0][10]


	def optimize_parameters(self):
		self.forward()
		self.update()