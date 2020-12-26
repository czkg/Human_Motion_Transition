import os
from data.base_dataset import BaseDataset
from utils.calculate_3Dheatmap import calculate_3Dheatmap
import scipy.io
import torch
import numpy as np
from glob import glob


class LafanDataset(BaseDataset):
	""" This dataset class can load Lafan data specified by the file path --dataroot/path/to/data
	"""

	def __init__(self, opt):
		""" Initialize this dataset class

		Parameters:
			opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""

		BaseDataset.__init__(self, opt)
		self.name2accnum = {}
		self.name2seqaccnum = {}
		self.is_local = opt.lafan_is_local
		self.is_quat = opt.lafan_is_quat
		self.norm = opt.lafan_norm
		self.mode = opt.lafan_mode
		self.window = opt.lafan_window
		self.offset = opt.lafan_offset
		self.samplerate = opt.lafan_samplerate

		if self.is_local is True:
			files = glob(os.path.join(self.root, '*_local.pkl'))
		else:
			files = glob(os.path.join(self.root, '*_global.pkl'))

		prev_pose_count = 0
		prev_seq_count = 0
		self.pose_count = 0
		self.seq_count = 0
		for i, f in enumerate(files):
			with open(os.path.join(self.root, f), 'rb') as ff:
				rawdata = pickle.load(ff, encoding='latin1')
			if self.is_quat is True:
				data = rawdata['Q']
			else:
				data = rawdata['X']
			self.name2accnum[files[i]] = data.shape[0] + prev_pose_count
			# math.floor((L-W)/off) + 1
			self.name2seqaccnum[files[i]] = math.floor((data[::self.framerate].shape[0] - self.window)/self.offset) + 1 
			prev_pose_count = data.shape[0]
			self.pose_count += data.shape[0]
		self.accnum2names = dict((v,k) for k,v in self.name2accnum.items())
		self.seqaccnum2names = dict((v,k) for k,v in self.name2seqaccnum.items())


	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains data and path
		"""
		if self.mode == 'pose':
			upper = [k for k in self.accnum2names.keys() if index < k]
			lower = [k for k in self.accnum2names.keys() if index >= k]
			k = upper[0]
			if len(lower) > 0:
				k_prev = lower[-1]
				in_idx = index - k_prev
			else:
				in_idx = index
			file = self.accnum2names[k]
			with open(os.path.join(self.root, file), 'rb') as f:
				rawdata = pickle.load(f, encoding='latin1')
			if self.is_quat is True:
				data = rawdata['Q']
			else:
				data = rawdata['X']
			pose = data[in_idx]

			return torch.tensor(pose)
		elif self.mode == 'seq':
			upper = [k for k in self.seqaccnum2names.keys() if index < k]
			lower = [k for k in self.seqaccnum2names.keys() if index >= k]
			k = upper[0]
			if len(lower) > 0:
				k_prev = lower[-1]
				init_idx = index - k_prev
			else:
				init_idx = index
			file = self.accnum2names[k]
			with open(os.path.join(self.root, file), 'rb') as f:
				rawdata = pickle.load(f, encoding='latin1')
			if self.is_quat is True:
				data = rawdata['Q']
			else:
				data = rawdata['X']
			seq = data[::self.framerate][init_idx*self.offset : init_idx*self.offset+self.window]
			return torch.tensor(seq)
		else:
			raise('Invalid mode!')

	def __len__(self):
		return self.total_count
		