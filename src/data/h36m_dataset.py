import os
from data.base_dataset import BaseDataset
from utils.calculate_3Dheatmap import calculate_3Dheatmap
import scipy.io
import torch
import numpy as np
from glob import glob
import pickle
import math
from numpy import genfromtxt


H36M_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

class Atree():
	def __init__(self):
		self.acts = []
		self.nums = []
		self.total_count = 0

	def add(self, act, num):
		self.acts.append(act)
		self.nums.append(num)
		self.total_count += num

class SATree():
	def __init__(self):
		self.subs = []
		self.atrees = []
		self.nums = []

	def add(self, sub, atree):
		self.subs.append(sub)
		self.atrees.append(atree)
		self.nums.append(atree.total_count)

class H36MDataset(BaseDataset):
	""" This dataset class can load H36M data specified by the file path --dataroot/path/to/data
	"""

	def __init__(self, opt):
		""" Initialize this dataset class

		Parameters:
			opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""

		BaseDataset.__init__(self, opt)
		self.mode = opt.h36m_mode
		self.total_count = 0
		self.satree = SATree()
		
		for sub in H36M_SUBJECTS:
			acts = os.listdir(os.path.join(self.root, sub))
			atree = Atree()
			for act in acts:
				path = os.path.join(self.root, sub, act)
				files = glob(os.path.join(path, '*.csv'))
				atree.add(act, len(files))
				self.total_count += len(files)
			self.satree.add(sub, atree)

	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains data and path
		"""
		if self.mode == 'pose':
			i = 0
			in_idx = 0
			for i in range(len(self.satree.subs)):
				ii = index / self.satree.nums[i]
				in_idx = index % self.satree.nums[i]
				if ii < 1:
					break
				else:
					index -= self.satree.nums[i]
			sub = self.satree.subs[i]

			j = 0
			in_in_idx = 0
			for j in range(len(self.satree.atrees[i].acts)):
				jj = in_idx / self.satree.atrees[i].nums[j]
				in_in_idx = in_idx % self.satree.atrees[i].nums[j]
				if jj < 1:
					break
				else:
					in_idx -= self.satree.atrees[i].nums[j]
			act = self.satree.atrees[i].acts[j]
			file_name = str(int(in_in_idx+1)) + '.csv'
			file_path = os.path.join(self.root, sub, act, file_name)
			pose = genfromtxt(file_path, delimiter=' ')[1:,...]
			pose = pose.reshape(-1)

			return torch.tensor(pose)
		elif self.mode == 'seq':
			# upper = [k for k in self.seqaccnum2names.keys() if index < k]
			# lower = [k for k in self.seqaccnum2names.keys() if index >= k]
			# k = upper[0]
			# if len(lower) > 0:
			# 	k_prev = lower[-1]
			# 	init_idx = index - k_prev
			# else:
			# 	init_idx = index
			# file = self.seqaccnum2names[k]
			# with open(file, 'rb') as f:
			# 	rawdata = pickle.load(f, encoding='latin1')
			# if self.is_quat is True:
			# 	data = rawdata['Q']
			# else:
			# 	data = rawdata['X']
			# seq = data[::self.samplerate][init_idx*self.offset : init_idx*self.offset+self.window]
			# seq = np.asarray(seq)	

			# add rv
			# if self.is_local is True:
			# 	global_file = file[:-9] + 'global.pkl'
			# else:
			# 	global_file = file
			# with open(os.path.join(self.root, global_file), 'rb') as f:
			# 	global_data = pickle.load(f, encoding='latin1')
			# rv = global_data['rv'][::self.framerate][init_idx*self.offset : init_idx*self.offset+self.window]
			# rv = rv[:,np.newaxis,...]
			# seq = np.concatenate((rv, seq), axis=1)

			# seq = seq.reshape(seq.shape[0], -1)

			# return torch.tensor(seq)
			return torch.tensor(0)
		else:
			raise('Invalid mode!')

	def __len__(self):
		if self.mode == 'pose':
			return self.total_count
		else:
			return 0
		