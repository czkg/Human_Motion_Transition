from data.base_dataset import BaseDataset
import numpy as np
import os
import scipy.io
import torch
from utils.utils import slerp
import random
from random import randrange

class CompoundPathDataset(BaseDataset):
	def __init__(self, opt):
		""" Initialize this dataset class, the combination of aligned path and random path
		Parameters:
			opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)
		self.randomroot = opt.randomroot
		self.paths = []
		self.poses = []
		path_files = os.listdir(self.root)
		pose_files = os.listdir(self.randomroot)
		for f in path_files:
			path = os.path.join(self.root, f)
			self.paths.append(path)
		for f in pose_files:
			pose = os.path.join(self.randomroot, f)
			self.poses.append(pose)
		random.shuffle(self.poses)
		self.path_size = len(self.paths)
		self.pose_size = len(self.poses)
		self.mode = opt.a_mode
		self.path_length = opt.path_length
		self.size = self.path_size + self.pose_size


	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains A and B
			A(tensor) -- input path (a vector of key poses)
			B(tensor) -- groundtruth path (a vector of poses)
		"""
		if index < self.path_size:
			current_path = self.paths[index]
			B = scipy.io.loadmat(current_path)['data']
			A = B.copy()

			steps = len(B)
			if self.path_length != steps:
				raise('path length not matching in data and opt')

			if self.mode == 'zeros':
				A[1:-1] = 0.0
			elif self.mode == 'ones':
				A[1:-1] = 1.0
			elif self.mode == 'linear':
				BA = np.array([slerp(B[0], B[-1], t) for t in np.linspace(0, 1, steps)])
				A[1:-1] = BA[1:-1]
			else:
				raise NotImplementedError('mode [%s] is not implemented' % self.mode)

			B = torch.tensor(np.ravel(B))
			A = torch.tensor(np.ravel(A))
			return {'A': A, 'B': B}

		else:
			source_pose = self.poses[randrange(self.pose_size)]
			target_pose = self.poses[randrange(self.pose_size)]
			A = np.zeros(self.path_length, len(source_pose))
			A[0] = source_pose
			A[-1] = target_pose
			B = None

			if self.mode == 'zeros':
				A[1:-1] = 0.0
			elif self.mode == 'ones':
				A[1:-1] = 1.0
			elif self.mode == 'linear':
				BA = np.array([slerp(A[0], A[-1], t) for t in np.linspace(0, 1, steps)])
				A[1:-1] = BA[1:-1]
			else:
				raise NotImplementedError('mode [%s] is not implemented' % self.mode)

			A = torch.tensor(np.ravel(A))
			return {'A': A, 'B': A}
