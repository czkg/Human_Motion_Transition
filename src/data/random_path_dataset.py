from data.base_dataset import BaseDataset
import numpy as np
import os
import scipy.io
import torch
from utils.utils import slerp
import random
from random import randrange

class RandomPathDataset(BaseDataset):
	def __init__(self, opt):
		""" Initialize this dataset class

		Parameters:
			opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		BaseDataset.__init__(self, opt)
		self.randomroot = opt.randomroot
		self.poses = []
		pose_files = os.listdir(self.randomroot)
		for f in pose_files:
			pose = os.path.join(self.randomroot, f)
			self.poses.append(pose)
		random.shuffle(self.poses)
		self.pose_size = len(self.poses)
		self.mode = opt.a_mode
		self.path_length = opt.path_length



	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains A and B
			A(tensor) -- input path (a vector of key poses)
			B(tensor) -- groundtruth path (a vector of poses)
		"""
		start_pose = self.poses[randrange(self.pose_size)]
		end_pose = self.poses[randrange(self.pose_size)]
		g = np.zeros(self.path_length, len(start_pose))
		g[0] = start_pose
		g[-1] = end_pose

		if self.mode == 'zeros':
			g[1:-1] = 0.0
		elif self.mode == 'ones':
			g[1:-1] = 1.0
		elif self.mode == 'linear':
			path = np.array([slerp(g[0], g[-1], t) for t in np.linspace(0, 1, steps)])
			g[1:-1] = path[1:-1]
		else:
			raise NotImplementedError('mode [%s] is not implemented' % self.mode)

		d = g.copy()
		A = torch.tensor(np.ravel(A))
		B = torch.tensor(np.ravel(B))
		return {'A': A, 'B': B}
