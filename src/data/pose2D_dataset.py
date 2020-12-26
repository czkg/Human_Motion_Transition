import os
from data.base_dataset import BaseDataset
from utils.calculate_3Dheatmap import calculate_3Dheatmap
import scipy.io
import torch
import numpy as np


class Pose2DDataset(BaseDataset):
	""" This dataset class can load poses specified by the file path --dataroot/path/to/data
	"""

	def __init__(self, opt):
		""" Initialize this dataset class

		Parameters:
			opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""

		BaseDataset.__init__(self, opt)
		self.paths = []
		files = os.listdir(self.root)
		for f in files:
			path = os.path.join(self.root, f)
			self.paths.append(path)


	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains data and path
		"""
		current_path = self.paths[index]
		pose = np.load(current_path)
		pose = pose.reshape((1,24,3))

		return torch.tensor(pose)

	def __len__(self):
		return len(self.paths)
		