from data.base_dataset import BaseDataset
import numpy as np
import os
import scipy.io
import torch
from utils.utils import slerp
import pickle



class PathDataset(BaseDataset):
	""" This dataset class can load path(data) specified by the file path --dataroot/path/to/data
	"""

	def __init__(self, opt):
		""" Initialize this dataset class

		Parameters:
			opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""

		BaseDataset.__init__(self, opt)
		self.paths = []
		self.names = []
		files = os.listdir(self.root)
		for f in files:
			path = os.path.join(self.root, f)
			self.paths.append(path)
			self.names.append(f)

		self.path_length = opt.path_length



	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains A and BSSSS
			A(tensor) -- input path (a vector of key poses)
			B(tensor) -- groundtruth path (a vector of poses)
		"""

		file = self.paths[index]
		file_name = self.names[index]
		with open(file, 'rb') as f:
			raw_data = pickle.load(f, encoding='latin1')
		data = raw_data['data']
		gt = raw_data['gt']
		A = {'data': torch.tensor(data), 'gt': torch.tensor(gt), 'info': file_name}

		return A

	def __len__(self):
		""" Return the total number of paths in the dataset
		"""
		return len(self.paths)