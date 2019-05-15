from data.base_dataset import BaseDataset
import numpy as np
import os
import scipy.io
import torch
import utils

# !TO DO!add modes later
#modes = ['zeros', 'linear', 'geodesic']


class AlignedPathDataset(BaseDataset):
	""" This dataset class can load a set of paths(data) specified by the file path --dataroot/path/to/data
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

		self.mode = opt.a_mode
		if self.mode is not in modes:
			raise('invalid mode for A!')



	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains A and BSSSS
			A(tensor) -- input path (a vector of key poses)
			B(tensor) -- groundtruth path (a vector of poses)
		"""

		current_path = self.paths[index]
		B = scipy.io.loadmat(current_path)['data']
		A = B.copy()

		if self.mode == 'zeros':
			A[1:-1] = 0.0
		elif self.mode == 'linear':
			steps = len(B)
			BA = np.array([utils.slerp(B[0], B[-1], t) for t in np.linspace(0, 1, steps)])
			A[1:-1] = BA[1:-1]
		else:
			raise NotImplementedError('mode [%s] is not implemented' % self.mode)

		B = torch.tensor(np.ravel(B))
		A = torch.tensor(np.ravel(A))
		return {'A': A, 'B': B}

	def __len__(self):
		""" Return the total number of paths in the dataset
		"""
		return len(self.data)