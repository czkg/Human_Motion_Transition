import os
from data.base_dataset import BaseDataset
from utils.calculate_3Dheatmap import calculate_3Dheatmap
from numpy import genfromtxt


class PoseDataset(BaseDataset):
	""" This dataset calss can load poses specified by the file path --dataroot/path/to/data
	"""

	def __init__(self, opt):
		""" Initialize this dataset class

		Parameters:
			opt (option calss) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""

		BaseDataset.__init__(self, opt)
		self.paths = []
		subs = os.listdir(self.root)
		for s in subs:
			acts = os.listdir(os.path.join(self.root, s))
			for a in acts:
				basepath = os.path.join(self.root, s, a)
				filenames = os.listdir(basepath)
				for f in filenames:
					path = os.path.join(basepath, f)
					self.paths.append(path)

		self.dim_heatmap = opt.dim_heatmap
		self.sigma = opt.sigma



	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains data and path
		"""
		current_path = self.paths[index]
		pts = genfromtxt(current_path, delimiter = ' ')
		heatmap = calculate_3Dheatmap(pts, self.dim_heatmap, self.sigma)

		return {'heatmap': heatmap}

	def __len__(self):
		return len(self.paths)
		