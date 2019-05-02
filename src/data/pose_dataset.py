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
		self.dict = {}
		subs = os.listdir(self.root)
		for s in subs:
			acts = os.listdir(self.root + '/' + s)
			self.dict[s] = acts



	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains data and path
		"""
		current_path = 
		