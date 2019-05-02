from data.base_dataset import BaseDataset
from models import networks
from numpy import genfromtxt
from utils.calculate_3Dheatmap import calculate_3Dheatmap

# !TO DO!add modes later
mode = []


class AlignedPathDataset(BaseDataset):
	""" This dataset class can load a set of paths(data) specified by the file path --dataroot/path/to/data
	"""

	def __init__(self, opt):
		""" Initialize this dataset class

		Parameters:
			opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""

		BaseDataset.__init__(self, opt)
		#self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))

		self.with_vae = opt.with_vae
		if self.with_vae:
			self.vae = network.VAE(opt.dim_heatmap, opt.z_dim, opt.pca_dim)
			self.vae.load_state_dict(torch.load(opt.vae_path))
		path = opt.datapath      #path to the json file

		self.data = []
		self.metadata = []
		count = 0
		with open(path) as f:
			rawdata = json.load(f)
		subs = list(rawdata.keys())

		for s in subs:
			acts = list(rawdata[s].keys())
			for a in acts:
				number = len(rawdata[s][a])
				for i in range(number):
					self.data.append(rawdata[s][a][i])
					self.metadata.extend(s + '/' + a)

		self.dim_heatmap = opt.dim_heatmap
		self.sigma = opt.sigma


	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains A and BSSSS
			A(tensor) -- input path (a vector of key poses)
			B(tensor) -- groundtruth path (a vector of poses)
		"""

		this_path_data = self.data[index]
		this_path_folder = self.metadata[index]
		A, B = path2data(this_path_data, this_path_folder)
		return {'A': A, 'B': B}

	def __len__(self):
		""" Return the total number of paths in the dataset
		"""
		return len(self.A_paths)


	def path2data(self, path_data, path_folder):
		path_len = len(path_data)

		B = []
		A = []
		#loop through every point in the path
		for i in range(path_len):
			current_path = self.root + '/' + path_folder + '/' + str(path_data[i]) + '.csv'   #dataroot: path to csv files
			pts = genfromtxt(current_path, delimiter = ' ')
			heatmap = calculate_3Dheatmap(pts, self.dim_heatmap, self.sigma)
			_, z = self.vae(heatmap)
			B.append(z)
		A = B
		A[1:-1] = 0.0
		return np.ravel(A), np.ravel(B)