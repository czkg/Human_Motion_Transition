from data.base_dataset import BaseDataset
from data.helper import make_dataset
from models import networks
from numpy import genfromtxt
from utils.calculate_3Dheatmap import calculate_3Dheatmap


DATA_EXTENSIONS = ['mat', 'csv', 'json', 'hdf5']
subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
acts = ['Direction', 'Directions 1', 'Discussion', 'Discusion 1', 'Eating', 'Eating 2', 'Greeting',
		   'Greating 1', 'Phoning', 'Phoning 1', 'Photo', 'Photo 1', 'Posing', 'Posing 1', 'Purchases',
		   'Purchases 1', 'Sitting 1', 'Sitting 2', 'SittingDown', 'SittingDown 2', 'Smoking', 'Smoking 1',
		   'Waiting', 'Waiting 1', 'WalkDog', 'WalkDog 1', 'Walking', 'Walking 1', 'WalkTogether', 'WalkTogether 1']


class SinglePathDataset(BaseDataset):
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
			self.vae = network.VAEModel(opt.dim_heatmap, opt.z_dim, opt.pca_dim)
			self.vae.load_state_dict(torch.load(opt.vae_path))
		self.path = opt.datapath

		self.data = []
		self.metadata = []
		count = 0
		with open(self.path) as f:
			rawdata = json.load(f)
		for s in subs:
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

		Returns a dictionary that contains A and A_paths
			A(tensor) -- a path in one domin
			A_paths(str) -- the path of the data(path)
		"""

		this_path_data = self.data[index]
		this_path_folder = self.metadata[index]
		A = path2data(this_path_data, this_path_folder)
		return {'A': A}

	def __len__(self):
		""" Return the total number of paths in the dataset
		"""
		return len(self.A_paths)


	def path2data(self, path_data, path_folder):
		path_len = len(path_data)

		data = []
		#loop through every point in the path
		for i in range(path_len):
			current_path = self.datapath + '/' + path_folder + '/' + str(path_data[i]) + '.csv'
			pts = genfromtxt(current_path, delimiter = ' ')
			heatmap = calculate_3Dheatmap(pts, self.dim_heatmap, self.sigma)
			_, z = self.vae(heatmap)
			data.append(z)
		return np.ravel(data)


	def load_path_data(file_path):
	ext = file_path[-3:]
	assert ext in DATA_EXTENSIONS, '%s format is not supported' % ext

	if ext == 'hdf5':
		with h5py.File(file_path) as f:
			data = f['data'].value
	elif ext == 'mat':
		data = scipy.io.loadmat(file_path)
	# we are now using json
	elif ext == 'json':
		with open(file_path) as f:
			data = json.load(f)['data']
	elif ext == 'csv':
		data = genfromtxt(file_path, delimiter=' ')

	return data