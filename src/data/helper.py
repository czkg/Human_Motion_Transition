import torch.utils.data as data
import scipy.io
import h5py
import os
import os.path


def make_dataset(dir, max_dataset_size=float('inf')):
	paths = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			path = os.path.join(root, fname)
			paths.append(path)

	return paths[:min(max_dataset_size, len(paths))]
