import numpy as np
import os
from glob import glob
from shutil import rmtree
import argparse
from shutil import copyfile
import random
import math

def split(in_path):
	base_path = in_path.split('/')[:-1]
	base_path = '/'.join(base_path)
	train_path = os.path.join(base_path, 'train')
	test_path = os.path.join(base_path, 'test')

	if os.path.exists(train_path):
		rmtree(train_path)
	os.makedirs(train_path)
	if os.path.exists(test_path):
		rmtree(test_path)
	os.makedirs(test_path)

	files = os.listdir(in_path)
	random.shuffle(files)
	train_len = math.floor(0.8 * len(files))
	train_set = files[:train_len]
	test_set = files[train_len:]

	for f in train_set:
		src = os.path.join(in_path, f)
		dst = os.path.join(train_path, f)
		copyfile(src, dst)

	for f in test_set:
		src = os.path.join(in_path, f)
		dst = os.path.join(test_path, f)
		copyfile(src, dst)

def calculate_minmax(path):
	file_list = os.listdir(path)
	mma = -100
	mmi = 100
	subs = os.listdir(path)
	for s in subs:
		file_list = glob(os.path.join(path, s) + '/*.npy')
		for f in file_list:
			data = np.load(f)
			iid = data.shape[0]
			for i in range(iid):
				pose = data[i]
				mmax = np.amax(pose)
				mmin = np.amin(pose)
				if mmax > mma:
					mma = mmax
				if mmin < mmi:
					mmi = mmin
	mm = []
	mm.append(mmi)
	mm.append(mma)
	with open(minmax_path, 'wb') as ff:
		np.save(ff, mm)
	return mm



def preprocess(in_path, out_path, norm, m_path, step, is_2D):
	# There are sub folders by default
	if norm:
		if m_path == None:
			minmax = calculate_minmax(in_path)
		else:
			minmax = np.load(m_path)
		mmin = minmax[0]
		mmax = minmax[1]

	if os.path.exists(out_path):
		rmtree(out_path)
	os.makedirs(out_path)

	subs = os.listdir(in_path)
	for s in subs:
		print('processing ', str(s))
		file_list = glob(os.path.join(in_path, s) + '/*.npy')
		for f in file_list:
			filename = f.split('/')[-1][:-4]
			data = np.load(f)
			data = data[0::step]
			iid = data.shape[0]
			for i in range(iid):
				pose = data[i]
				if norm:
					pose = (pose - mmin) / (mmax - mmin)
					new_name = filename + '_' + str(i+1) + '_normalized.npy'
				else:
					new_name = filename + '_' + str(i+1) + '.npy'
				if not is_2D:
					pose = pose.reshape(-1)
				
				with open(os.path.join(out_path, new_name), 'wb') as ff:
					np.save(ff, pose)


def postprocess(in_path, out_path, m_path, is_2D):
	# There is no sub folders by default
	if m_path == None:
		raise Exception('m_path cannot be None')
	minmax = np.load(m_path)
	mmin = minmax[0]
	mmax = minmax[1]

	if os.path.exists(out_path):
		rmtree(out_path)
	os.makedirs(out_path)

	file_list = os.listdir(in_path)
	for f in file_list:
		new_name = f[:-15] + '_recon.npy'
		pose = np.load(os.path.join(in_path, f))[0]
		if is_2D:
			pose = pose.reshape(-1)
		pose = mmin + (mmax-mmin)*pose
		with open(os.path.join(out_path, new_name), 'wb') as ff:
			np.save(ff, pose)
		


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, help='input path')
	parser.add_argument('--output', type=str, help='output_path')
	parser.add_argument('--mode', type=str, help='whic mode to use, [pre | post]')
	parser.add_argument('--m_path', type=str, default=None, help='minmax file to load')
	parser.add_argument('--step', type=int, default=1, help='step to use')
	parser.add_argument('--norm', dest='norm', action='store_true')
	parser.add_argument('--no-norm', dest='norm', action='store_false')
	parser.set_defaults(norm=True)
	parser.add_argument('--is_2D', dest='is_2D', action='store_true')
	parser.add_argument('--no-is_2D', dest='is_2D', action='store_false')
	parser.set_defaults(is_2D=False)
	args = parser.parse_args()

	input_path = args.input
	output_path = args.output
	mode = args.mode
	norm = args.norm
	m_path = args.m_path
	step = args.step
	is_2D = args.is_2D

	if mode == 'pre':
		preprocess(input_path, output_path, norm, m_path, step, is_2D)
		split(output_path)
	elif mode == 'post':
		postprocess(input_path, output_path, m_path, is_2D)





