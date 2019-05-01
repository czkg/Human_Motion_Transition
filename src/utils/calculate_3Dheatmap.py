import os
import numpy as np
from numpy import genfromtxt
from shutil import rmtree
from glob import glob
import torch
import time
import scipy.io


def drawGaussian3D(pt, dim_heatmap = 64, sigma = 0.2):
	x0, y0, z0 = pt
	g = np.zeros((dim_heatmap, dim_heatmap, dim_heatmap))

	z = np.linspace(-1, 1, dim_heatmap)
	y = np.linspace(-1, 1, dim_heatmap)
	# in this case we have dim_heatmap[0] == dim_heatmap[1] == 64
	x = y[:, np.newaxis]

	g_xy = (1. / 2 * np.pi * sigma ** 2) * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) * (1. / sigma) ** 2)
	g_z = (1. / 2 * np.pi * sigma ** 2) * np.exp(-0.5 * (z - z0) ** 2 * (1. / sigma) ** 2)

	for i,val in enumerate(z):
		g[i] = g_xy * g_z[i]
		
	gmin = g.min()
	gmax = g.max()
	g = (g - gmin) / (gmax - gmin)

	#indices = g < thres
	#g[indices] = 0.

	return g

def drawHeatmap3D(pts, dim_heatmap = 64, sigma = 0.2):
	num_joints = pts.shape[0]
	heatmap = np.zeros((num_joints, dim_heatmap, dim_heatmap, dim_heatmap), dtype = np.float32)

	for i in range(num_joints):
		heatmap[i] = drawGaussian3D(pts[i], dim_heatmap, sigma)

	heatmap = np.reshape(heatmap, (num_joints * dim_heatmap * dim_heatmap * dim_heatmap,))

	return heatmap

def calculate_3Dheatmap(pts, dim_heatmap, sigma):
	heatmaps = drawHeatmap3D(pts, dim_heatmap, sigma)
	return heatmaps


# if __name__ == '__main__':
# 	if os.path.exists(output_basename):
# 		rmtree(output_basename)

# 	os.makedirs(output_basename)
# 	for s in subs:
# 		os.makedirs(output_basename + s)

# 	print('Converting...')

# 	for idx, s in enumerate(subs):
# 		print('generating', s + ':')
# 		act_list = glob(dataset + s + '/*/')
# 		for act in act_list:
# 			action = act.split('/')[-2]
# 			print('generating', s + '-' + action)
# 			os.makedirs(output_basename + s + '/' + action + '/')
# 			file_list = glob(dataset + s + '/' + action + '/*.csv')
# 			for f in file_list:
# 				filename = f.split('/')[-1][:-4]
# 				pts = genfromtxt(f, delimiter = ' ')
# 				#start = time.time()
# 				#print('start cal...')
# 				heatmaps = drawHeatmap3D(pts, 0.2)
# 				#end = time.time()
# 				#print('cal time:', end-start)
# 				filename = output_basename + s + '/' + action + '/' + filename + '.mat'
# 				#start = time.time()
# 				scipy.io.savemat(filename, {'heatmaps': heatmaps})
# 				#end = time.time()
# 				#print('write time:', end-start)

# 	print('Done!')
