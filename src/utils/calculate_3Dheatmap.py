import os
import numpy as np
from numpy import genfromtxt
from shutil import rmtree
from glob import glob
import torch
import time
import scipy.io

thres = 1e-4


def drawGaussian3D(pt, dim_heatmap = 64, sigma = 0.05):
	x0, y0, z0 = pt
	#g = np.zeros((dim_heatmap, dim_heatmap, dim_heatmap))

	z = np.linspace(-1, 1, dim_heatmap)
	y = np.linspace(-1, 1, dim_heatmap)
	# in this case we have dim_heatmap[0] == dim_heatmap[1] == 64
	x = y[:, np.newaxis]

	gxy = (1. / 2 * np.pi * sigma ** 2) * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) * (1. / sigma) ** 2)
	gz = (1. / 2 * np.pi * sigma ** 2) * np.exp(-0.5 * (z - z0) ** 2 * (1. / sigma) ** 2)

		
	gxy_min = gxy.min()
	gxy_max = gxy.max()
	g_xy = (gxy - gxy_min) / (gxy_max - gxy_min)
	under_thres_xy_indices = g_xy < thres
	g_xy[under_thres_xy_indices] = 0.0

	gz_min = gz.min()
	gz_max = gz.max()
	g_z = (gz - gz_min) / (gz_max - gz_min)
	under_thres_z_indices = g_z < thres
	g_z[under_thres_z_indices] = 0.0


	return g_xy, g_z

def drawHeatmap3D(pts, dim_heatmap = 64, sigma = 0.05):
	num_joints = pts.shape[0]
	heatmap_xy = np.zeros((num_joints, dim_heatmap, dim_heatmap), dtype = np.float32)
	heatmap_z = np.zeros((num_joints, dim_heatmap), dtype = np.float32)

	for i in range(num_joints):
		heatmap_xy[i], heatmap_z[i] = drawGaussian3D(pts[i], dim_heatmap, sigma)

	heatmap_xy = np.resize(heatmap_xy, (num_joints, dim_heatmap * dim_heatmap))

	return heatmap_xy, heatmap_z

def calculate_3Dheatmap(pts, dim_heatmap, sigma):
	heatmap_xy, heatmap_z = drawHeatmap3D(pts, dim_heatmap, sigma)
	heatmap = np.concatenate((heatmap_xy, heatmap_z), axis = 1)
	return np.ravel(heatmap)


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
