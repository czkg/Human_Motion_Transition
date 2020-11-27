from minimal_ik.models import *
from minimal_ik.armatures import *
from minimal_ik.solver import *
from minimal_ik.config import *
from minimal_ik.utils import *

import numpy as np
import argparse
import os
from glob import glob
from shutil import rmtree

import matplotlib as mpl
from mayavi import mlab
import matplotlib.pyplot as plt

'''
LSP-14 to smpl
|index     |  joint name      |    corresponding SMPL joint ids   |
|----------|:----------------:|---- -----------------------------:|
| 0        |  Right hip       |2                                  |
| 1        |  Right knee      |5                                  |
| 2        |  Right foot      |8                                  |
| 3        |  Left hip        |1                                  |
| 4        |  Left knee       |4                                  |
| 5        |  Left foot       |7                                  |
| 6        |  Thorax(neck)    |12                                 |
| 7        |  Head            |vertex 411 (see line 233:fit_3d.py)|
| 8        |  Left shoulder   |16                                 |
| 9        |  Left elbow      |18                                 |
| 10       |  Left wrist      |20                                 |
| 11       |  Right shoulder  |17                                 |
| 12       |  Right elbow     |19                                 |
| 13       |  Right wrist     |21                                 |
'''

'''
H36M-17 to smpl
|index     |  joint name      |    corresponding SMPL joint ids   |
|----------|:----------------:|---- -----------------------------:|
| 0        |  Pelvis          |0                                  |
| 1        |  Right hip       |2                                  |
| 2        |  Right knee      |5                                  |
| 3        |  Right ankle     |8                                  |
| 4        |  Left hip        |1                                  |
| 5        |  Left knee       |4                                  |
| 6        |  Left ankle      |7                                  |
| 7        |  Spine           |6                                  |
| 8        |  neck            |15                                 |
| 9        |  Head            |vertex 411                         |
| 10       |  Thorax          |12                                 |
| 11       |  Left shoulder   |16                                 |
| 12       |  Left elbow      |18                                 |
| 13       |  Left wrist      |20                                 |
| 14       |  Right shoulder  |17                                 |
| 15       |  Right elbow     |19                                 |
| 16       |  Right wrist     |21                                 | 
'''

H36M2SMPL = {
	0: 0,
	1: 2,
	2: 5,
	3: 8,
	4: 1,
	5: 4,
	6: 7,
	7: 6,
	8: 15,
	10:12,
	11:16,
	12:18,
	13:20,
	14:17,
	15:19,
	16:21 
}

SMPL2H36M = {
	0: 0,
	1: 4,
	2: 1,
	3: -1,
	4: 5,
	5: 2,
	6: 7,
	7: 6,
	8: 3,
	9: -1,
	10: -1,
	11: -1,
	12: 10,
	13: -1,
	14: -1,
	15: 8,
	16: 11,
	17: 14,
	18: 12,
	19: 15,
	20: 13,
	21: 16,
	22: -1,
	23: -1
}

H36M22SMPL = {
	2: 5,
	3: 8,
	5: 4,
	6: 7,
	8: 15,
	11:16,
	12:18,
	13:20,
	14:17,
	15:19,
	16:21
}

SMPLParents = [0,0,0,0,1,2,3,4,5,6,7,8,
			   9,9,9,12,13,14,16,17,18,19,20,21]
H36MParents = [0,0,1,2,0,4,5,0,10,8,7,10,11,12,10,14,15]

smpl_limbs_roots = [7, 8, 20, 21]
smpl_limbs = [10, 11, 22, 23]

smpl_redundant_joints = [3, 9, 10, 11, 13, 14, 22, 23]

rest_pose_file = './minimal_ik/model/00018_body.pkl'


H36M_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

def plot_skeleton(pose, type):
	"""Plot skeleton. Type 0: smpl; Type 1: h36m 
	"""
	pose = pose.transpose()
	if type == 0:
		kin = np.array([[0,1], [0,2], [0,3], [1,4], [2,5], [3,6], [4,7], [5,8], [6,9], 
						[7,10], [8,11], [9,12], [9,13], [9,14], [12,15], [13,16], [14,17],
						[16,18], [17,19], [18,20], [19,21], [20,22], [21,23]])
	elif type == 1:
		kin = np.array([[0, 7], [7, 8], [8, 9], [10, 9], [8, 11], [11, 12], [12, 13], 
						[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]])
	else:
		raise Exception('Unexpect type!')

	mpl.rcParams['legend.fontsize'] = 10

	fig = mlab.figure(bgcolor=(1,1,1))
	ax_ranges = [-1, 1, -1, 1, -1, 1]
	#ax = fig.gca(projection='3d')
	#ax.view_init(azim=-90, elev=15)
	mlab.view(azimuth=-90, elevation=10)
	if type == 0:
		colors = np.ones([24, 3])
	elif type == 1:
		colors = np.ones([17,3])

	for idx, link in enumerate(kin):
		mlab.plot3d(pose[0, link], pose[2, link], pose[1, link], color=(colors[idx][0],colors[idx][1],colors[idx][2]), line_width=2.0, figure=fig)

	mlab.show()

def load_pose(file):
	with open(file, 'rb') as f:
		data = pickle.load(f, encoding='latin1')
	return data['pose']

def transform(source, target):
	root_t = target[0]

	r_hip2knee_t = target[2] - target[5]
	r_knee2ankle_t = target[5] - target[8]
	l_hip2knee_t = target[1] - target[4]
	l_knee2ankle_t = target[4] - target[7]

	r_shoulder2elbow_t = target[17] - target[19]
	r_elbow2wrist_t = target[19] - target[21]
	l_shoulder2elbow_t = target[16] - target[18]
	l_elbow2wrist_t = target[18] - target[20]

	root_s = source[0]

	r_hip2knee_s = source[1] - source[2]
	r_knee2ankle_s = source[2] - source[3]
	l_hip2knee_s = source[4] - source[5]
	l_knee2ankle_s = source[5] - source[6]

	r_shoulder2elbow_s = source[14] - source[15]
	r_elbow2wrist_s = source[15] - source[16]
	l_shoulder2elbow_s = source[11] - source[12]
	l_elbow2wrist_s = source[12] - source[13]

	print('---------')
	print(np.linalg.norm(r_hip2knee_s)/np.linalg.norm(r_hip2knee_t))
	print(np.linalg.norm(r_knee2ankle_s)/np.linalg.norm(r_knee2ankle_t))
	print(np.linalg.norm(l_hip2knee_s)/np.linalg.norm(l_hip2knee_t))
	print(np.linalg.norm(l_knee2ankle_s)/np.linalg.norm(l_knee2ankle_t))
	print(np.linalg.norm(r_shoulder2elbow_s)/np.linalg.norm(r_shoulder2elbow_t))
	print(np.linalg.norm(r_elbow2wrist_s)/np.linalg.norm(r_elbow2wrist_t))
	print(np.linalg.norm(l_shoulder2elbow_s)/np.linalg.norm(l_shoulder2elbow_t))
	print(np.linalg.norm(l_elbow2wrist_s)/np.linalg.norm(l_elbow2wrist_t))
	print('---------')

# def translate(source, offset):
# 	return source+offset

# def skew(vec):
# 	res = np.zeros((3, 3))
# 	res[0][1] = -vec[2]
# 	res[0][2] = vec[1]
# 	res[1][0] = vec[2]
# 	res[1][2] = -vec[0]
# 	res[2][0] = -vec[1]
# 	res[2][1] = vec[0]

# 	return res

# def get_R(vec_s, vec_t):
# 	v = np.cross(vec_s, vec_t)
# 	s = np.linalg.norm(v)
# 	c = np.dot(vec_s, vec_t)
# 	I = np.eye(3)
# 	vx = skew(v)

# 	R = I + vx + np.dot(vx, vx) * (1. - c) / s**2
# 	return R

# def computeBoneLength(points, ty):
# 	boneLength = []
# 	if ty == 0:
# 		for i in range(24):
# 			bone = np.linalg.norm(points[i] - points[SMPLParents[i]])
# 			boneLength.append(bone)
# 	else:
# 		for i in range(17):
# 			bone = np.linalg.norm(points[i] - points[H36MParents[i]])
# 			boneLength.append(bone)
# 	return np.asarray(boneLength)

# def computeRelativeOffUnit(points, ty):
# 	offs = []
# 	boneLength = computeBoneLength(points, ty)
# 	if ty == 0:
# 		for i in range(24):
# 			off = points[i] - points[SMPLParents[i]]
# 			offs.append(off)
# 	else:
# 		for i in range(17):
# 			off = points[i] - points[H36MParents[i]]
# 			offs.append(off)
		
# 	offs = np.asarray(offs)
# 	offUnit = [offs[i] / boneLength[i] for i in range(len(boneLength))]
# 	offUnit[0] = np.zeros((3))
# 	offUnit = np.asarray(offUnit)

# 	return offUnit


if __name__ == '__main__':
	n_pose = 23 * 3 # degree of freedom, (n_joints - 1) * 3
	n_shape = 10
	pose_pca = np.random.uniform(-0.2, 0.2, size=n_pose)
	pose_glb = np.zeros((1, 3))
	shape = np.random.normal(size=n_shape)
	mesh = KinematicModel(SMPL_MODEL_PATH, SMPLArmature)

	wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
	solver = Solver(verbose=True)

	h36m_path = H36M_PATH
	output_path = OUTPUT_PATH

	#extract pose parameters from rest pose
	rest_pose = load_smpl_pose(rest_pose_file, OFFICIAL_SMPL_PATH)

	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)
	for s in H36M_SUBJECTS:
		os.makedirs(os.path.join(output_path, s))

	for idx, s in enumerate(H36M_SUBJECTS):
		acts = os.listdir(os.path.join(h36m_path, s))
		for act in acts:
			os.makedirs(os.path.join(output_path, s, act))
			file_list = glob(os.path.join(h36m_path, s, act) + '/*.csv')
			for f in file_list:
				filename = f.split('/')[-1][:-4]

				_, smpl_points = mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb, shape=shape)
				#boneLength = computeBoneLength(smpl_points, 0)
				#np.savetxt('est.csv', smpl_points, delimiter=',')

				# smpl_points_mask = np.ones(smpl_points.shape[0], dtype=bool)
				# smpl_points_mask[smpl_redundant_joints] = False

				rest_pose_mask = np.ones(rest_pose.shape[0], dtype=bool)
				#rest_pose_mask[smpl_redundant_joints] = False

				# vec1 = smpl_points[6] - smpl_points[0]
				# vec1 = vec1 / np.linalg.norm(vec1)

				# h36m_points = np.genfromtxt(f, delimiter=' ')
				# vec2 = h36m_points[7] - h36m_points[0]
				# vec2 = vec2 / np.linalg.norm(vec2)

				# #apply scale

				# #apply rotation
				# R = get_R(vec2, vec1)
				# h36m_points_R = [np.matmul(R,p) for p in h36m_points]
				# h36m_points = np.vstack(h36m_points_R)

				# #apply translation
				# h36m_points = translate(h36m_points, smpl_points[0] - h36m_points[0])

				# #compute new offsets based on their parents
				# offUnitH36M = computeRelativeOffUnit(h36m_points, 1)
				# offUnitSMPL = computeRelativeOffUnit(smpl_points, 0)

				# plot_skeleton(smpl_points, 0)

				# for smpl_id, h36m_id in SMPL2H36M.items():
				# 	if h36m_id == -1:
				# 		smpl_points[smpl_id] = smpl_points[SMPLParents[smpl_id]] + boneLength[smpl_id]*offUnitSMPL[smpl_id]
				# 	else:
				# 		smpl_points[smpl_id] = smpl_points[SMPLParents[smpl_id]] + boneLength[smpl_id]*offUnitH36M[h36m_id]

				rest_pose = registrate(rest_pose, smpl_points)
				#plot_skeleton(h36m_points, 1)
				plot_skeleton(smpl_points, 0)
				plot_skeleton(rest_pose, 0)
				print(rest_pose_mask.shape)

				params_est = solver.solve(wrapper, rest_pose, rest_pose_mask)
				shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)
				mesh.set_params(pose_pca=pose_pca_est)
				mesh.save_obj('./est.obj')
				break
			break
		break

	
