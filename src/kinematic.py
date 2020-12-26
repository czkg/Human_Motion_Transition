import numpy as np
from smpl_webuser.serialization import load_model
from shutil import rmtree
import os
import pickle
import argparse

neutral_model = '../models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
reference = '/home/cz/cs/PG19/src/minimal_ik/model/00018_body.pkl'


def output_mesh(path, verts, faces):
    with open(path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, help='input path')
	parser.add_argument('--output', type=str, help='output path')
	parser.add_argument('--mode', type=str, help='single or sequence')
	args = parser.parse_args()
	input_path = args.input
	output_path = args.output
	mode = args.mode

	# set pose and shape
	with open(reference, 'rb') as f:
		data = pickle.load(f, encoding='latin1')
	betas = data['betas']

	# set smpl model
	smpl = load_model(neutral_model)
	smpl.betas = betas

	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)

	files = os.listdir(input_path)
	for f in files:
		f_name = f.split('/')[-1][:-4]
		pose = np.load(os.path.join(input_path, f))
		if mode == 'sequence':
			path_length = pose.shape[0]
			for i in range(path_length):
				#smpl.pose[:] = np.squeeze(pose[i], axis=0)
				smpl.pose[:] = pose[i]

				verts = np.float32(smpl.r)
				faces = np.int32(smpl.f)

				out_file = os.path.join(output_path, f_name + '_' + str(i+1) + '.obj')
				output_mesh(out_file, verts, faces)
		elif mode == 'single':
			smpl.pose[:] = pose
			verts = np.float32(smpl.r)
			faces = np.int32(smpl.f)
			out_file = os.path.join(output_path, f_name + '.obj')
			output_mesh(out_file, verts, faces)
		else:
			raise Exception('Invalid mode!')




