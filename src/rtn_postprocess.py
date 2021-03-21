import numpy as np
import argparse
import pickle
import os
from shutil import rmtree

from heatmap2pose import heatmap2pose


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, help='input path')
	parser.add_argument('--output_path', type=str, help='output path')
	parser.add_argument('--heatmap', dest='heatmap', action='store_true')
	parser.add_argument('--no-heatmap', dest='heatmap', action='store_false')
	parser.set_defaults(heatmap=True)
	args = vars(parser.parse_args())

	input_path = args['input_path']
	output_path = args['output_path']
	heatmap = args['heatmap']

	past_len = 10

	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)

	files = os.listdir(input_path)
	for f in files:
		with open(os.path.join(input_path, f), 'rb') as ff:
			data = pickle.load(ff, encoding='latin1')
		est = data['data']
		gt = data['gt']

		past = gt[0:past_len]
		transition = est[past_len:]

		if heatmap:
			past = [heatmap2pose(p_data, 64, 21) for p_data in past]
			transition = [heatmap2pose(t_data, 64, 21) for t_data in transition]
			gt = [heatmap2pose(g_data, 64, 21) for g_data in gt]
			est = [heatmap2pose(e_data, 64, 21) for e_data in est]
			past = np.asarray(past)
			transition = np.asarray(transition)
			gt = np.asarray(gt)
			est = np.asarray(est)

		target_est = transition[-1]
		target_gt = gt[-1]

		e = target_gt - target_est
		d = est.shape[0] - past_len - 1
		for t, x in enumerate(transition):
			w = 1. - (float(d-t) / float(d))
			transition[t] = transition[t] + w*e

		past_est = est[past_len-1]

		new_data = {'past': past, 'transition': transition, 'est': est, 'gt': gt}

		with open(os.path.join(output_path, f), 'wb') as ff:
			pickle.dump(new_data, ff, protocol=pickle.HIGHEST_PROTOCOL)




