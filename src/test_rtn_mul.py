import os
import scipy.io
from glob import glob
import tqdm
from shutil import rmtree
import torch
import numpy as np
import time
import pickle
import argparses

from models import networks
	
def convert2heatmap(seq, heatmap_dim, sigma):
	# [n_joints, 3]
	heatmaps = []
	if len(seq.shape) == 2:
		seq = seq[np.newaxis, ...]
	for i in range(seq.shape[0]):
		pose = seq[i]
		heatmap = calculate_3Dheatmap(pose, heatmap_dim, sigma)
		heatmaps.append(heatmap)

	return np.asarray(heatmaps)


def post_process(transition, gt, past_len):
	target_est = transition[-1]
	target_gt = gt[-1]

	e = target_gt - target_est
	d = 30 - past_len - 1
	for t, x in enumerate(transition):
		w = 1. - (float(d-t) / float(d))
		transition[t] = transition[t] + w*e
	return transition

def slerp(p0, p1, t):
    """
    Spherical linear interpolation
    """
    omega = torch.acos(torch.dot(torch.squeeze(p0/torch.norm(p0)),
                             torch.squeeze(p1/torch.norm(p1))))
    so = torch.sin(omega)
    return torch.sin(1.0 - t) * omega / so * p0 + torch.sin(t * omega) / so * p1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--past_seq', type=str, help='input path as past')
	parser.add_argument('--output_path', type=str, help='output path')
	parser.add_argument('--targets', type=str, help='multiple targets', nargs='*')
	args = vars(parser.parse_args())

	past_seq = args.past_seq
	targets = args.targets
	output_path = args.output_path
	if os.path.exists(output_path):
		rmtree(output_path)
	os.makedirs(output_path)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#load fvae model
	fvae_model = networks.VAEDMP(87360, 32, 32, 128, 64, 32, False, device)
	fvae_model.load_state_dict(torch.load(fvae_path))
	fvae_model.eval()

	#load rtn model
	rtn_model = networks.RTN(32, 512)
	rtn_model.load_state_dict(torch.load(rtn_path))
	rtn_model.eval()

	num_targets = len(targets)
	#load past seq
	past_len = 10
	with open(past_seq, 'rb') as f:
		data = pickle.load(f, encoding='latin1')
	past = convert2heatmap(data['X'][:past_len], 64, 0.05)
	past = past.reshape(seq.shape[0], -1)
	past = past.unsqueeze(0)

	fvae_model.set_input(past)
	_, _, _, zs, _ = fvae_model.inference()

	trans = []
	pasts = []
	pasts.append(zs[0][:past_len])
	for t in targets:
		with open(t, 'rb') as f:
			data = pickle.load(f, encoding='latin1')
		tt = t['X']
		tt = tt.unsqueeze(0)

		# compute past
		rtn_model.set_input(zs)
		transition,gt,_ = rtn_model.inference()
		transition = transition[0][past_len:]

		# compute target
		fvae_model.set_input(tt)
		_, _, _, zt, _ = fvae_model.inference()
		zt = zt[0]

		end_idx = torch.argmin(torch.norm((transition - zt), dim=-1))
		min_dist = torch.min(torch.norm((transition - zt), dim=-1))
		max_dist = torch.max(torch.norm((transition[1:] - transition[:-1]), dim=-1))

		if min_dist <= max_dist*1.1:
			transition = transition[:end_idx+1]
		else:
			steps = floor(min_dist / max_dist)
			addon = ([slerp(transition[end_idx], zt, i) for i in np.linspace(0, 1, steps)])
			addon = torch.stack(addon)
			transition = torch.cat((transition, addon), 0)
		trans.append(transition)

		zs = transition[-past_len+1:]
		zs = torch.cat((zs, zt), 0)
		add1 = torch.zeros_like(zs)
		add2 = torch.zeros_like(zs)
		add = torch.cat((add1, add2), 0)
		zs = torch.cat((zs, add), 0)


	files = glob(input_path + '*.pkl')