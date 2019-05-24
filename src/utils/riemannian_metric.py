import numpy as np
from .base_metric import BaseMetric
import torch
from torch.autograd import Variable


class RiemannianMetric(BaseMetric):
	def __init__(self, x, z, opt):
		BaseMetric.__init__(self, opt)

		self.x = x
		self.z = z

	def create_jacob(self):
		self.J = torch.autograd.grad(self.x, self.z, grad_outputs=torch.ones_like(self.x), create_graph=True, retain_graph=True, only_inputs=True)[0]

	def riemannian_distance_along_line(self, z0_id, z1_id):
		"""
		calculate the riemannian distance between two near points in latent space on a straight line

		z0: start point id
		z1: end point id
		n_steps: number of discretization steps of the integral
		"""
		dt = 1.0
		j0 = self.J[z0_id]
		j1 = self.J[z1_id]
		j = torch.stack([j0, j1], dim = 0)
		#derivative of each output dim wrt input
		G = torch.transpose(j, 0, 1) @ j

		L_discrete = torch.sqrt((self.z[z0_id] - self.z[z1_id]) @ G @ (self.z[z0_id] - self.z[z1_id]).view(1, -1).t())
		L_discrete = L_discrete.flatten()

		L = torch.sum(dt * L_discrete)

		det = np.linalg.det(G.data.cpu().numpy())
		MF = np.sqrt(det)

		return L.data.cpu().numpy(), MF

