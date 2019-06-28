import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import sys, getopt, os
import pdb
from mayavi import mlab

def main(argv):

	csvfile = argv[0];

	if not os.path.isfile(csvfile):
		print('ERROR! ', csvfile,' is not a valid file location')
		sys.exit()

	try:
		pose3D = genfromtxt(csvfile, delimiter=' ')
		pose3D = pose3D.transpose();

		kin = np.array([[0, 7], [7, 8], [8, 9], [10, 9], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]);
		order = np.array([0,1,2]);

		mpl.rcParams['legend.fontsize'] = 10

		fig = mlab.figure(bgcolor=(1,1,1))
		ax_ranges = [-1, 1, -1, 1, -1, 1]
		#ax = fig.gca(projection='3d')
		#ax.view_init(azim=-90, elev=15)
		mlab.view(azimuth=-90, elevation=10)
		colors = np.empty([17,3])
		#colors
		colors[0] = [0.97,0.66,0.15]
		colors[1] = [0.97,0.66,0.15]
		colors[2] = [1,0.59,0]
		colors[3] = [1,0.59,0]
		colors[4] = [1,0.76,0.03]
		colors[5] = [1,0.76,0.03]
		colors[6] = [1,0.76,0.03]
		colors[7] = [1,0.92,0.23]
		colors[8] = [1,0.92,0.23]
		colors[9] = [1,0.92,0.23]
		colors[10] = [0.8,0.86,0.22]
		colors[11] = [0.8,0.86,0.22]
		colors[12] = [0.8,0.86,0.22]
		colors[13] = [0.55,0.76,0.29]
		colors[14] = [0.55,0.76,0.29]
		colors[15] = [0.55,0.76,0.29]

		for idx, link in enumerate(kin):
			mlab.plot3d(pose3D[0, link], pose3D[2, link], pose3D[1, link], color=(colors[idx][0],colors[idx][1],colors[idx][2]), line_width=2.0, figure=fig)
			# mlab.axes(dis, color=(.7, .7, .7),
   #            	ranges=ax_ranges,
   #            	xlabel='x', ylabel='y', zlabel='z')
			if False:
				dis.actor.property.opacity = 0.5
				fig.scene.renderer.use_depth_peeling = 1
		#ax.legend()
		mlab.show()
		
	
		# ax.set_xlabel('X')
		# ax.set_ylabel('Y')
		# ax.set_zlabel('Z')
		# ax.set_aspect('equal')

		# X = pose3D[0, :]
		# Y = pose3D[1, :]
		# Z = pose3D[2, :]
		# max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

		# mid_x = (X.max()+X.min()) * 0.5
		# mid_y = (Y.max()+Y.min()) * 0.5
		# mid_z = (Z.max()+Z.min()) * 0.5
		# ax.set_xlim(mid_x - max_range, mid_x + max_range)
		# ax.set_ylim(mid_y - max_range, mid_y + max_range)
		# ax.set_zlim(mid_z - max_range, mid_z + max_range)
		# plt.show()

	except Exception as err:
		print(type(err))
		print (err.args)
		print (err)
		sys.exit(2)

        

if __name__ == "__main__":
   main(sys.argv[1:])
