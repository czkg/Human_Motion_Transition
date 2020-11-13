import numpy as np

SMPLParents = [0,0,0,0,1,2,3,4,5,6,7,8,
			   9,9,9,12,13,14,16,17,18,19,20,21]
H36MParents = [0,0,1,2,0,4,5,0,10,8,7,10,11,12,10,14,15]

def registrate(skeleton_s, skeleton_t):
	"""Map skeleton_s to skeleton_t
	params:
		skeleton_s: source skeleton of shape [17,3]
		skeleton_t: target sekelton of shape [24,3]
	return:
		skeleton: registrated source skeleton
	"""
	n = skeleton_s.shape[0]
	if n != 17 or n != 24:
		raise Exception('Invalid input for skeleton_s!')
	if n == 17:
		vec_s = skeleton_s[7] - skeleton_s[0]
		vec_s = vec_s / np.linalg.norm(vec_s)
	else:
		vec_s = skeleton_s[6] - skeleton_s[0]
		vec_s = vec_s / np.linalg.norm(vec_s)
	vec_t = skeleton_t[6] - skeleton_t[0]
	vec_t = vec_t / np.linalg.norm(vec_t)

	#rotate
	R = get_R(vec_s, vec_t)
	skeleton = [np.matmul(R,p) for p in skeleton_s]
	skeleton_s = np.vstack(skeleton)

	#translate
	skeleton_s = translate(skeleton_s, skeleton_t[0] - skeleton_s[0])

	if n == 24:
		boneLength = computeBoneLength(skeleton_t, 0)
		offUnit = computeRelativeOffUnit(skeleton_s, 0)

		for i in range(n):
			skeleton_s[i] = skeleton_s[SMPLParents[i]] + boneLength[i]*offUnit[i]

	return skeleton_s



def compute_J(A, b):
	"""compute joint regressor J
	params:
		A: verts of mesh
		b: target skeleton
	"""

def computeBoneLength(points, ty):
	boneLength = []
	if ty == 0:
		for i in range(24):
			bone = np.linalg.norm(points[i] - points[SMPLParents[i]])
			boneLength.append(bone)
	else:
		for i in range(17):
			bone = np.linalg.norm(points[i] - points[H36MParents[i]])
			boneLength.append(bone)
	return np.asarray(boneLength)

def computeRelativeOffUnit(points, ty):
	offs = []
	boneLength = computeBoneLength(points, ty)
	if ty == 0:
		for i in range(24):
			off = points[i] - points[SMPLParents[i]]
			offs.append(off)
	else:
		for i in range(17):
			off = points[i] - points[H36MParents[i]]
			offs.append(off)
		
	offs = np.asarray(offs)
	offUnit = [offs[i] / boneLength[i] for i in range(len(boneLength))]
	offUnit[0] = np.zeros((3))
	offUnit = np.asarray(offUnit)

	return offUnit

def translate(source, offset):
	return source+offset

def skew(vec):
	res = np.zeros((3, 3))
	res[0][1] = -vec[2]
	res[0][2] = vec[1]
	res[1][0] = vec[2]
	res[1][2] = -vec[0]
	res[2][0] = -vec[1]
	res[2][1] = vec[0]

	return res

def get_R(vec_s, vec_t):
	v = np.cross(vec_s, vec_t)
	s = np.linalg.norm(v)
	c = np.dot(vec_s, vec_t)
	I = np.eye(3)
	vx = skew(v)

	R = I + vx + np.dot(vx, vx) * (1. - c) / s**2
	return R