from config import *
import pickle
import numpy as np

# The SMPL_MASK is used to simplify SMPL joint so we can have LSP-like joints
SMPL_14_MASK = [1,2,4,5,7,8,12,16,17,18,19,20,21]
SMPL_17_MASK = []

def get

def prepare_smpl_14_model():
  """
  Convert the official SMPL model into compatible format with this project.
  """
  with open(OFFICIAL_SMPL_PATH, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

  # J_regressor
  J_regressor = data['J_regressor'].toarray()
  J_regressor = np.array([J_regressor[i] for i in SMPL_14_MASK])
  # weights
  weights = data['weights']
  weights = np.transpose(np.array([weights[:,i] for i in SMPL_14_MASK]))
  # posedirs
  posedirs = data['posedirs']
  idx = [slice(i*9,(i+1)*9) for i in SMPL_14_MASK]
  posedirs = [posedirs[:,:,i] for i in idx]
  posedirs = np.concatenate(posedirs, axis=-1)

  params = {
    # SMPL does not provide pose PCA
    'pose_pca_basis': np.eye(12 * 3),
    'pose_pca_mean': np.zeros(12 * 3),
    'J_regressor': J_regressor,
    'skinning_weights': weights,
    # pose blend shape
    'mesh_pose_basis': posedirs,
    'mesh_shape_basis': np.array(data['shapedirs']),
    'mesh_template': np.array(data['v_template']),
    'faces': np.array(data['f']),
    'parents': data['kintree_table'][0].tolist(),
  }
  params['parents'][0] = None
  with open(SMPL_MODEL_PATH, 'wb') as f:
    pickle.dump(params, f)


def prepare_smpl_17_model():
  """
  Convert the official SMPL model into compatible format with this project.
  """
  with open(OFFICIAL_SMPL_PATH, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

  # J_regressor
  J_regressor = data['J_regressor'].toarray()
  J_regressor = np.array([J_regressor[i] for i in SMPL_17_MASK])
  # weights
  weights = data['weights']
  weights = np.transpose(np.array([weights[:,i] for i in SMPL_17_MASK]))
  # posedirs
  posedirs = data['posedirs']
  idx = [slice(i*9,(i+1)*9) for i in SMPL_17_MASK]
  posedirs = [posedirs[:,:,i] for i in idx]
  posedirs = np.concatenate(posedirs, axis=-1)

  params = {
    # SMPL does not provide pose PCA
    'pose_pca_basis': np.eye(15 * 3),
    'pose_pca_mean': np.zeros(15 * 3),
    'J_regressor': J_regressor,
    'skinning_weights': weights,
    # pose blend shape
    'mesh_pose_basis': posedirs,
    'mesh_shape_basis': np.array(data['shapedirs']),
    'mesh_template': np.array(data['v_template']),
    'faces': np.array(data['f']),
    'parents': data['kintree_table'][0].tolist(),
  }
  params['parents'][0] = None
  with open(SMPL_MODEL_PATH, 'wb') as f:
    pickle.dump(params, f)

if __name__ == '__main__':
  prepare_smpl_model()
