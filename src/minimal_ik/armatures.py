# it must be assured that parent joint appears before child joint


class SMPLArmature:
  n_joints = 24

  #keypoints_ext = [2446, 5907, 3216, 6618, 411]
  #keypoints_ext = [411]
  keypoints_ext = []

  n_keypoints = n_joints + len(keypoints_ext)

  labels = [
    'pelvis',
    'left leg root', 'right leg root',
    'lowerback',
    'left knee', 'right knee',
    'upperback',
    'left ankle', 'right ankle',
    'thorax',
    'left toes', 'right toes',
    'lowerneck',
    'left clavicle', 'right clavicle',
    'upperneck',
    'left armroot', 'right armroot',
    'left elbow', 'right elbow',
    'left wrist', 'right wrist',
    'left hand', 'right hand'
    # extended
    'left finger tip', 'right finger tip', 'left toe tip', 'right toe tip',
    'head_top'
  ]

  class H36MArmature:
    n_joints = 17

    #keypoints_ext = [2446, 5907, 3216, 6618, 411]
    keypoints_ext = [411]
    n_keypoints = n_joints + len(keypoints_ext)


    labels = [
    'pelvis',
    'right hip', 'right knee', 'right ankle',
    'left hip', 'left knee', 'left ankle',
    'spine', 'neck', 'head', 'thorax'
    'left shoulder', 'left elbow', 'left wrist',
    'right shoulder', 'right elbow', 'right wrist'
    # extended
    'left finger tip', 'right finger tip', 'left toe tip', 'right toe tip',
    'head_top'
  ]