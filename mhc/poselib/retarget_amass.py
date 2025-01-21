import isaacgym
import os
import torch
import numpy as np
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

from poselib.core.rotation3d import *

from ase.utils.torch_utils import exp_map_to_quat, quat_to_exp_map
from ase.poselib.poselib.core.rotation3d import quat_from_rotation_matrix, quat_mul_norm, quat_rotate

file_dir = os.path.dirname(os.path.abspath(__file__))

amass_npz_fname = os.path.join(file_dir, '0008_Yoga001_poses.npz') 
smpl_data = np.load(amass_npz_fname)
n_steps = smpl_data['poses'].shape[0]
poses_smplh = torch.tensor( smpl_data['poses'][:, :66].reshape(n_steps,-1,3) )
trans = torch.tensor( smpl_data['trans'] )

smplh_skeleton = {'pelvis': 0, 'spine1': 3, 'spine2': 6, 'spine3': 9, 'neck': 12, 'head': 15,
                    'l_hip': 1, 'l_knee': 4, 'l_ankle': 7, 'l_foot': 10,
                    'r_hip': 2, 'r_knee': 5, 'r_ankle': 8, 'r_foot': 11,
                    'l_collar': 13,  'l_shoulder': 16,  'l_elbow': 18,  'l_wrist': 20,
                    'r_collar': 14,  'r_shoulder': 17,  'r_elbow': 19,  'r_wrist': 21}
amp_skeleton = {'pelvis':0, 'torso':1, 'head':2,
                'right_upper_arm':3, 'right_lower_arm':4, 'right_hand':5,
                'left_upper_arm':6, 'left_lower_arm':7, 'left_hand':8,
                'right_thigh':9, 'right_shin':10, 'right_foot':11,
                'left_thigh':12, 'left_shin':13, 'left_foot':14}
amp2smpl_body = {'pelvis':'pelvis', 'torso':'spine3', 'head':'neck',
                'right_upper_arm':'r_shoulder', 'right_lower_arm':'r_elbow', 'right_hand':'r_wrist',
                'left_upper_arm':'l_shoulder', 'left_lower_arm':'l_elbow', 'left_hand':'l_wrist',
                'right_thigh':'r_hip', 'right_shin':'r_knee', 'right_foot':'r_ankle',
                'left_thigh':'l_hip', 'left_shin':'l_knee', 'left_foot':'l_ankle'}
amp2smpl_mapping = {}
for body in amp2smpl_body:
    amp2smpl_mapping[body] = smplh_skeleton[ amp2smpl_body[body] ]

poses_exp_map = torch.zeros((n_steps,15,3))
for body in amp_skeleton:
    amp_body_idx = amp_skeleton[body]
    smplh_body_idx = amp2smpl_mapping[body]
    poses_exp_map[:,amp_body_idx,:] = poses_smplh[:,smplh_body_idx,:]

# ------ add rotation from missing joint ------
# 3,6
poses_quat_amass_spine1 = exp_map_to_quat( poses_smplh[:,smplh_skeleton['spine1'],:] )
poses_quat_amass_spine2 = exp_map_to_quat( poses_smplh[:,smplh_skeleton['spine2'],:] )
poses_quat_amass_spine3 = exp_map_to_quat( poses_smplh[:,smplh_skeleton['spine3'],:] )
poses_quat_amp_torso = quat_mul_norm(poses_quat_amass_spine3, quat_mul_norm(poses_quat_amass_spine2,poses_quat_amass_spine1) )
poses_exp_map[:,amp_skeleton['torso'],:] = quat_to_exp_map(poses_quat_amp_torso)
# 13,14
poses_quat_amass_lcollar = exp_map_to_quat( poses_smplh[:,smplh_skeleton['l_collar'],:] )
poses_quat_amass_lshoulder = exp_map_to_quat( poses_smplh[:,smplh_skeleton['l_shoulder'],:] )
poses_quat_amass_rcollar = exp_map_to_quat( poses_smplh[:,smplh_skeleton['r_collar'],:] )
poses_quat_amass_rshoulder = exp_map_to_quat( poses_smplh[:,smplh_skeleton['r_shoulder'],:] )
poses_quat_amp_lshoulder = quat_mul_norm(poses_quat_amass_lshoulder,poses_quat_amass_lcollar)
poses_quat_amp_rshoulder = quat_mul_norm(poses_quat_amass_rshoulder,poses_quat_amass_rcollar)
poses_exp_map[:,amp_skeleton['left_upper_arm'],:] = quat_to_exp_map(poses_quat_amp_lshoulder)
poses_exp_map[:,amp_skeleton['right_upper_arm'],:] = quat_to_exp_map(poses_quat_amp_rshoulder)

# # ------ T-pose ------
# poses_exp_map[:] = 0
# trans[:] = 0
# poses = exp_map_to_quat(poses_exp_map)
# t = SkeletonTree.from_mjcf('/home/liupan/om/assets/mjcf/amp_humanoid.xml')
# new_pose = SkeletonState.from_rotation_and_root_translation(
#     skeleton_tree=t,
#     r=poses[0],
#     t=trans[0],
#     is_local=True
#     )
# plot_skeleton_state(new_pose)

# poses2 = poses.clone()
rotation_matrix_to_tpose_l = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]])      # x 90
rotation_to_tpose_l = quat_from_rotation_matrix(rotation_matrix_to_tpose_l)
rotation_matrix_to_tpose_r = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]])      # x -90
rotation_to_tpose_r = quat_from_rotation_matrix(rotation_matrix_to_tpose_r)

# poses2[:,3] = quat_mul_norm(rotation_to_tpose_r, poses2[:,3])
# poses2[:,6] = quat_mul_norm(rotation_to_tpose_l, poses2[:,6])
# t = SkeletonTree.from_mjcf('/home/liupan/om/assets/mjcf/amp_humanoid.xml')
# new_pose = SkeletonState.from_rotation_and_root_translation(
#     skeleton_tree=t,
#     r=poses2[0],
#     t=trans[0],
#     is_local=True
#     )
# plot_skeleton_state(new_pose)
# # ------ T-pose ------

rot_x90 = quat_from_rotation_matrix( torch.tensor([[1,0,0],[0,0,-1],[0,1,0]]) )
rot_x90_minus = quat_from_rotation_matrix( torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]) )
rot_y90 = quat_from_rotation_matrix( torch.tensor([[0,0,1],[0,1,0],[-1,0,0]]) )
rot_y90_minus = quat_from_rotation_matrix( torch.tensor([[0,0,-1],[0,1,0],[1,0,0]]) )
rot_z90 = quat_from_rotation_matrix( torch.tensor([[0,-1,0],[1,0,0],[0,0,1]]) )
rot_z90_minus = quat_from_rotation_matrix( torch.tensor([[0,1,0],[-1,0,0],[0,0,1]]) )

rotation_matrix_to_upright = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]])      # x -90
rotation_to_upright = quat_from_rotation_matrix(rotation_matrix_to_upright)
rotation_matrix_to_z180 = torch.tensor([[-1,0,0],[0,-1,0],[0,0,1]])      # z 180
rotation_to_z180 = quat_from_rotation_matrix(rotation_matrix_to_z180)
rot1_matrix = torch.tensor([[0,0,-1],[-1,0,0],[0,1,0]])
rot1 = quat_from_rotation_matrix(rot1_matrix)
rot2_matrix = torch.tensor([[0,0,-1],[0,1,0],[1,0,0]])      # y -90
rot2 = quat_from_rotation_matrix(rot2_matrix)
rot6 = quat_from_rotation_matrix( torch.tensor([[0,1,0],[-1,0,0],[0,0,1]]) )
rot7 = quat_from_rotation_matrix( torch.tensor([[0,-1,0],[1,0,0],[0,0,1]]) )
rot3 = quat_mul_norm(rotation_to_tpose_l,rot6)
rot4 = quat_mul_norm(rotation_to_tpose_r,rot6)
rotation_matrix_z90 = torch.tensor([[0,-1,0],[1,0,0],[0,0,1]])      # z 90
rotation_z90 = quat_from_rotation_matrix(rotation_matrix_to_z180)
rot5 = quat_mul_norm(rotation_z90,rotation_to_tpose_r)

rot_x90_then_y90 = quat_mul_norm(rot_y90,rot_x90)
rot_x90_then_y90_minus = quat_mul_norm(rot_y90_minus,rot_x90)
rot_x90_minus_then_y90 = quat_mul_norm(rot_y90,rot_x90_minus)
rot_x90_minus_then_y90_minus = quat_mul_norm(rot_y90_minus,rot_x90_minus)
rot_x90_then_z90 = quat_mul_norm(rot_z90,rot_x90)
rot_x90_then_z90_minus = quat_mul_norm(rot_z90_minus,rot_x90)
rot_x90_minus_then_z90 = quat_mul_norm(rot_z90,rot_x90_minus)
rot_x90_minus_then_z90_minus = quat_mul_norm(rot_z90_minus,rot_x90_minus)

rot_y90_then_x90 = quat_mul_norm(rot_x90,rot_y90)
rot_y90_then_x90_minus = quat_mul_norm(rot_x90_minus,rot_y90)
rot_y90_minus_then_x90 = quat_mul_norm(rot_x90,rot_y90_minus)
rot_y90_minus_then_x90_minus = quat_mul_norm(rot_x90_minus,rot_y90_minus)
rot_y90_then_z90 = quat_mul_norm(rot_z90,rot_y90)
rot_y90_then_z90_minus = quat_mul_norm(rot_z90_minus,rot_y90)
rot_y90_minus_then_z90 = quat_mul_norm(rot_z90,rot_y90_minus)
rot_y90_minus_then_z90_minus = quat_mul_norm(rot_z90_minus,rot_y90_minus)

rot_z90_then_y90 = quat_mul_norm(rot_y90,rot_z90)
rot_z90_then_y90_minus = quat_mul_norm(rot_y90_minus,rot_z90)
rot_z90_minus_then_y90 = quat_mul_norm(rot_y90,rot_z90_minus)
rot_z90_minus_then_y90_minus = quat_mul_norm(rot_y90_minus,rot_z90_minus)
rot_z90_then_x90 = quat_mul_norm(rot_x90,rot_z90)
rot_z90_then_x90_minus = quat_mul_norm(rot_x90_minus,rot_z90)
rot_z90_minus_then_x90 = quat_mul_norm(rot_x90,rot_z90_minus)
rot_z90_minus_then_x90_minus = quat_mul_norm(rot_x90_minus,rot_z90_minus)

# trans[:] = 0
poses_exp_map2 = torch.clone(poses_exp_map)
# 1
# poses_exp_map[:,:,0] = poses_exp_map2[:,:,1]
# poses_exp_map[:,:,1] = poses_exp_map2[:,:,0]
# poses_exp_map[:,:,2] = -poses_exp_map2[:,:,2]
# 2
# poses_exp_map[:,:,0] = poses_exp_map2[:,:,2]
# poses_exp_map[:,:,1] = poses_exp_map2[:,:,1]
# poses_exp_map[:,:,2] = poses_exp_map2[:,:,0]
# 3 [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]
poses_exp_map[:,:,0] = poses_exp_map2[:,:,2]
poses_exp_map[:,:,1] = poses_exp_map2[:,:,0]
poses_exp_map[:,:,2] = poses_exp_map2[:,:,1]

poses = exp_map_to_quat(poses_exp_map)
poses[:,0,:] = quat_mul_norm(rot2, poses[:,0,:])
# poses[:,0,:] = quat_mul_norm(rotation_to_upright, poses[:,0,:])
# poses[:,0,:] = quat_mul_norm(rotation_to_z180, poses[:,0,:])
# poses[:,3,:] = quat_mul_norm(rotation_to_tpose_r, poses[:,3,:])
# poses[:,6,:] = quat_mul_norm(rotation_to_tpose_l, poses[:,6,:])
# poses[:,3,:] = quat_mul_norm(rot4, poses[:,3,:])
# poses[:,6,:] = quat_mul_norm(rot3, poses[:,6,:])
# poses[:,3,:] = quat_mul_norm(rot5, poses[:,3,:])
# rot_z90_minus_then_x90_minus, rot_x90_minus_then_y90_minus, rot_y90_minus_then_z90_minus
poses[:,3,:] = quat_mul_norm(rot_y90_minus_then_z90_minus, poses[:,3,:])
poses[:,6,:] = quat_mul_norm(rot_y90_minus_then_z90, poses[:,6,:])

t = SkeletonTree.from_mjcf('/home/liupan/om/assets/mjcf/amp_humanoid.xml')
new_pose = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=t,
    r=poses[600],
    t=trans[0],
    is_local=True
    )
plot_skeleton_state(new_pose)

state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=t,
            r=poses,
            t=trans,
            is_local=True,
        )
motion = SkeletonMotion.from_skeleton_state(state, 60)
# motion.to_file("sfu_0008_Yoga001_poses.npy")

plot_skeleton_motion_interactive(motion)




