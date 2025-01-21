import isaacgym
import os
import torch
import numpy as np

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
# from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

from ase.utils.torch_utils import exp_map_to_quat
from ase.poselib.poselib.core.rotation3d import quat_inverse, quat_mul_norm, quat_rotate, quat_yaw_rotation

import yaml
import joblib
from tqdm import tqdm

JOINT_NAMES_SMPL = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index',
    'right_index',
]

JOINT_NAMES_HUMANOID = [
    'pelvis',
    'left_hip',
    'left_knee',
    'left_ankle',
    'left_foot',
    'right_hip',
    'right_knee',
    'right_ankle',
    'right_foot',
    'spine1',
    'spine2',
    'spine3',
    'neck',
    'head',
    'left_collar',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'left_index',
    'right_collar',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'right_index',
]

SMPL2IDX = {v: k for k, v in enumerate(JOINT_NAMES_SMPL)}
SMPL2HUMANOID = [SMPL2IDX[j] for j in JOINT_NAMES_HUMANOID]


def convert_amass(file_dir, file_name, output_dir):
    amass_motion_name = file_name.split('.')[0]
    amass_npz_fname = os.path.join(file_dir, file_name)
    converted_amass_file = os.path.join(output_dir, amass_motion_name+".npy")
    if os.path.isfile(converted_amass_file):    # check if file already exist
        return converted_amass_file
    
    smpl_data = np.load(amass_npz_fname)
    n_steps = smpl_data['poses'].shape[0]
    if n_steps < 10:
        return ''
    poses_smplh = torch.tensor( smpl_data['poses'][:, :72].reshape(n_steps,-1,3) )
    trans = torch.tensor( smpl_data['trans'] )

    poses_exp_map = poses_smplh
    poses_exp_map = poses_exp_map[:, SMPL2HUMANOID]
    poses_exp_map = poses_exp_map[:, :, [2, 0, 1]]
    trans = trans[:, [2, 0, 1]]

    poses = exp_map_to_quat(poses_exp_map)

    # remove the additional y-axis (original x-axis) 90 degree rotation added by AMASS
    # init_pose = poses[0, 0].clone()       # the root_orientation at time 0, poses[0,0], may not be upright
    init_pose = torch.tensor([0,0.5,0,0.5]) # 90 degree rotation around y-axis
    poses[:, 0] = quat_mul_norm(quat_inverse(init_pose), poses[:, 0])
    trans = quat_rotate(quat_inverse(init_pose), trans)

    # rotate by a yaw angle to make the character face the x-axis
    init_pose = quat_yaw_rotation(poses[0, 0], z_up=True)
    poses[:, 0] = quat_mul_norm(quat_inverse(init_pose), poses[:, 0])
    trans = quat_rotate(quat_inverse(init_pose), trans)

    t = SkeletonTree.from_mjcf('mhc/data/assets/mjcf/smpl_humanoid.xml')
    # new_pose = SkeletonState.from_rotation_and_root_translation(
    #     skeleton_tree=t,
    #     r=poses[200],
    #     t=trans[0],
    #     is_local=True
    #     )
    # plot_skeleton_state(new_pose)

    state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree=t,
                r=poses,
                t=trans,
                is_local=True,
            )
    target_motion = SkeletonMotion.from_skeleton_state(state, smpl_data['mocap_framerate'].item())
    # plot_skeleton_motion_interactive(motion)

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h
    
    # adjust the height of the root to avoid ground penetration
    root_height_offset = 0.0
    root_translation[:, 2] += root_height_offset
    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                        target_motion.skeleton_tree, 
                        local_rotation, 
                        root_translation, 
                        is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    target_motion.to_file(converted_amass_file)
    return converted_amass_file

amass_dataset = [
    "ACCAD",
    "BMLhandball",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "DanceDB",
    "EKUT",
    "Eyes_Japan_Dataset",
    "MPI_HDM05",
    "HumanEva",
    "KIT",
    "MPI_mosh",
    "MPI_Limits",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
]

if __name__ == '__main__':
    '''
    To run:
    1. Download datasets (tar.bz2 files) from AMASS website to ./ase/data/amass
    2. `cd ase/data/amass && python unzip.py`
    3. update the 'amass_data_dir' and 'amass_output_dir' below
    4. run this script `python ase/poselib/process_amass.py`
    '''
    amass_data_dir = "/home/pan/Downloads/amass"
    amass_output_dir = "/home/pan/Downloads/amass_processed"
    amass_occlusion = joblib.load("ase/data/amass/amass_copycat_occlusion_v2.pkl")
    all_motions = []
    for dataset in tqdm(amass_dataset):
        curr_data_dir = os.path.join(amass_data_dir,dataset)
        curr_dataset_motions = []
        for root, dirs, files in os.walk(curr_data_dir):
            for file in sorted(files):
                if file.endswith("shape.npz"):      # shape.npz does not contain 'poses'
                    continue
                elif file.endswith(".npz"):
                    # print(os.path.join(root, file))
                    output_dir = root.replace(amass_data_dir, amass_output_dir)
                    os.makedirs(output_dir, exist_ok=True)

                    output_file_name = convert_amass(root, file, output_dir)
                    if output_file_name and ('0-'+dataset+"_"+root.replace(curr_data_dir,'')[1:]+'_'+file[:-4] not in amass_occlusion):
                        all_motions.append({"file": output_file_name.replace(amass_output_dir+"/", ""), "weight": 1.0})
                        curr_dataset_motions.append({"file": output_file_name.replace(os.path.join(amass_output_dir,dataset)+"/", ""), "weight": 1.0})
        
        with open(os.path.join(amass_output_dir, dataset, "dataset_"+dataset+".yaml"), 'w') as f:
            print(dataset, " number of motions: ", len(curr_dataset_motions))
            yaml.dump({"motions": curr_dataset_motions}, f)
    
    print("\n-------------------")
    print("number of motions: ", len(all_motions))
    print("writing dataset_amass.yaml to ", os.path.join(amass_output_dir, "dataset_amass.yaml"))
    with open(os.path.join(amass_output_dir, "dataset_amass.yaml"), 'w') as f:
        yaml.dump({"motions": all_motions}, f)


