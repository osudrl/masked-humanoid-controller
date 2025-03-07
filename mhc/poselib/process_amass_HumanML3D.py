import isaacgym
import os
import torch
import numpy as np
import pandas as pd

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
# from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

from poselib.torch_utils import exp_map_to_quat
from poselib.core.rotation3d import quat_inverse, quat_mul_norm, quat_rotate, quat_yaw_rotation

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


def convert_amass(motion_source_path, motion_output_path, truncate_frame_start, motion_start_frame, motion_end_frame):
    # amass_motion_name = file_name.split('.')[0]
    # amass_npz_fname = os.path.join(file_dir, file_name)
    # converted_amass_file = os.path.join(output_dir, amass_motion_name+".npy")
    # if os.path.isfile(converted_amass_file):    # check if file already exist
    #     return converted_amass_file
    smpl_data = np.load(motion_source_path)
    fps = smpl_data['mocap_framerate']
    n_steps = smpl_data['poses'].shape[0]
    poses_smplh = torch.tensor( smpl_data['poses'][:, :72].reshape(n_steps,-1,3) )
    trans = torch.tensor( smpl_data['trans'] )

    # if n_steps < 10:
    #     return ''
    assert n_steps>10, "n_steps < 10, n_steps: {}".format(n_steps)

    fps_HumanML3D = 20
    frame_truncate_up_scale = int(fps/fps_HumanML3D)
    truncate_frame_start = truncate_frame_start * frame_truncate_up_scale
    motion_start_frame = motion_start_frame * frame_truncate_up_scale
    motion_end_frame = motion_end_frame * frame_truncate_up_scale

    poses_smplh = poses_smplh[truncate_frame_start:]
    trans = trans[truncate_frame_start:]
    poses_smplh = poses_smplh[motion_start_frame:motion_end_frame]
    trans = trans[motion_start_frame:motion_end_frame]

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

    target_motion.to_file(motion_output_path)
    return

amass_dataset = [
    # "ACCAD",
    # "BioMotionLab_NTroje",
    # "BMLhandball",
    # "BMLmovi",
    # "CMU",
    # "DFaust_67",
    # "EKUT",
    # "Eyes_Japan_Dataset",
    # "HumanEva",
    # "KIT",
    # "MPI_HDM05",
    # "MPI_Limits",
    # "MPI_mosh",
    "SFU",
    # "SSM_synced",
    # "TCD_handMocap",      # not used in HumanML3D
    # "TotalCapture",
    # "Transitions_mocap",
    # # "humanact12",
]

dataset_truncate = {'Eyes_Japan_Dataset': 3*20, 'MPI_HDM05': 3*20, 'TotalCapture': 1*20, 'MPI_Limits': 1*20, 'Transitions_mocap': int(0.5*20)}
#  if 'humanact12' not in source_path:
#         if 'Eyes_Japan_Dataset' in source_path:
#             data = data[3*fps:]
#         if 'MPI_HDM05' in source_path:
#             data = data[3*fps:]
#         if 'TotalCapture' in source_path:
#             data = data[1*fps:]
#         if 'MPI_Limits' in source_path:
#             data = data[1*fps:]
#         if 'Transitions_mocap' in source_path:
#             data = data[int(0.5*fps):]

if __name__ == '__main__':
    '''
    To run:
    1. Download datasets (tar.bz2 files) from AMASS website to ~/Downloads/amass
    2. `cd ase/data/amass && python unzip.py`
    3. update the 'amass_data_dir' and 'amass_output_dir' below
    4. run this script `python ase/poselib/process_amass_HumanML3D.py`
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    os.chdir(project_root)

    home_folder = os.path.expanduser("~")
    amass_data_dir = home_folder + "/Downloads/"
    amass_output_dir = 'mhc/data/motions/amass/'
    amass_occlusion = joblib.load("mhc/data/motions/amass/amass_copycat_occlusion_v3.pkl")
    text_data_dir = "mhc/data/motions/amass/texts"

    index_path = 'mhc/data/motions/amass/index.csv'
    index_file = pd.read_csv(index_path)

    all_motions = []
    for dataset in tqdm(amass_dataset):
        curr_data_dir = os.path.join(amass_data_dir,dataset)
        output_dir = curr_data_dir.replace(amass_data_dir, amass_output_dir)
        os.makedirs(output_dir, exist_ok=True)

        motion_list = index_file[index_file['source_path'].str.contains(dataset, case=True, na=False)]

        truncate_frame_start = dataset_truncate[dataset] if dataset in dataset_truncate else 0

        curr_dataset_motions = []
        for index, row in motion_list.iterrows():
            motion_source_path = home_folder + row['source_path'].replace('./pose_data', amass_data_dir).replace('.npy', '.npz')
            motion_start_frame = row['start_frame']
            motion_end_frame = row['end_frame']
            motion_output_path = os.path.join(output_dir, row['new_name'])
            motion_text_path = os.path.join(text_data_dir, row['new_name'].replace('.npy', '.txt'))

            #with open(motion_text_path, 'r', encoding='utf-8') as text_file:
            #    motion_texts = text_file.readlines()
            #motion_texts = [x.split('#')[0] for x in motion_texts]

            if ('0-'+dataset+"_"+motion_source_path.split('/')[-2]+'_'+motion_source_path.split('/')[-1][:-4] in amass_occlusion):
                continue
            
            if not os.path.isfile(motion_output_path) and (motion_end_frame-motion_start_frame>5):    # check if file already exist
                convert_amass(motion_source_path, motion_output_path, truncate_frame_start, motion_start_frame, motion_end_frame)

            all_motions.append({"file": motion_output_path.replace(amass_output_dir, ""), "weight": 1.0})
            curr_dataset_motions.append({"file": motion_output_path.replace(os.path.join(amass_output_dir,dataset)+"/", ""), "weight": 1.0})

        with open(os.path.join(amass_output_dir, dataset, "dataset_"+dataset+".yaml"), 'w') as f:
            print(dataset, " number of motions: ", len(curr_dataset_motions))
            yaml.dump({"motions": curr_dataset_motions}, f)
    
    print("\n-------------------")
    print("number of motions: ", len(all_motions))
    print("writing dataset_amass.yaml to ", os.path.join(amass_output_dir, "amp_humanoid_amass_dataset.yaml"))
    with open(os.path.join(amass_output_dir, "amp_humanoid_amass_dataset.yaml"), 'w') as f:
        yaml.dump({"motions": all_motions}, f)