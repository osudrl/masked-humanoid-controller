import torch
import numpy as np

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
    'torso',
    'head',
    'right_upper_arm',
    'right_lower_arm',
    'right_hand',
    'sword',
    'left_upper_arm',
    'left_lower_arm',
    'shield',
    'left_hand',
    'right_thigh',
    'right_shin',
    'right_foot',
    'left_thigh',
    'left_shin',
    'left_foot',
]


joint_mapping = {
        "pelvis": "pelvis",
        "torso": "spine2",
        "head": "neck",
        "right_upper_arm": "right_shoulder",
        "right_lower_arm": "right_elbow",
        "right_hand": "right_wrist",
        # sword
        "left_upper_arm": "left_shoulder",
        "left_lower_arm": "left_elbow",
        # shield
        "left_hand": "left_wrist",
        "right_thigh": "right_hip", 
        "right_shin": "right_knee", 
        "right_foot": "right_ankle", 
        "left_thigh": "left_hip",
        "left_shin": "left_knee",
        "left_foot": "left_ankle",
}


def interpolate_frames(input_tensor, target_fps, source_fps=20):
    """
    Interpolates a tensor from source_fps to target_fps.

    Args:
    input_tensor (torch.Tensor): The input tensor of shape (n, m, 3) where n is the number of frames,
                                m is the number of joints.
    target_fps (int): The target frames per second.
    source_fps (int, optional): The source frames per second. Defaults to 20.

    Returns:
    torch.Tensor: Interpolated tensor.
    """

    n, m, _ = input_tensor.shape
    scale_factor = target_fps / source_fps
    new_n = int(np.ceil(n * scale_factor))

    # Create an output tensor
    output_tensor = torch.zeros((new_n, m, 3))

    # Time points for original and interpolated data
    original_times = np.linspace(0, n, n)
    interpolated_times = np.linspace(0, n, new_n)

    for joint in range(m):
        for dim in range(3):
            # Linearly interpolate each joint's trajectory
            output_tensor[:, joint, dim] = torch.from_numpy(
                np.interp(
                    interpolated_times, 
                    original_times, 
                    input_tensor[:, joint, dim].numpy()
                )
            )

    return output_tensor



# ---------------------------------------------------------------------------------------
# text to motion data
# ---------------------------------------------------------------------------------------
def extract_keypoints_from_text(file_name, source_fps=20, target_fps=30):
    SMPL2IDX = {v: k for k, v in enumerate(JOINT_NAMES_SMPL)}
    HUMANOID2IDX = {v: k for k, v in enumerate(JOINT_NAMES_HUMANOID)}

    SMPL2HUMANOID = []
    for j in JOINT_NAMES_HUMANOID:
        if j in joint_mapping:
            SMPL2HUMANOID.append(SMPL2IDX[joint_mapping[j]])
        else:
            SMPL2HUMANOID.append(0)

    sword_idx = JOINT_NAMES_HUMANOID.index('sword')
    shield_idx = JOINT_NAMES_HUMANOID.index('shield')
    sword_shiled_mask = torch.zeros(len(JOINT_NAMES_HUMANOID))
    sword_shiled_mask[sword_idx] = 1
    sword_shiled_mask[shield_idx] = 1

    video_keypos = torch.from_numpy(np.load(file_name))[:, SMPL2HUMANOID]
    video_keypos[:,:,2] -= video_keypos[:,:,2].min()
    video_keypos[:,sword_idx] = 0
    video_keypos[:,shield_idx] = 0


    video_keypos = interpolate_frames(torch.FloatTensor(video_keypos), source_fps = source_fps, target_fps = target_fps).numpy()

    catchup_window = 60  # Replace with your desired number
    video_keypos_padded = np.pad(video_keypos, ((catchup_window, catchup_window), (0,0), (0, 0)), mode='constant', constant_values=0)
    video_keypos_padded[:catchup_window] = video_keypos[0:1] 
    video_keypos_padded[-catchup_window:] = video_keypos[-1:] 

    return video_keypos_padded, sword_shiled_mask

# ---------------------------------------------------------------------------------------
# video data
# ---------------------------------------------------------------------------------------
def extract_keypoints_from_video(file_name, source_fps=45, target_fps=30):
    SMPL2IDX = {v: k for k, v in enumerate(JOINT_NAMES_SMPL)}
    HUMANOID2IDX = {v: k for k, v in enumerate(JOINT_NAMES_HUMANOID)}

    SMPL2HUMANOID = []
    for j in JOINT_NAMES_HUMANOID:
        if j in joint_mapping:
            SMPL2HUMANOID.append(SMPL2IDX[joint_mapping[j]])
        else:
            SMPL2HUMANOID.append(0)

    sword_idx = JOINT_NAMES_HUMANOID.index('sword')
    shield_idx = JOINT_NAMES_HUMANOID.index('shield')
    sword_shiled_mask = torch.zeros(len(JOINT_NAMES_HUMANOID))
    sword_shiled_mask[sword_idx] = 1
    sword_shiled_mask[shield_idx] = 1

    video_keypos = torch.from_numpy(np.load(file_name))[:, SMPL2HUMANOID]/1000
    video_keypos[..., 1], video_keypos[..., 2] = video_keypos[..., 2], -video_keypos[..., 1]
    video_keypos[:,:,2] -= video_keypos[:,:,2].min()

    video_keypos[:,sword_idx] = 0
    video_keypos[:,shield_idx] = 0

    video_keypos = interpolate_frames(video_keypos, source_fps = source_fps, target_fps = target_fps).numpy()

    catchup_window = 60  # Replace with your desired number
    video_keypos_padded = np.pad(video_keypos, ((catchup_window, catchup_window), (0,0), (0, 0)), mode='constant', constant_values=0)
    video_keypos_padded[:catchup_window] = video_keypos[0:1] 
    video_keypos_padded[-catchup_window:] = video_keypos[-1:] 

    return video_keypos_padded, sword_shiled_mask

# ---------------------------------------------------------------------------------------
# vr data
# ---------------------------------------------------------------------------------------
import re

# ---------------------------------------------------------------------------------------
# vr data
# ---------------------------------------------------------------------------------------
import re
import numpy as np
def extract_data_from_vr(file_name, source_fps = 30, target_fps = 30, root_shift = 0.625, global_shift_down = 0.15, max_foot_distance = 0.3):

    def extract_data(text, type):
        if type == "quat":
            # Regular expression to match quaternion data
            pattern = r"X:\s*(-?\d+\.\d+),\s*Y:\s*(-?\d+\.\d+),\s*Z:\s*(-?\d+\.\d+),\s*W:\s*(-?\d+\.\d+)"
        elif type == "pos":
            # Regular expression to match position data
            pattern = r"X=(-?\d+\.?\d*)\sY=(-?\d+\.?\d*)\sZ=(-?\d+\.?\d*)"
        
        # Search for the pattern in the text
        match = re.search(pattern, text)

        # assert match is found
        assert match is not None, "No match found"
        
        return torch.tensor( tuple(float(num) for num in match.groups()) )
    
    
    with open(file_name, 'r') as file:
        data = file.read()
    data = data.split('\n')
    if data[0].startswith("[2024"):
        for i in range(len(data)):
            data[i] = data[i][30:]
    # truncate data from first "Logging start" to last "Logging complete", and remove "Logging start" and "Logging complete" lines
    idx_logging_start = data.index("LogBlueprintUserMessages: [VRPawn_C_0] Logging start")
    idx_logging_complete = len(data) - 1 - data[::-1].index("LogBlueprintUserMessages: [VRPawn_C_0] Logging complte")       # the last index of "Logging complete"
    data = data[idx_logging_start+1:idx_logging_complete]
    data = [d for d in data if d != "LogBlueprintUserMessages: [VRPawn_C_0] Logging start" and d != "LogBlueprintUserMessages: [VRPawn_C_0] Logging complte"]

    assert len(data) % 6 == 0, "Length of data is not divisible by 6"

    n_steps = len(data) // 6

    rot_data = torch.zeros(len(data) // 6, 3, 4)
    pos_data = torch.zeros(len(data) // 6, 3, 3)
    
    for i in range(n_steps):
        idx = i * 6
        head_rot_text = data[idx]
        hand_left_rot_text = data[idx+1]
        hand_right_rot_text = data[idx+2]
        head_pos_text = data[idx+3]
        hand_left_pos_text = data[idx+4]
        hand_right_pos_text = data[idx+5]

        assert "Head_rot" in head_rot_text, "Head_rot not found in head_rot_text"
        assert "Hand_left_rot" in hand_left_rot_text, "Hand_left_rot not found in hand_left_rot_text"
        assert "Hand_right_rot" in hand_right_rot_text, "Hand_right_rot not found in hand_right_rot_text"
        assert "Head_pos" in head_pos_text, "Head_pos not found in head_pos_text"
        assert "Hand_left_pos" in hand_left_pos_text, "Hand_left_pos not found in hand_left_pos_text"
        assert "Hand_right_pos" in hand_right_pos_text, "Hand_right_pos not found in hand_right_pos_text"

        head_rot = extract_data(head_rot_text, "quat")
        hand_left_rot = extract_data(hand_left_rot_text, "quat")
        hand_right_rot = extract_data(hand_right_rot_text, "quat")
        head_pos = extract_data(head_pos_text, "pos")
        hand_left_pos = extract_data(hand_left_pos_text, "pos")
        hand_right_pos = extract_data(hand_right_pos_text, "pos")

        rot_data[i, 0] = head_rot
        rot_data[i, 1] = hand_left_rot
        rot_data[i, 2] = hand_right_rot
        pos_data[i, 0] = head_pos
        pos_data[i, 1] = hand_left_pos
        pos_data[i, 2] = hand_right_pos

    pos_data /= 100     # convert from cm to m
    
    # convert from left-handed to right-handed coordinate system
    rot_data[..., 0], rot_data[..., 2] = -rot_data[..., 0], -rot_data[..., 2]
    pos_data[..., 1] = -pos_data[..., 1]


    root_idx = JOINT_NAMES_HUMANOID.index('pelvis')
    head_idx = JOINT_NAMES_HUMANOID.index('head')
    left_hand_idx = JOINT_NAMES_HUMANOID.index('left_hand')
    right_hand_idx = JOINT_NAMES_HUMANOID.index('right_hand')
    left_foot_idx = JOINT_NAMES_HUMANOID.index('left_foot')
    right_foot_idx = JOINT_NAMES_HUMANOID.index('right_foot')

    body_mask = torch.zeros(len(JOINT_NAMES_HUMANOID))
    body_mask[:] = 1
    body_mask[root_idx], body_mask[head_idx], body_mask[left_hand_idx], body_mask[right_hand_idx] = 0, 0, 0, 0
    body_mask[left_foot_idx], body_mask[right_foot_idx] = 0, 0

    keypos = torch.zeros(n_steps, len(JOINT_NAMES_HUMANOID), 3)
    
    # Manually decrease the height of head
    pos_data[:, 0, 2] = pos_data[:, 0, 2] - 0.1

    # Fill in the keypos tensor
    keypos[:, head_idx] = pos_data[:, 0]
    keypos[:, left_hand_idx] = pos_data[:, 1]
    keypos[:, right_hand_idx] = pos_data[:, 2]

    # make root position 1m below the head
    keypos[:, root_idx] = pos_data[:, 0] 
    keypos[:, root_idx, 2] = pos_data[:, 0, 2] - root_shift

    # Assuming pos_data contains the positions of head, left hand, and right hand respectively
    # and root_shift is the distance by which the root is shifted below the head
    # Define the maximum distance cap
    

    # Define base vectors for the foot positions, 0.3 meters apart
    left_foot_base = np.array([0.2, 0, 0])  # 0.15 meters to the left of the root
    right_foot_base = np.array([-0.2, 0, 0])  # 0.15 meters to the right of the root

    # Calculate the vector from left hand to right hand
    hand_line_vector = keypos[:, right_hand_idx,:2] - keypos[:, left_hand_idx,:2]
    # Normalize the vector to get the direction
    hand_line_direction = hand_line_vector / np.linalg.norm(hand_line_vector, axis=1, keepdims=True)
    hand_line_direction[:, 0] = 0  
    hand_line_direction[:, 1] = 1  

    keypos[:, left_foot_idx, :2] = hand_line_direction * max_foot_distance + keypos[:, root_idx, :2]
    keypos[:, right_foot_idx, :2] = -hand_line_direction * max_foot_distance + keypos[:, root_idx, :2]
    keypos[:, left_foot_idx, 2] = 0
    keypos[:, right_foot_idx, 2] = 0


    # Shift everythign down globally to match height. 
    keypos[:, head_idx, 2] -= global_shift_down
    keypos[:, left_hand_idx, 2] -= global_shift_down
    keypos[:, right_hand_idx, 2] -= global_shift_down
    keypos[:, root_idx, 2] -= global_shift_down
    keypos = interpolate_frames(keypos, source_fps = source_fps, target_fps = target_fps).numpy()
    
    extra = {'head_quat': rot_data[:, 0], 'left_hand_quat': rot_data[:, 1], 'right_hand_quat': rot_data[:, 2]}

    return keypos, body_mask, extra



if __name__ == "__main__":
    file_name = 'ase/experimental/pan/t2m_gpt/a person is boxing with a punching bag.npy'
    video_keypos, sword_shiled_mask = extract_keypoints_from_text(file_name)
    print(video_keypos.shape)
    print(sword_shiled_mask)
    print("")

    file_name = 'ase/experimental/pan/visualize_video_motion/motions/IMG 1199 - squat.npy'
    video_keypos, sword_shiled_mask = extract_keypoints_from_video(file_name)
    print(video_keypos.shape)
    print(sword_shiled_mask)
    print("")

    file_name = "ase/experimental/pan/vr/data1.txt"
    keypos, body_mask, extra = extract_data_from_vr(file_name)
    print(keypos.shape)
    print(body_mask)
    print("")

    




'''
# ---------------------------------------------------------------------------------------
# video to motion data data
# ---------------------------------------------------------------------------------------
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
head_idx = JOINT_NAMES_HUMANOID.index('head')
left_foot_idx = JOINT_NAMES_HUMANOID.index('left_foot')
right_foot_idx = JOINT_NAMES_HUMANOID.index('right_foot')

# video_keypos = torch.from_numpy(np.load('ase/data/video/IMG 1197 - walking.npy'))[:, SMPL2HUMANOID]/1000
# video_keypos = torch.from_numpy(np.load('ase/data/video/IMG 1198 - jumping.npy'))[:, SMPL2HUMANOID]/1000
video_keypos = torch.from_numpy(np.load('ase/data/video/IMG 1199 - squat.npy'))[:, SMPL2HUMANOID]/1000
# video_keypos = torch.from_numpy(np.load('ase/data/video/IMG 1200 - jumping jacks.npy'))[:, SMPL2HUMANOID]/1000
# video_keypos = torch.from_numpy(np.load('ase/data/video/IMG 1201 - frontkick.npy'))[:, SMPL2HUMANOID]/1000
# video_keypos = torch.from_numpy(np.load('ase/data/video/IMG 1203 - skipping.npy'))[:, SMPL2HUMANOID]/1000
# video_keypos = torch.from_numpy(np.load('ase/data/video/IMG 1204 - lateral stepping.npy'))[:, SMPL2HUMANOID]/1000
video_keypos[..., 1], video_keypos[..., 2] = video_keypos[..., 2], -video_keypos[..., 1]
video_keypos[:,:,2] -= video_keypos[:,:,2].min()

head_height = video_keypos[0, head_idx, 2]
left_foot_height = video_keypos[0, left_foot_idx, 2]
right_foot_height = video_keypos[0, right_foot_idx, 2]
video_body_height = head_height - min(left_foot_height, right_foot_height)

demo_keypos = dummy_replay_buffer.demo_store["keypos_obs"][0,0 ,:].view(24,3)
head_height = demo_keypos[head_idx, 2]
left_foot_height = demo_keypos[left_foot_idx, 2]
right_foot_height = demo_keypos[right_foot_idx, 2]
demo_body_height = head_height - min(left_foot_height, right_foot_height)

scaling_factor = video_body_height / demo_body_height

video_keypos -= video_keypos[0,0].clone()
video_keypos /= scaling_factor
video_keypos[:,:,2] -= video_keypos[:,:,2].min()


N = video_keypos.shape[0]
to_override_demo_id = 0
print(dummy_replay_buffer.demo_store["keypos_obs"][:N, to_override_demo_id, :].shape)
print(video_keypos.shape)
dummy_replay_buffer.demo_store["keypos_obs"][:N, to_override_demo_id, :] = video_keypos.view(-1, 24*3)
python ase/train_catchup_amp_v1_amass.py --visualize_policy --num_envs 16 --demo_motion_ids 0 --mlp_units 1024,512 --disc_units 1024,512 --disable_early_termination --keypos_lookahead --wandb_path dacmdp/catchup_amp_amass_v1/sbnhjuln/latest_model.pth
'''