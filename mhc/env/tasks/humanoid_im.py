# Imports

# Isaac Gym Imports
from isaacgym.torch_utils import *

# Python Imports
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm 
from enum import Enum
from munch import Munch
from munch import munchify

# Env Imports
from env.tasks.humanoid import Humanoid, dof_to_obs

# Utils Imports
from utils import torch_utils
from utils.inv_kin_motion_lib import InvKinMotionLib


# This is to be the simplest version where a MHC can be trained and evaluated. 

# --------------------------------------------
# ---------------Build Observation -----------
# --------------------------------------------

class ObsBuilder:
    """Namespace for observation building functions"""
    class __module: pass

    @staticmethod
    @torch.jit.script
    def build_full_state_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        if (local_root_obs):
            local_root_rot = quat_mul(heading_rot, root_rot)
        else:
            local_root_rot = root_rot
        root_rot_obs = torch_utils.quat_to_tan_norm(local_root_rot)
        
        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h
        
        local_root_vel = quat_rotate(heading_rot, root_vel)
        local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

        root_pos_expand = root_pos.unsqueeze(-2)
        local_key_body_pos = key_body_pos - root_pos_expand
        
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
        flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
        flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])
        local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
        flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
        
        dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

        non_localized_body_pos = key_body_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
        obs = torch.cat((root_pos, root_rot, root_vel, root_ang_vel, root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_pos, dof_obs, dof_vel, flat_local_key_pos, non_localized_body_pos), dim=-1)
        return obs

    @staticmethod
    def get_full_state_obs_from_motion_frame(task, motion_ids, motion_times):
        device = task._motion_lib._device
        mids = motion_ids.to(device)
        mts = motion_times.to(device)
        body_ids = torch.arange(task._rigid_body_pos.size(-2)).to(device)
        
        # Get all body positions as lookahaead
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = task._motion_lib.get_motion_state(mids, mts, key_body_ids = body_ids)
        full_state_obs = ObsBuilder.build_full_state_observations(root_pos.cuda(), root_rot.cuda(), root_vel.cuda(), root_ang_vel.cuda(),
                                            dof_pos.cuda(), dof_vel.cuda(), key_pos.cuda(),
                                            task.obs_flags["local_root_obs"], task.obs_flags["root_height_obs"],
                                            task._dof_obs_size, task._dof_offsets)
        
        info = {"root_pos": root_pos, 
                "root_rot": root_rot,
                "root_vel": root_vel,
                "root_ang_vel": root_ang_vel,
                "dof_pos": dof_pos,
                "dof_vel": dof_vel,
                "key_pos": key_pos
                }
        return full_state_obs, info
    
    @staticmethod
    def get_full_body_state_index_dict(dof_obs_size, num_dof, num_bodies, key_body_ids):
        """
        HardCoded for Reallusion Humanoid
        """

        split_state_indxs = {}

        split_state_indxs["root_pos"] = torch.arange(3).cuda()
        split_state_indxs["root_rot"] = torch.arange(3,7).cuda()
        split_state_indxs["root_vel"] = torch.arange(7,10).cuda()
        split_state_indxs["root_ang_vel"] = torch.arange(10,13).cuda()

        split_state_indxs["root_height"] = torch.arange(13,14).cuda()
        split_state_indxs["root_local_rot_obs"] = torch.arange(14,20).cuda()
        split_state_indxs["root_local_vel_obs"] = torch.arange(20,23).cuda()
        split_state_indxs["root_local_ang_vel_obs"] = torch.arange(23,26).cuda()

        split_state_indxs["dof_pos"] = torch.arange(26 , 26+num_dof).cuda()
        split_state_indxs["dof_pos_obs"] = torch.arange(26+num_dof , 26+num_dof+dof_obs_size).cuda()
        split_state_indxs["dof_vel"] = torch.arange(26+num_dof+dof_obs_size , 26+num_dof*2+dof_obs_size).cuda()
        split_state_indxs["dof_local_xyz"] = torch.arange(26+num_dof*2+dof_obs_size , 26+num_dof*2+dof_obs_size+num_bodies*3).cuda()
        split_state_indxs["dof_xyz"] = torch.arange(26+num_dof*2+dof_obs_size+num_bodies*3 , 26+num_dof*2+dof_obs_size+num_bodies*3*2).cuda()
        split_state_indxs["dof_local_key_xyz"] = split_state_indxs["dof_local_xyz"].reshape(-1,3)[key_body_ids].reshape(-1)
        split_state_indxs["dof_key_xyz"] = split_state_indxs["dof_xyz"].reshape(-1,3)[key_body_ids].reshape(-1)

        return split_state_indxs

    @staticmethod
    def get_amp_indices_for_full_state(dof_obs_size, num_dofs, num_bodies, key_body_ids):
        # ------------------- HARD CODED AMP INDICES -------------------
        # root_height.   root_local_rot_obs.   root_local_vel_obs.   root_local_ang_vel_obs.   dof_pos_obs.    dof_vel.   dof_local_key_xyz.
        # root_height.   root_local_rot_obs.   root_local_vel_obs.   root_local_ang_vel_obs.   ___________.    ________.   _________________.
        # ___________.   __________________.   _________________.    ______________________.    _l_body___.    _l_body_.   _l_body__________.
        # ___________.   __________________.   _________________.    ______________________.   _u_body_r__.    _u_body_r.  _u_body_r________.
        # ___________.   __________________.   _________________.    ______________________.   _u_body_l__.    _u_body_l.  _u_body_l________.

        split_state_indxs = ObsBuilder.get_full_body_state_index_dict(dof_obs_size, num_dofs, num_bodies, key_body_ids)
        return torch.cat([
                    split_state_indxs["root_height"],
                    split_state_indxs["root_local_rot_obs"],
                    split_state_indxs["root_local_vel_obs"],
                    split_state_indxs["root_local_ang_vel_obs"],
                    split_state_indxs["dof_pos_obs"],
                    split_state_indxs["dof_vel"],
                    split_state_indxs["dof_local_key_xyz"]])

    @staticmethod
    def get_lookahead_indices_for_full_state(dof_obs_size, num_dofs, num_bodies, key_body_ids):
        split_state_indxs = ObsBuilder.get_full_body_state_index_dict(dof_obs_size, num_dofs, num_bodies, key_body_ids)
        return torch.cat([split_state_indxs["root_pos"],
                    split_state_indxs["root_rot"],
                    split_state_indxs["root_vel"],
                    split_state_indxs["root_local_ang_vel_obs"],
                    split_state_indxs["dof_pos_obs"],
                    split_state_indxs["dof_vel"],
                    split_state_indxs["dof_local_xyz"],
                    split_state_indxs["dof_xyz"]])
    
    @staticmethod
    def get_keypos_indices_for_full_state(dof_obs_size, num_dofs, num_bodies, key_body_ids):
        split_state_indxs = ObsBuilder.get_full_body_state_index_dict(dof_obs_size, num_dofs, num_bodies, key_body_ids)
        return split_state_indxs["dof_xyz"]

    @staticmethod
    def get_split_indicies_for_lookahead_obs(dof_obs_size, num_dof, num_bodies, key_body_ids):
        # ------------------- HARD CODED RANDOM MOTIONS -------------------
        # root_pos *     root_rot *             root_vel *            root_local_ang_vel_obs 0    ___________.      ________.      ___________.
        # root_pos *     root_rot *             root_vel *            root_local_ang_vel_obs 0    [1] u_body_r.    [2] u_body_l.

        # ------------------- NATURAL COMPOSITION OF MOTIONS ---------------
        # [1] root_pos   [1] root_rot *         [1] root_vel *        [1] root_local_ang_vel_obs  [2] u_body_r.    [3] u_body_l.   [1] l_body 

        # ------------------- NATURAL FULL LOOKHEADS ---------------------
        # root_pos.      root_rot.              root_vel.             root_local_ang_vel_obs.     dof_pos_obs.      dof_vel.       dof_xyz.

        # ------------------- NATURAL PARTIAL LOOKHEADS ---------------------
        # root_pos.      root_rot.              root_vel.             root_local_ang_vel_obs.     ___________.      ________.      ___________.
        # ___________.   __________________.   _________________.     ______________________.     _l_body____.      _l_body_.      _l_body____.
        # ___________.   __________________.   _________________.     ______________________.     _u_body_r__.      _u_body_r.     _u_body_r__.
        # ___________.   __________________.   _________________.     ______________________.     _u_body_l__.      _u_body_l.     _u_body_l__.
        # ___________.   __________________.   _________________.     ______________________.     [1] u_body_r.    [2] u_body_l     [3] l_body.
        # ___________.   __________________.   _________________.     ______________________.     [1] hand, foot, sword shield (Keypos only)


        # Make sure this is inline with how the obs is built
        split_state_indxs = {}
        split_state_indxs["root_pos"] = torch.arange(3).cuda()
        split_state_indxs["root_rot"] = torch.arange(3,7).cuda()
        split_state_indxs["root_vel"] = torch.arange(7,10).cuda()
        split_state_indxs["root_local_ang_vel_obs"] = torch.arange(10,13).cuda()

        split_state_indxs["dof_pos_obs"] = torch.arange(13 , 13+dof_obs_size).cuda()
        split_state_indxs["dof_vel"] = torch.arange(13+dof_obs_size , 13+num_dof+dof_obs_size).cuda()
        split_state_indxs["dof_local_xyz"] = torch.arange(13+num_dof+dof_obs_size , 13+num_dof+dof_obs_size+num_bodies*3).cuda()
        split_state_indxs["dof_xyz"] = torch.arange(13+num_dof+dof_obs_size+num_bodies*3 , 13+num_dof+dof_obs_size+num_bodies*3*2).cuda()
        split_state_indxs["dof_local_key_xyz"] = split_state_indxs["dof_local_xyz"].reshape(-1,3)[key_body_ids].reshape(-1)
        split_state_indxs["dof_key_xyz"] = split_state_indxs["dof_xyz"].reshape(-1,3)[key_body_ids].reshape(-1)

        return split_state_indxs


# --------------------------------------------
# ---------------Reward Calculation-----------
# --------------------------------------------

class RewardUtils:

    """Namespace for reward building functions"""
    class __module: pass

    @staticmethod
    # @torch.jit.script
    def compute_facing_direction(root_pos, root_rot):
        """Computes the direction the character is facing based on root position and rotation.
        
        Args:
            root_pos: Root position tensor
            root_rot: Root rotation quaternion tensor
            
        Returns:
            root_facing_dir: Direction vector the character is facing
        """
        # Get the rotation around vertical axis
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        
        # Create reference direction along x-axis
        ref_dir = torch.zeros_like(root_pos) 
        ref_dir[..., 0] = 1.0
        
        # Rotate reference direction by heading to get facing direction
        root_facing_dir = quat_rotate(heading_rot, ref_dir)
        
        return root_facing_dir
    
    @staticmethod
    # @torch.jit.script
    def compute_root_imit_reward(root_pos, root_rot, root_vel, root_ang_vel, tar_pos, tar_rot, tar_vel, tar_ang_vel):
        # type: (Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        ###########################################################################
        # Calculate height reward
        ###########################################################################
        root_h = root_pos[:, 2:3] # Calculate height from root position
        tar_h = tar_pos[:, 2:3] # Calculate height from target position
        h_err = torch.abs(tar_h-root_h).squeeze() # Calculate height error
        h_reward = torch.exp(-8*h_err*h_err) # Calculate height reward
        ###########################################################################


        ###########################################################################
        # Calculate root facing direction
        ###########################################################################
        root_facing_dir = RewardUtils.compute_facing_direction(root_pos, root_rot) # Calculate root facing direction
        tar_facing_dir = RewardUtils.compute_facing_direction(tar_pos, tar_rot) # Calculate target facing direction
        facing_err = torch.sum(tar_facing_dir[..., 0:2] * root_facing_dir[..., 0:2], dim=-1) # Calculate facing error
        facing_reward = torch.clamp_min(facing_err, 0.0) # Calculate facing reward
        ###########################################################################
        

        ###########################################################################
        # Calculate velocity reward
        ###########################################################################
        u = root_vel[..., :2]
        v = tar_vel[..., :2]

        u_norm = torch.norm(u, dim=-1) + 1e-6
        v_norm = torch.norm(v, dim=-1) + 1e-6

        u_dot_v = torch.sum(u * v, dim=-1)
        u_proj_v = v * (u_dot_v / (v_norm**2)).unsqueeze(1)
        v_proj_u = u * (u_dot_v / (u_norm**2)).unsqueeze(1)

        vel_err = (torch.norm(u_proj_v - v, dim = -1) + torch.norm(v_proj_u - u, dim = -1))/2
        vel_dir_reward = torch.exp(-vel_err*vel_err)
        ###########################################################################


        ###########################################################################
        # Calculate angular velocity reward
        ###########################################################################
        u = root_ang_vel[..., :2]
        v = tar_ang_vel[..., :2]

        u_norm = torch.norm(u, dim=-1) + 1e-6
        v_norm = torch.norm(v, dim=-1) + 1e-6

        u_dot_v = torch.sum(u * v, dim=-1)
        u_proj_v = v * (u_dot_v / (v_norm**2)).unsqueeze(1)
        v_proj_u = u * (u_dot_v / (u_norm**2)).unsqueeze(1)

        ang_vel_err = (torch.norm(u_proj_v - v, dim = -1) + torch.norm(v_proj_u - u, dim = -1))/2
        ang_dir_reward = torch.exp(-ang_vel_err*ang_vel_err)
        ###########################################################################

        return h_reward, vel_dir_reward, ang_dir_reward,  facing_reward

    @staticmethod
    # @torch.jit.script
    def compute_dof_xyz_reward(curr_xyz_pos, tar_xyz_pos, dof_xyz_mask, dist_w):
        # type: (Tensor, Tensor, Tensor, int) -> Tensor
        """Compute reward based on xyz position differences between current and target poses.
        
        Args:
            curr_xyz_pos: Current xyz positions of joints # Num Env x Num Dofs x 3
            tar_xyz_pos: Target xyz positions of joints # Num Env x Num Dofs x 3
            dof_xyz_mask: Mask indicating which joints to ignore (1 = ignore) # Num Env x Num Dofs
            dist_w: Weight factor for distance penalty
            
        Returns:
            Reward value between 0 and 1, averaged over tracked joints
        """
        # Calculate euclidean distance between current and target positions
        eucl_dist = torch.norm(tar_xyz_pos - curr_xyz_pos, dim=-1)
        
        # Set large distance for masked joints to zero out their reward contribution
        eucl_dist[dof_xyz_mask] = 9999
        
        # Calculate reward as exponential of negative squared distance
        # Average only over unmasked joints, add small epsilon to avoid division by zero
        num_tracked_joints = torch.sum(~dof_xyz_mask, dim=-1) + 1e-5
        reward_sum = torch.sum(torch.exp(-dist_w * eucl_dist * eucl_dist), dim=-1) + 1e-5
        reward = reward_sum / num_tracked_joints
        
        return reward

    @staticmethod
    # @torch.jit.script
    def compute_dof_pos_reward(root_pos, tar_pos, dof_pos_mask, pos_diff_w):
        # type: (Tensor, Tensor, Tensor, int) -> Tensor
        """Compute reward based on position differences between current and target poses.
        
        Args:
            root_pos: Current positions of joints # Num Env x Num Dofs
            tar_pos: Target positions of joints # Num Env x Num Dofs
            dof_pos_mask: Mask indicating which joints to ignore (1 = ignore) # Num Env x Num Dofs
            pos_diff_w: Weight factor for position difference penalty
            
        Returns:
            Reward value between 0 and 1, averaged over tracked joints
        """
        # Calculate euclidean distance between current and target positions
        eucl_dist = torch.abs(tar_pos - root_pos)
        
        # Set large distance for masked joints to zero out their reward contribution
        eucl_dist[dof_pos_mask] = 9999
        
        # Calculate reward as exponential of negative squared distance
        # Average only over unmasked joints, add small epsilon to avoid division by zero
        num_tracked_joints = torch.sum(~dof_pos_mask, dim=-1) + 1e-5
        reward_sum = torch.sum(torch.exp(-pos_diff_w * eucl_dist * eucl_dist), dim=-1) + 1e-5
        reward = reward_sum / num_tracked_joints
        
        return reward

    @staticmethod
    def compute_benchmark_reward_using_full_body_state(full_body_state, tar_body_state, lookahead_obs, lookahead_obs_mask, dof_obs_size, num_dofs, num_bodies, key_body_ids, lookahead_obs_split_indxs_dict, ret_info = False, honor_mask_for_rewards = False):
        state_dim = full_body_state.size(-1)
        bs = full_body_state.size(0)
        # This mask is 1 if the elment is masked off in lookahead, even if one is not masked off it is not masked off
        # If it is masked off ,then it is excluded from the reward computatuion
        dof_xyz_mask_global = (torch.sum(lookahead_obs_mask[:, -num_bodies*3:].view(bs, -1, 3), dim = -1) == 3)
        dof_xyz_mask_local = (torch.sum(lookahead_obs_mask[:, -num_bodies*3*2:-num_bodies*3].view(bs, -1, 3), dim = -1) == 3)
        dof_key_xyz_mask_global = dof_xyz_mask_global[:, key_body_ids]
        dof_key_xyz_mask_local = dof_xyz_mask_local[:, key_body_ids]
        dof_xyz_mask = dof_xyz_mask_global * dof_xyz_mask_local
        dof_key_xyz_mask = dof_key_xyz_mask_global * dof_key_xyz_mask_local

        # Calculate MAsk ########################################################################### 
        lk_indxs_dict = lookahead_obs_split_indxs_dict
        # This mask is 1 if the elment is masked off in lookahead, even if one is not masked off it is not masked off
        # If it is masked off ,then it is excluded from the reward computatuion
        dof_xyz_mask_global = (torch.sum(lookahead_obs_mask[:, lk_indxs_dict["dof_xyz"]].view(bs, -1, 3), dim = -1) == 3)
        dof_xyz_mask_local = (torch.sum(lookahead_obs_mask[:, lk_indxs_dict["dof_local_xyz"]].view(bs, -1, 3), dim = -1) == 3)
        dof_vel_mask = lookahead_obs_mask[:, lk_indxs_dict["dof_vel"]]
        dof_vel_env_not_masked = (torch.sum(lookahead_obs_mask[:, lk_indxs_dict["dof_vel"]], dim =-1) == 0)

        dof_xyz_mask = dof_xyz_mask_global * dof_xyz_mask_local
        dof_key_xyz_mask_global = dof_xyz_mask_global[:, key_body_ids]
        dof_key_xyz_mask_local = dof_xyz_mask_local[:, key_body_ids]
        dof_key_xyz_mask = dof_key_xyz_mask_global * dof_key_xyz_mask_local

        # dont mask the xyz reward if dof vel is not masked
        # Exception is if the dof vel is not masked but the dof xyz is masked
        dof_xyz_mask[dof_vel_env_not_masked] = torch.zeros_like(dof_xyz_mask[dof_vel_env_not_masked]).type(torch.bool)
        dof_key_xyz_mask[dof_vel_env_not_masked] = torch.zeros_like(dof_key_xyz_mask[dof_vel_env_not_masked]).type(torch.bool)
        ##########################################################################################



        # lower_body_lk_masked 
        # upper_body_right_lk_masked
        # upper_body_left_lk_masked 
        # root_obs_lk_masked
        
        split_indxs = ObsBuilder.get_full_body_state_index_dict(dof_obs_size, num_dofs, num_bodies, key_body_ids)

        curr_root_xyz = full_body_state[:,split_indxs["root_pos"]]
        curr_root_rot = full_body_state[:,split_indxs["root_rot"]]
        curr_root_vel = full_body_state[:,split_indxs["root_vel"]]
        curr_root_ang_vel = full_body_state[:,split_indxs["root_local_ang_vel_obs"]]
        curr_dof_key_xyz = full_body_state[:,split_indxs["dof_local_key_xyz"]]
        global_curr_dof_xyz = full_body_state[:,split_indxs["dof_xyz"]]
        local_curr_dof_xyz = full_body_state[:,split_indxs["dof_local_xyz"]]


        tar_root_xyz = tar_body_state[:,split_indxs["root_pos"]]
        tar_root_rot = tar_body_state[:,split_indxs["root_rot"]]
        tar_root_vel = tar_body_state[:,split_indxs["root_vel"]]
        tar_root_ang_vel = tar_body_state[:,split_indxs["root_local_ang_vel_obs"]]
        tar_dof_key_xyz = tar_body_state[:,split_indxs["dof_local_key_xyz"]]
        global_tar_dof_xyz = tar_body_state[:,split_indxs["dof_xyz"]]
        local_tar_dof_xyz = tar_body_state[:,split_indxs["dof_local_xyz"]]

        # If Dof xyz is masked then we need not track this.
        # Only Track Joystick Commands
        # tar_dof_key_xyz[dof_xyz_lookahead_mask] = curr_dof_key_xyz[dof_xyz_lookahead_mask]

        # Calculate Rewards
        h_reward, v_reward, av_reward, o_reward = RewardUtils.compute_root_imit_reward(root_pos = curr_root_xyz,
                                                                            root_rot = curr_root_rot,
                                                                            root_vel = curr_root_vel,
                                                                            root_ang_vel = curr_root_ang_vel,
                                                                            tar_pos = tar_root_xyz,
                                                                            tar_rot = tar_root_rot,
                                                                            tar_vel = tar_root_vel,
                                                                            tar_ang_vel = tar_root_ang_vel,)

        # ref_dir = torch.zeros_like(curr_root_xyz)
        # ref_dir[..., 0] = 1.0
        # root_facing_dir = quat_rotate(torch_utils.calc_heading_quat(curr_root_rot), ref_dir.clone())
        # tar_facing_dir = quat_rotate(torch_utils.calc_heading_quat(tar_root_rot), ref_dir.clone())
        # facing_err = torch.sum(tar_facing_dir[..., 0:2] * root_facing_dir[..., 0:2], dim=-1)
        # facing_reward = torch.clamp_min(facing_err, 0.0)

        # dof_pos_reward = RewardUtils.compute_dof_pos_reward(curr_dof_pos, tar_dof_pos)
        # dof_vel_reward = RewardUtils.compute_dof_vel_reward(curr_dof_vel, tar_dof_vel)

        # eucl_dist = torch.norm(root_keypos - tar_keypos, dim = -1)
        # dist_w = 25
        # reward = torch.mean(torch.exp(-dist_w*eucl_dist*eucl_dist), dim = -1)
        
        def translate_dof_xyz_pos(dof_xyz_pos_flat):
            bs = dof_xyz_pos_flat.size(0)
            dof_xyz_pos = dof_xyz_pos_flat.view(bs,-1,3).clone()
            root_xyz_pos = dof_xyz_pos[:,0:1,:].clone()
            local_dof_xyz_pos = dof_xyz_pos - root_xyz_pos
            local_dof_xyz_pos[:,0,2] = root_xyz_pos[:,0,2]
            assert local_dof_xyz_pos[:,0,:2].sum().item() < 1e-3
            return local_dof_xyz_pos.view(bs, -1)

        shifted_global_curr_dof_xyz = translate_dof_xyz_pos(global_curr_dof_xyz)
        shifted_global_tar_dof_xyz = translate_dof_xyz_pos(global_tar_dof_xyz)

        shifted_joint_dist_global = torch.norm(shifted_global_curr_dof_xyz.view(bs, -1, 3)[:,:,:] - 
                                                shifted_global_tar_dof_xyz.view(bs, -1, 3)[:,:,:], dim = -1)
        joint_dist_local = torch.norm(local_curr_dof_xyz.view(bs, -1, 3)[:,:,:] - 
                                        local_tar_dof_xyz.view(bs, -1, 3)[:,:,:], dim = -1)
        joint_dist_global = torch.norm(global_curr_dof_xyz.view(bs, -1, 3)[:,:,:] - 
                                        global_tar_dof_xyz.view(bs, -1, 3)[:,:,:], dim = -1)
        root_dist_global = torch.norm(curr_root_xyz - tar_root_xyz, dim = -1)
        root_vel_error = torch.norm(curr_root_vel - tar_root_vel, dim = -1)


        mean_root_vel_error = torch.mean(root_vel_error, dim = -1)*1000
        root_dist_global = root_dist_global.view(-1)  * 1000

        if honor_mask_for_rewards: 
            _joint_dist_global, _joint_dist_local, _joint_dist_shifted = joint_dist_global.clone(), joint_dist_local.clone(), shifted_joint_dist_global.clone()
            _joint_dist_global[dof_xyz_mask] = 0
            _joint_dist_local[dof_xyz_mask] = 0
            _joint_dist_shifted[dof_xyz_mask] = 0

            mean_joint_dist_global = (torch.sum(_joint_dist_global, dim = -1) / (torch.sum(~dof_xyz_mask, dim = -1) + 1e-6) )*1000
            mean_joint_dist_local = (torch.sum(_joint_dist_local, dim = -1) / ( torch.sum(~dof_xyz_mask, dim = -1) + 1e-6) )*1000
            mean_joint_dist_shifted = (torch.sum(_joint_dist_shifted, dim = -1) /  (torch.sum(~dof_xyz_mask, dim = -1)+ 1e-6) )*1000

            max_joint_dist_global = torch.max(_joint_dist_global, dim = -1)[0]*1000 # in mm
            max_joint_dist_local = torch.max(_joint_dist_local, dim = -1)[0]*1000
            max_joint_dist_shifted = torch.max(_joint_dist_shifted, dim = -1)[0]*1000
        else:
            mean_joint_dist_global = torch.mean(joint_dist_global, dim = -1)*1000
            mean_joint_dist_local = torch.mean(joint_dist_local, dim = -1)*1000
            mean_joint_dist_shifted = torch.mean(shifted_joint_dist_global, dim = -1)*1000

            max_joint_dist_global = torch.max(joint_dist_global, dim = -1)[0]*1000 # in mm
            max_joint_dist_local = torch.max(joint_dist_local, dim = -1)[0]*1000
            max_joint_dist_shifted = torch.max(shifted_joint_dist_global, dim = -1)[0]*1000
                                    



        # Reward activation logic
        reward_dict =   {"h_reward" : h_reward,
                        "o_reward": o_reward,
                        "v_reward": v_reward,
                        "av_reward": av_reward,
                        "mean_joint_dist_global": mean_joint_dist_global,
                        "mean_joint_dist_local": mean_joint_dist_local,
                        "max_joint_dist_local": max_joint_dist_local, 
                        "mean_joint_dist_shifted": mean_joint_dist_shifted, 
                        "max_joint_dist_shifted": max_joint_dist_shifted,
                        "root_dist_global": root_dist_global,
                        "root_vel_error": root_vel_error
        }

        # total_reward = 0
        # ignore_mask = h_reward < 1000 # All True, ignore if 0 continue if 1
        # for k, r_item in reward_dict.items():
        #     total_reward = total_reward + r_item["weight"] * r_item["value"] * ignore_mask
        #     ignore_mask = ignore_mask * (r_item["value"] > r_item["threshold"]) # If 1 continue if 0
        
        # if not ret_info:
        #     return total_reward
        # else:
        #     return total_reward, {k:r_item["value"] for k, r_item in reward_dict.items()}
        return reward_dict

    @staticmethod
    def compute_imit_reward_using_full_body_state(full_body_state, tar_body_state, lookahead_obs,
                                                lookahead_obs_mask, dof_obs_size, num_dofs, num_bodies,
                                                    key_body_ids, helper_indexes, ret_info = False, assert_check = False, 
                                                    reward_config = None):

        state_dim = full_body_state.size(-1)
        bs = full_body_state.size(0)
        
        def translate_dof_xyz_pos(dof_xyz_pos_flat):
            bs = dof_xyz_pos_flat.size(0)
            dof_xyz_pos = dof_xyz_pos_flat.view(bs,-1,3).clone()
            root_xyz_pos = dof_xyz_pos[:,0:1,:].clone()
            local_dof_xyz_pos = dof_xyz_pos - root_xyz_pos
            local_dof_xyz_pos[:,0,2] = root_xyz_pos[:,0,2]
            assert local_dof_xyz_pos[:,0,:2].sum().item() < 1e-3
            return local_dof_xyz_pos.view(bs, -1)

        # Calculate MAsk ########################################################################### 
        lk_indxs_dict = reward_config["lookahead_obs_split_indxs_dict"]
        # This mask is 1 if the elment is masked off in lookahead, even if one is not masked off it is not masked off
        # If it is masked off ,then it is excluded from the reward computatuion
        dof_xyz_mask_global = (torch.sum(lookahead_obs_mask[:, lk_indxs_dict["dof_xyz"]].view(bs, -1, 3), dim = -1) == 3)
        dof_xyz_mask_local = (torch.sum(lookahead_obs_mask[:, lk_indxs_dict["dof_local_xyz"]].view(bs, -1, 3), dim = -1) == 3)
        dof_vel_mask = lookahead_obs_mask[:, lk_indxs_dict["dof_vel"]]
        dof_vel_env_not_masked = (torch.sum(lookahead_obs_mask[:, lk_indxs_dict["dof_vel"]], dim =-1) == 0)

        dof_xyz_mask = dof_xyz_mask_global * dof_xyz_mask_local
        dof_key_xyz_mask_global = dof_xyz_mask_global[:, key_body_ids]
        dof_key_xyz_mask_local = dof_xyz_mask_local[:, key_body_ids]
        dof_key_xyz_mask = dof_key_xyz_mask_global * dof_key_xyz_mask_local

        # dont mask the xyz reward if dof vel is not masked
        # Exception is if the dof vel is not masked but the dof xyz is masked
        dof_xyz_mask[dof_vel_env_not_masked] = torch.zeros_like(dof_xyz_mask[dof_vel_env_not_masked]).type(torch.bool)
        dof_key_xyz_mask[dof_vel_env_not_masked] = torch.zeros_like(dof_key_xyz_mask[dof_vel_env_not_masked]).type(torch.bool)
        ##########################################################################################
        
        # Split the full body state
        # ------------------------------------------------------------------------------------------------
        split_indxs = ObsBuilder.get_full_body_state_index_dict(dof_obs_size, num_dofs, num_bodies, key_body_ids)
        curr_root_xyz = full_body_state[:,split_indxs["root_pos"]]
        curr_root_rot = full_body_state[:,split_indxs["root_rot"]]
        curr_root_vel = full_body_state[:,split_indxs["root_vel"]]
        curr_root_ang_vel = full_body_state[:,split_indxs["root_local_ang_vel_obs"]]
        curr_dof_xyz = full_body_state[:,split_indxs["dof_local_xyz"]]
        curr_dof_key_xyz = full_body_state[:,split_indxs["dof_local_key_xyz"]]
        curr_dof_pos = full_body_state[:,split_indxs["dof_pos"]]
        global_curr_dof_xyz = full_body_state[:,split_indxs["dof_xyz"]]
        shifted_global_curr_dof_xyz = translate_dof_xyz_pos(global_curr_dof_xyz)
        # ------------------------------------------------------------------------------------------------

        # Split the target body state
        # ------------------------------------------------------------------------------------------------  
        tar_root_xyz = tar_body_state[:,split_indxs["root_pos"]]
        tar_root_rot = tar_body_state[:,split_indxs["root_rot"]]
        tar_root_vel = tar_body_state[:,split_indxs["root_vel"]]
        tar_root_ang_vel = tar_body_state[:,split_indxs["root_local_ang_vel_obs"]]
        tar_dof_xyz = tar_body_state[:,split_indxs["dof_local_xyz"]] # this is localized to the heading. 
        tar_dof_key_xyz = tar_body_state[:,split_indxs["dof_local_key_xyz"]] # this is localized to the heading.
        tar_dof_pos = tar_body_state[:,split_indxs["dof_pos"]]
        global_tar_dof_xyz = tar_body_state[:,split_indxs["dof_xyz"]]
        shifted_global_tar_dof_xyz = translate_dof_xyz_pos(global_tar_dof_xyz) # this is only translated to the root. Preserves the relative orientation.
        # ------------------------------------------------------------------------------------------------
        
        # ------------------------------------------------------------------------------------------------
        # Assert Check
        # ------------------------------------------------------------------------------------------------
        if assert_check:
            # if root not masked both must be same
            root_not_masked = torch.sum(lookahead_obs[:, 2:10], dim = -1) != 0
            assert torch.sum(tar_body_state[:,2:10][root_not_masked] - lookahead_obs[:, 2:10][root_not_masked] ) == 0

            ang_vel_not_masked = torch.sum(lookahead_obs[:, 9:12], dim = -1) != 0
            assert torch.sum(tar_body_state[:,23:26][root_not_masked] - lookahead_obs[:, 10:13][root_not_masked] ) == 0
            
            # IF local xyz not masked both must be same
            dof_key_xyz_mask_local_not = ~dof_key_xyz_mask_local
            local_key_diff = torch.sum(tar_body_state[:,split_indxs["dof_local_key_xyz"]].view(bs, -1, 3) -
                                        lookahead_obs[:, -num_bodies*3*2:-num_bodies*3].view(bs, -1, 3)[:, key_body_ids, :], 
                                        dim = -1)
            assert torch.sum(local_key_diff[dof_key_xyz_mask_local_not]) == 0
        # ------------------------------------------------------------------------------------------------
        

        # Calculate Root Imitation Rewards
        # ------------------------------------------------------------------------------------------------
        h_reward, v_reward, av_reward, o_reward = RewardUtils.compute_root_imit_reward(root_pos = curr_root_xyz,
                                                                            root_rot = curr_root_rot,
                                                                            root_vel = curr_root_vel,
                                                                            root_ang_vel = curr_root_ang_vel,
                                                                            tar_pos = tar_root_xyz,
                                                                            tar_rot = tar_root_rot,
                                                                            tar_vel = tar_root_vel,
                                                                            tar_ang_vel = tar_root_ang_vel,
                                                                            )
        # ------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------
        # Calculate dof xyz reward from the joint xyz positions 
        # ------------------------------------------------------------------------------------------------
        # Use translated joint xyz positions for reward. This will preserve the relative orientation of the joints.
        # ------------------------------------------------------------------------------------------------
        if reward_config["use_shifted_xyz_reward"]:
            dof_xyz_reward = RewardUtils.compute_dof_xyz_reward(shifted_global_curr_dof_xyz.view(bs, -1, 3), 
                                                    shifted_global_tar_dof_xyz.view(bs, -1, 3), 
                                                    dof_xyz_mask.view(bs, -1), 
                                                    dist_w = reward_config["dof_xyz_reward_weight"])
        # ------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------
        # Use the Key joints only for the reward. The rest are not tracked. the xyz is in local oritation frame. 
        # ------------------------------------------------------------------------------------------------
        elif reward_config["keypos_reward_only"]:
            dof_xyz_reward = RewardUtils.compute_dof_xyz_reward(curr_dof_key_xyz.view(bs, -1, 3), 
                                                    tar_dof_key_xyz.view(bs, -1, 3), 
                                                    dof_key_xyz_mask.view(bs, -1), 
                                                    dist_w = reward_config["dof_xyz_reward_weight"])
        # ------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------
        # Use all joints for the reward. The key joints are given more weight..the xyz is in local oritation frame. 
        # ------------------------------------------------------------------------------------------------
        elif reward_config["keypos_big_weight_reward"]:
            dof_xyz_reward = RewardUtils.compute_dof_xyz_reward(curr_dof_xyz.view(bs, -1, 3), 
                                                    tar_dof_xyz.view(bs, -1, 3), 
                                                    dof_xyz_mask.view(bs, -1), 
                                                    dist_w = reward_config["dof_xyz_reward_weight"])
            
            dof_key_xyz_reward = RewardUtils.compute_dof_xyz_reward(curr_dof_key_xyz.view(bs, -1, 3), 
                                                    tar_dof_key_xyz.view(bs, -1, 3), 
                                                    dof_key_xyz_mask.view(bs, -1), 
                                                    dist_w = reward_config["dof_xyz_reward_weight"])

            dof_xyz_reward = (dof_xyz_reward + 3*dof_key_xyz_reward)/4
        # ------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------
        # Use all joints for the reward. All joints are given equal weight. the xyz is in local orientation frame.
        # ------------------------------------------------------------------------------------------------
        else:
            dof_xyz_reward = RewardUtils.compute_dof_xyz_reward(curr_dof_xyz.view(bs, -1, 3), 
                                                    tar_dof_xyz.view(bs, -1, 3), 
                                                    dof_xyz_mask.view(bs, -1), 
                                                    dist_w = reward_config["dof_xyz_reward_weight"])
        # ------------------------------------------------------------------------------------------------


        # ------------------------------------------------------------------------------------------------
        # Cascading Reward Activation Logic
        # The next set of rewards are activated if the previous set of rewards are greater than the threshold.
        # ------------------------------------------------------------------------------------------------
        reward_activation_config =   {"h_reward" : {"value": h_reward, "weight": 0.1 , "threshold":0.85},
                                    "o_reward": {"value": o_reward, "weight": 0.1 , "threshold":0.85},
                                    "v_reward": {"value": v_reward, "weight": 0.1 , "threshold":0.75},
                                    "av_reward": {"value": av_reward, "weight": 0.1 , "threshold":0.75},
                                    # "dof_pos_reward": {"value": dof_pos_reward, "weight": 0.2 , "threshold":0},
                                    # "dof_vel_reward": {"value": dof_vel_reward, "weight": 0.2 , "threshold":0},
                                    "dof_xyz_reward": {"value": dof_xyz_reward, "weight": 0.6 , "threshold":0},
                    }

        total_reward = 0
        cascading_activation_mask = torch.ones_like(h_reward, dtype=torch.bool) 
        for k, r_item in reward_activation_config.items():
            total_reward = total_reward + r_item["weight"] * r_item["value"] * cascading_activation_mask
            cascading_activation_mask = cascading_activation_mask * (r_item["value"] > r_item["threshold"]) # If 1 continue if 0
        # ------------------------------------------------------------------------------------------------
        

        reward_info = {k: r_item["value"] for k, r_item in reward_activation_config.items()}
        return (total_reward, reward_info) if ret_info else total_reward


# -------------------------------------------------------------------
# --------------- Demo Store for Target Lookahead Caching -----------
# --------------- Calculate one then lookup -------------------------
# -------------------------------------------------------------------
class DemoStoreUtils:

    def __init__(self):
        pass
    
    @staticmethod
    def get_demo_store(task, device):
        """
        """
        demo_store_len = 1500
        num_demos = task._motion_lib.num_motions()
        keypos_obs_dim = task._num_keypos_obs_per_step #?
        full_state_obs_dim = task._full_state_obs_dim #?
        amp_obs_dim = task._num_amp_obs_per_step #?
        action_dim = task._num_actions #?
        num_key_pos = len(task._key_body_ids) #?
        device = device

        # We pad the tensor here because we wish to calculate lookahead till the last frame of the demo.
        def get_flat_tensor(key, demo_dicts):
            return torch.concat([F.pad(d[key].transpose(0,1)[None], (0,25), "replicate").squeeze() for d in demo_dicts], dim = -1).to(device)

        DEMO_DICTS = DemoStoreUtils.extract_motion_sequences_from_motion_lib(task, horizon = demo_store_len)

        # ------------------------------------------------------------------------------------------------
        # Setup Demo Store
        # ------------------------------------------------------------------------------------------------
        demo_store = {}
        for k in ["amp_obs", "full_state_obs", "keypos_obs", "root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"]:
            demo_store[k] = get_flat_tensor(k, DEMO_DICTS)
        demo_store["length_starts"] = torch.cumsum(torch.LongTensor([0] + [d["amp_obs"].size(0)+25 for d in DEMO_DICTS[:-1]]).to(device), 0).cuda()

        demo_store["demo_lengths"] = torch.zeros(num_demos).long().to(device)
        demo_store["demo_names"] = [""]*num_demos
        for motion_id in range(len(DEMO_DICTS)):
            demo_store["demo_lengths"][motion_id] = int(task._motion_lib._motion_lengths[motion_id]/task.dt)
            demo_store["demo_names"][motion_id] = task._motion_lib._motion_files[motion_id].split("/")[-1].replace(".npy","")
        # ------------------------------------------------------------------------------------------------
        return demo_store

    @staticmethod
    def extract_motion_sequences_from_motion_lib(task, horizon, verbose = True):
        
        # For mimicry 
        def _get_mimicry_seq(task, motion_id, verbose = False):
            
            # Get the motion length and the motion ids and times
            motion_len_dt = task._motion_lib._motion_lengths[motion_id]
            motion_len = int(motion_len_dt/task.dt)
            motion_ids = torch.LongTensor([motion_id]*motion_len).cuda()
            motion_times = torch.FloatTensor(np.arange(motion_len)*task.dt).cuda()

            # Get the full state observations
            # Is designed in sucha away tah last 17 things still mean the main keypoint joints
            full_state_obs_demo, full_state_obs_info = ObsBuilder.get_full_state_obs_from_motion_frame(task, motion_ids, motion_times)
            assert full_state_obs_demo.size(-1) == task._full_state_obs_dim, f"Assertion failed found size: {full_state_obs_demo.size(-1)}"

            # Get the AMP observations
            amp_obs_demo = full_state_obs_demo[:, task._full_state_amp_obs_indxs]
            assert amp_obs_demo.size(-1) == task._amp_obs_dim, f"Assertion failed found size: {amp_obs_demo.size(-1)}"

            # Get the keypos observations
            key_pos = full_state_obs_info["key_pos"]
            keypos_obs_demo = key_pos.view(key_pos.size(0), -1) 
            assert keypos_obs_demo.size(-1) == task._keypos_obs_dim, f"Assertion failed found size: {keypos_obs_demo.size(-1)}"
            

            if verbose:
                motion_name = task._motion_lib._motion_files[motion_id].split("/")[-1].replace(".npy","")
                print(f"Motion Loaded. \tId: {motion_id}\t Name: {motion_name}\t Shape: {amp_obs_demo.shape}".expandtabs(25))

            out_dict = {"amp_obs": amp_obs_demo, # Shape: (motion_len, 3+4+3+amp_obs_dim)
                        "full_state_obs": full_state_obs_demo,  # Shape: (motion_len, 3+4+3+amp_obs_dim)
                        "keypos_obs": keypos_obs_demo, 
                        "root_pos": full_state_obs_info["root_pos"], # Shape: (motion_len, 3)
                        "root_rot": full_state_obs_info["root_rot"], # Shape: (motion_len, 4)
                        "root_vel": full_state_obs_info["root_vel"], # Shape: (motion_len, 3) 
                        "root_ang_vel": full_state_obs_info["root_ang_vel"], # Shape: (motion_len, 3)
                        "dof_pos": full_state_obs_info["dof_pos"], # Shape: (motion_len, dof_obs_size)
                        "dof_vel": full_state_obs_info["dof_vel"], # Shape: (motion_len, dof_obs_size)
                        "key_pos": full_state_obs_info["key_pos"] # Shape: (motion_len, 17, 3)
                        }
            
            return out_dict
   
        DEMO_DICTS = [_get_mimicry_seq(task, i) for i in tqdm(range(task._motion_lib.num_motions()), "Building Demo Dicts")]
        return DEMO_DICTS


# -------------------------------------------------------------------
# --------------- Mimic Environment ---------------------------------
# -------------------------------------------------------------------

class HumanoidAMPRobust(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # HouseKeeping Code
        self._verbose_reset = False

        self.num_envs = cfg["env"]["numEnvs"]

        self.task_properties = Munch({})
        # self.task_properties.num_envs = cfg["env"]["numEnvs"]

        self.agent_properties = Munch({})
        self.agent_properties.key_body_names = cfg["env"]["keyBodies"]

        self.reward_properties = Munch({})
        self.reward_properties.keypos_reward_only = cfg["env"]["keyposRewardOnly"] # Uses only key joints for reward (Mode 0)
        self.reward_properties.keypos_big_weight_reward = cfg["env"]["keypos_big_weight_reward"] # Uses all joints for reward but gives key joints more weight (Mode 1)
        self.reward_properties.use_shifted_xyz_reward = cfg["env"]["use_shifted_xyz_reward"] # Uses shifted xyz for the reward, has info for local orientation (Mode 3)
        self.reward_properties.dof_xyz_reward_w = cfg["env"]["dof_xyz_reward_w"] # weight for the xyz reward


        self._enable_body_pos_obs = cfg["env"]["enableBodyPosObs"] #RobustAdditionalCodeLine
        self._state_init_rotate = cfg["env"]["stateRotate"] #RobustAdditionalCodeLine
        self._state_demo_rotate = cfg["env"]["stateDemoRotate"] #RobustAdditionalCodeLine
        self._switch_demos_within_episode = cfg["env"]["switchDemosWithinEpisode"] #RobustAdditionalCodeLine
        self._to_mimic_demo_ids = cfg["env"]["demoMotionIds"] #RobustAdditionalCodeLine
        self._height_offset = cfg["env"]["heightOffset"] #RobustAdditionalCodeLine
        self._penalty_multiplyer = cfg["env"]["penaltyMultiplyer"] #RobustAdditionalCodeLine
        self._asset_file_name = cfg["env"]["asset"]["assetFileName"]

        self._state_init = HumanoidAMPRobust.StateInit[cfg["env"]["stateInit"]]
        self._demo_init = HumanoidAMPRobust.StateInit[cfg["env"]["demoInit"]]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        self._energy_penalty_activated = cfg["env"]["ENERGY_PENALTY_ACTIVATED"]
        # self._cache_horizon = cfg["env"]["CACHE_HORIZON"]
        self._motion_lib_device = cfg["env"]["MOTIONLIB_DEVICE"]
        # self._use_only_keypos_for_amp_obs = cfg["env"]["useOnlyKeyposForAMP"]
        self._motion_file = cfg['env']['motion_file']
        self._motion_is_sampled_by_frames = cfg["env"]["motion_is_sampled_by_frames"]
        # self._random_dir_speed_lookahead_share = cfg["env"]["random_dir_speed_lookahead_share"]
        # self._compose_demo_targets_share = cfg["env"]["compose_demo_targets_share"]
        self._enable_lookahead_mask = cfg["env"]["enable_lookahead_mask"]
        self._use_predefined_demo_weights_for_sampling = cfg["env"]["use_predefined_demo_weights_for_sampling"]
        self._start_demo_at_agent_state_prob = cfg["env"]["start_demo_at_agent_state_prob"]

        self._disable_lk_mask_0 = cfg["env"]["disable_lk_mask_0"]
        self._disable_lk_mask_1 = cfg["env"]["disable_lk_mask_1"]
        self._disable_lk_mask_2 = cfg["env"]["disable_lk_mask_2"]
        self._disable_lk_mask_3 = cfg["env"]["disable_lk_mask_3"]
        self._disable_lk_mask_4 = cfg["env"]["disable_lk_mask_4"]
        self._disable_lk_mask_5 = cfg["env"]["disable_lk_mask_5"]
        self._disable_lk_mask_6 = cfg["env"]["disable_lk_mask_6"]

        self.num_lookahead_channels = 5
        self.lk_embedding_dim =  cfg["env"]["lk_embedding_dim"]

        # self._enable_intra_channel_masking = False
        self._enable_lk_jl_mask =  cfg["env"]["enable_lk_jl_mask"]
        self._enable_lk_channel_mask =  cfg["env"]["enable_lk_channel_mask"]
        self._use_predefined_jl_mask = False
        self._predefined_jl_mask_joint_prob = 0
        self._jl_mask_prob = 0.5

        self._uniform_targets = False
        self._use_predefined_lookahead_mask = False
        self._predefined_lookahead_mask = torch.zeros(self.lk_embedding_dim).type(torch.bool)

        ############################################################################################################################################
        self._use_predefined_targets = False
        self._predefined_target_demo_start_motion_ids = torch.zeros(self.num_envs).type(torch.long).cuda() #RobustAdditionalCodeLine
        self._predefined_target_demo_start_motion_times = torch.zeros(self.num_envs).cuda() #RobustAdditionalCodeLine
        self._predefined_target_demo_start_rotations = torch.cat((torch.zeros(self.num_envs, 3), torch.ones(self.num_envs, 1)), dim = -1).cuda() #RobustAdditionalCodeLine
        ############################################################################################################################################

        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        # Set the observation space for AMP
        # if self._use_only_keypos_for_amp_obs:
        #     self._num_amp_obs_per_step = 3*self._key_body_ids.size(0) #RobustAdditionalCodeLine

        self._amp_obs_dim = self._num_amp_obs_per_step
        print(f"AMP OBS DIM : CALCULATED DYNAMICALLY : {self._amp_obs_dim}")
        # assert self._num_amp_obs_per_step == self._amp_obs_dim, f"{self._num_amp_obs_per_step=}, {self._amp_obs_dim=}" #RobustAdditionalCodeLine

        # Set the observation space for Lookahead
        self.num_bodies = self._rigid_body_pos.size(-2)
        self._num_keypos_obs_per_step = self.num_bodies*3 #RobustAdditionalCodeLine
        self._keypos_obs_dim = self._num_keypos_obs_per_step #RobustAdditionalCodeLine
        print(f"KEYPOS OBS DIM : CALCULATED DYNAMICALLY : {self._keypos_obs_dim}")
        # assert self._num_keypos_obs_per_step == self._keypos_obs_dim, f"{self._num_keypos_obs_per_step=}, {self._keypos_obs_dim=}" #RobustAdditionalCodeLine

        self._lookahead_timesteps = 10 #RobustAdditionalCodeLine
        self._lookahead_obs_dim = 13 +  self._dof_obs_size + self._dof_offsets[-1] + 2 * 3 * self.num_bodies
        self._full_state_obs_dim = 13 + 13 + self._dof_offsets[-1] + self._dof_obs_size + self._dof_offsets[-1]  +  3 * self.num_bodies * 2

        self._global_demo_start_motion_ids = torch.zeros(self.num_envs).type(torch.long).cuda() #RobustAdditionalCodeLine
        self._global_demo_start_rotations = torch.cat((torch.zeros(self.num_envs, 3), torch.ones(self.num_envs, 1)), dim = -1).cuda() #RobustAdditionalCodeLine
        self._global_demo_start_motion_times = torch.zeros(self.num_envs).cuda() #RobustAdditionalCodeLine
        self._global_demo_lookahead_mask = torch.zeros(self.num_envs, self._lookahead_obs_dim).type(torch.bool).cuda() # 1 if needs to be masked out from lookahead
        self._global_demo_lookahead_mask_env2idxmap = torch.zeros(self.num_envs).type(torch.LongTensor).cuda() 
        self._fixed_demo_lookahead_mask_env2idxmap_flag = False

        self._global_agent_start_motion_ids = torch.zeros(self.num_envs).type(torch.long).cuda() #RobustAdditionalCodeLine
        self._global_agent_start_rotations = torch.cat((torch.zeros(self.num_envs, 3), torch.ones(self.num_envs, 1)), dim = -1).cuda() #RobustAdditionalCodeLine
        self._global_agent_start_motion_times = torch.zeros(self.num_envs).cuda() #RobustAdditionalCodeLine

        self._global_amp_obs_mask = torch.zeros(self.num_envs, self._amp_obs_dim).type(torch.bool).cuda()

        # Setup Other Things
        self._load_motion(self._motion_file)
        self.range_idxs = torch.arange(self._lookahead_timesteps).repeat(self.num_envs*2).cuda() #RobustAdditionalCodeLine
        self.ts_range = torch.arange(5000).cuda()

        # ------------------------------------------------------------------------------------------------
        # Setup Target Buffers
        # ------------------------------------------------------------------------------------------------
        # TThis buffer holds flattened state of all **amp obs** for future lookahead timesteps of each env. (num_envs, lookahead_timesteps * num_amp_obs_per_step)
        self._tar_flat_amp_obs = torch.zeros(self.num_envs, self._lookahead_timesteps * self._num_amp_obs_per_step).cuda() #RobustAdditionalCodeLine
        # This buffer holds flattened state of all **lookahead obs** for future lookahead timesteps of each env. (num_envs, lookahead_timesteps * lookahead_obs_dim)
        self._tar_flat_lookahead_obs = torch.zeros(self.num_envs, self._lookahead_timesteps * self._lookahead_obs_dim).cuda() #RobustAdditionalCodeLine
        # This buffer holds flattened state of all **full state obs** for future lookahead timesteps of each env. (num_envs, lookahead_timesteps * full_state_obs_dim)
        self._tar_flat_full_state_obs = torch.zeros(self.num_envs, self._lookahead_timesteps * self._full_state_obs_dim).cuda() #RobustAdditionalCodeLine
        # ------------------------------------------------------------------------------------------------
        # Setup Current Buffers
        # ------------------------------------------------------------------------------------------------
        # This buffer holds current **amp obs** for each env. (num_envs, num_amp_obs_steps, num_amp_obs_per_step)
        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        # This buffer holds current **keypos obs** for each env. (num_envs, num_keypos_obs_per_step)
        self._curr_keypos_obs_buf = torch.zeros((self.num_envs, self._num_keypos_obs_per_step), device=self.device, dtype=torch.float) #RobustAdditionalCodeLine
        # This buffer holds current **full state obs** for each env. (num_envs, full_state_obs_dim)
        self._curr_full_state_obs_buf = torch.zeros((self.num_envs, self._full_state_obs_dim), device=self.device, dtype=torch.float) #RobustAdditionalCodeLine
        # ------------------------------------------------------------------------------------------------

        print("curr amp obs buf shape", self._curr_amp_obs_buf.shape)
        print("curr full state obs buf shape", self._curr_full_state_obs_buf.shape)

        self._prev_action_buf = torch.zeros((self.num_envs, self._num_actions), device=self.device, dtype=torch.float).cuda()
        self._amp_obs_demo_buf = None

        # RobustAdditionalCodeBlock
        # -------------------------------------------------------
        self._demo_motion_weights = torch.zeros_like(self._motion_lib._motion_weights).to(self.device)
        self._demo_motion_weights[self._to_mimic_demo_ids] = 1
        self._demo_motion_weights =  self._demo_motion_weights / torch.sum(self._demo_motion_weights)
        if self._use_predefined_demo_weights_for_sampling:
            self._demo_motion_weights[:] = self._motion_lib._motion_weights
            print("Using predefined demo weights for sampling")

        self._demo_tar_reset_steps = torch.zeros(self.num_envs, dtype = torch.int32, device =self.device)
        self._demo_tar_reset_steps_max = 240
        self._demo_tar_reset_steps_min = 120

        self._start_state_motion_weights = self._motion_lib._motion_weights.clone().to(self.device)
        self._start_state_motion_weights[self._to_mimic_demo_ids] = torch.sum(self._start_state_motion_weights)
        self._start_state_motion_weights =  self._start_state_motion_weights / torch.sum(self._start_state_motion_weights)

        # Manually recalculating this because the num of frames may be different. this is calculated as per the number of frames in demo store
        self._motion_max_steps = (self._motion_lib._motion_lengths/self.dt).type(torch.LongTensor).cuda()
        self._req_height = None
        self._random_speed_min = 0
        self._random_speed_max = 3
        self._global_random_dir_speed_lookahead_bucket = self._sample_random_lookahead_diff_orient_speed(self.num_envs).cuda() #RobustAdditionalCodeLine

        self._random_dir_speed_lookahead_share = cfg["env"]["random_dir_speed_lookahead_share"]
        self._demo_env_offset = self._init_random_dir_env_indexes(start_at = 0)

        self._random_dir_speed_mimic_upper_prob = 0.8
        self._init_random_dir_upper_env_indexes()

        self._compose_demo_targets_share = cfg["env"]["compose_demo_targets_share"]
        self._demo_env_offset = self._init_compose_env_indexes(start_at = self._demo_env_offset)

        # -------------------------------------------------------

        self._all_env_ids = torch.LongTensor(range(self.num_envs)).cuda()
        self._full_state_amp_obs_indxs = ObsBuilder.get_amp_indices_for_full_state(self._dof_obs_size, self._dof_offsets[-1], self.num_bodies, self._key_body_ids)
        self._full_state_keypos_obs_indxs = ObsBuilder.get_keypos_indices_for_full_state(self._dof_obs_size, self._dof_offsets[-1], self.num_bodies, self._key_body_ids)
        self._full_state_lookahead_obs_indxs = ObsBuilder.get_lookahead_indices_for_full_state(self._dof_obs_size, self._dof_offsets[-1], self.num_bodies, self._key_body_ids)

        # Demo Store:
        self._motion_lib.demo_store = DemoStoreUtils.get_demo_store(self, device = self._motion_lib_device)

        self._init_obs_mask_indxs()
        self._init_lookahead_mask_pool()
        self._init_amp_obs_mask_pool()

        return

    def _init_random_dir_env_indexes(self, start_at):
        n_env_slots = int(self.num_envs*self._random_dir_speed_lookahead_share)
        self._random_dir_speed_env_idxs = torch.arange(start_at, start_at  + n_env_slots).cuda()

        self._random_dir_speed_env_idx_mask = torch.zeros(self.num_envs).type(torch.bool).cuda()
        self._random_dir_speed_env_idx_mask[self._random_dir_speed_env_idxs] = True

        return start_at + n_env_slots

    def _init_random_dir_upper_env_indexes(self):
        mimic_upper_mask = torch.rand(self.num_envs).cuda() < self._random_dir_speed_mimic_upper_prob
        self._random_dir_speed_upper_body_env_idx_mask = mimic_upper_mask * self._random_dir_speed_env_idx_mask
        self._random_dir_speed_upper_body_env_idxs = torch.nonzero(self._random_dir_speed_upper_body_env_idx_mask).view(-1)
        return

    def _init_compose_env_indexes(self, start_at):
        n_env_slots = int(self.num_envs*self._compose_demo_targets_share)
        self._compose_demo_targets_env_idxs = torch.arange(start_at, start_at  + n_env_slots).cuda()

        self._compose_demo_targets_env_map = torch.zeros(self.num_envs).type(torch.bool).cuda()
        self._compose_demo_targets_env_map[self._compose_demo_targets_env_idxs] = True

        return start_at + n_env_slots

    def _init_obs_mask_indxs(self):

        self._full_state_split_indxs = ObsBuilder.get_full_body_state_index_dict(self._dof_obs_size, self._dof_offsets[-1], self.num_bodies, self._key_body_ids)
        self._lookahead_obs_split_indxs = ObsBuilder.get_split_indicies_for_lookahead_obs(self._dof_obs_size, self._dof_offsets[-1], self.num_bodies, self._key_body_ids)

        # TODO Move this to a config file.
        if "amp_humanoid.xml" in self._asset_file_name:
            jb_names_dict = munchify({
                        "joints": {
                            "upper_left": ['left_shoulder', 'left_elbow'],
                            "upper_right": ['right_shoulder', 'right_elbow'],
                            "upper_torso": ['abdomen', 'neck'],
                            "lower_left": ['left_hip', 'left_knee', 'left_ankle'],
                            "lower_right": ['right_hip', 'right_knee', 'right_ankle']
                        },
                        "rigid_body": {
                            "upper_left": ['left_upper_arm', 'left_lower_arm', 'left_hand'],
                            "upper_right": ['right_upper_arm', 'right_lower_arm', 'right_hand'],
                            "upper_torso": ['pelvis', 'torso', 'head'],
                            "lower_left": ['left_thigh', 'left_shin', 'left_foot'],
                            "lower_right": ['right_thigh', 'right_shin', 'right_foot']
                        }
                    })

        elif "amp_humanoid_sword_shield.xml" in self._asset_file_name:
            jb_names_dict = munchify({
                        "joints": {
                            "upper_left": ['left_shoulder', 'left_elbow'],
                            "upper_right": ['right_shoulder', 'right_elbow', 'right_hand'],
                            "upper_torso": ['abdomen', 'neck'],
                            "lower_left": ['left_hip', 'left_knee', 'left_ankle'],
                            "lower_right": ['right_hip', 'right_knee', 'right_ankle']
                        },
                        "rigid_body": {
                            "upper_left": ['left_upper_arm', 'left_lower_arm','left_hand', 'shield'],
                            "upper_right": ['right_upper_arm', 'right_lower_arm', 'right_hand', 'sword'],
                            "upper_torso": ['pelvis', 'torso', 'head'],
                            "lower_left": ['left_thigh', 'left_shin', 'left_foot'],
                            "lower_right": ['right_thigh', 'right_shin', 'right_foot']
                        }
                    })

        elif "smpl_humanoid.xml" in self._asset_file_name:
            jb_names_dict = munchify({
                "joints": {
                    "upper_left": [ "L_Wrist","L_Elbow","L_Hand","L_Shoulder","L_Thorax","L_Toe",],
                    "upper_right": ["R_Hand","R_Shoulder","R_Toe","R_Wrist","R_Thorax","R_Elbow"],
                    "upper_torso": ["Torso", "Chest", "Head", "Spine", "Neck"],
                    "lower_left": ["L_Ankle", "L_Hip", "L_Knee"],
                    "lower_right": ["R_Ankle", "R_Hip", "R_Knee"],
                },
                "rigid_body": {
                    "upper_left": ["L_Toe", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
                    "upper_right": ["R_Toe", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
                    "upper_torso": ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"],
                    "lower_left": ["L_Hip", "L_Knee", "L_Ankle"],
                    "lower_right": ["R_Hip", "R_Knee", "R_Ankle"],
                },
            })
             
        else:
            assert False, "Not implemented for this humnoid skeleon / asset type file"

        # assert that when body is combined it matches this rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], 0)
        num_joints = len(self._dof_offsets) - 1
        actor_dof_names = self.gym.get_actor_dof_names(self.envs[0], 0)
        joint_names = list(set([n.replace("_x", "").replace("_y","").replace("_z","") for n in actor_dof_names]))
        rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], 0)
        assert num_joints == len(joint_names)
        assert set(rigid_body_names) == set(body for body_list in jb_names_dict.rigid_body.values() for body in body_list), "Combined rigid body names do not match"
        assert set(joint_names) == set(joint for joint_list in jb_names_dict.joints.values() for joint in joint_list), "Combined joint names do not match"

        dof_joint_names = joint_names
        dof_vel_indxs_dict = {}
        dof_pos_obs_indxs_dict = {}
        for n in dof_joint_names:
            dof_vel_indxs_dict[n] = list(range(self._dof_offsets[joint_names.index(n)],
                                                self._dof_offsets[joint_names.index(n)+1]))
            dof_pos_obs_indxs_dict[n] = joint_names.index(n)*6 + np.arange(6)

        # Masking Inedexes for dof velocities
        dof_vel_indxs_set_left_upper = [i for n in jb_names_dict.joints.upper_left 
                                            for i in range(self._dof_offsets[joint_names.index(n)],
                                                            self._dof_offsets[joint_names.index(n)+1])]
        dof_vel_indxs_set_right_upper = [i for n in jb_names_dict.joints.upper_torso + jb_names_dict.joints.upper_right
                                            for i in range(self._dof_offsets[joint_names.index(n)],
                                                            self._dof_offsets[joint_names.index(n)+1])]
        dof_vel_indxs_set_lower = [i for n in jb_names_dict.joints.lower_left + jb_names_dict.joints.lower_right
                                            for i in range(self._dof_offsets[joint_names.index(n)],
                                                            self._dof_offsets[joint_names.index(n)+1])]
        assert len(np.unique(dof_vel_indxs_set_right_upper + dof_vel_indxs_set_left_upper + dof_vel_indxs_set_lower)) == self._dof_offsets[-1]

        # Masking Inedexes for dof position observations
        dof_obs_indxs_set_left_upper = [i for n in jb_names_dict.joints.upper_left   
                                            for i in joint_names.index(n)*6 + np.arange(6)]
        dof_obs_indxs_set_right_upper = [i for n in jb_names_dict.joints.upper_torso + jb_names_dict.joints.upper_right
                                            for i in joint_names.index(n)*6 + np.arange(6)]
        dof_obs_indxs_set_lower = [i for n in jb_names_dict.joints.lower_left + jb_names_dict.joints.lower_right
                                            for i in joint_names.index(n)*6 + np.arange(6)]
        assert len(np.unique(dof_obs_indxs_set_right_upper + dof_obs_indxs_set_left_upper + dof_obs_indxs_set_lower)) == self._dof_obs_size

        rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], 0)

        # Masking Inedexes for dof xyz positions
        upper_rigid_body_indxs = [rigid_body_names.index(n) for n in jb_names_dict.rigid_body.upper_left + jb_names_dict.rigid_body.upper_right + jb_names_dict.rigid_body.upper_torso]
        lower_rigid_body_indxs = [rigid_body_names.index(n) for n in  jb_names_dict.rigid_body.lower_left +  jb_names_dict.rigid_body.lower_right]

        dof_xyz_pos_indxs_set_left_upper = [i for n in jb_names_dict.rigid_body.upper_left 
                                                for i in rigid_body_names.index(n)*3 + np.arange(3)]
        dof_xyz_pos_indxs_set_right_upper = [i for n in jb_names_dict.rigid_body.upper_right + jb_names_dict.rigid_body.upper_torso
                                                for i in rigid_body_names.index(n)*3 + np.arange(3)]
        dof_xyz_pos_indxs_set_lower = [i for n in jb_names_dict.rigid_body.lower_left +  jb_names_dict.rigid_body.lower_right
                                                for i in rigid_body_names.index(n)*3 + np.arange(3)]
        assert len(np.unique(dof_xyz_pos_indxs_set_right_upper + dof_xyz_pos_indxs_set_left_upper + dof_xyz_pos_indxs_set_lower)) == self.num_bodies*3

        key_body_names = self.agent_properties.key_body_names
        upper_body_left_key_body_names = [k for k in self.agent_properties.key_body_names if k in jb_names_dict.rigid_body.upper_left ] #['left_hand', 'shield']
        upper_body_right_key_body_names = [k for k in self.agent_properties.key_body_names if k in jb_names_dict.rigid_body.upper_right ] #['right_hand', 'sword']
        upper_body_torso_key_body_names = [k for k in self.agent_properties.key_body_names if k in jb_names_dict.rigid_body.upper_torso ] # []
        lower_body_left_key_body_names = [k for k in self.agent_properties.key_body_names if k in jb_names_dict.rigid_body.lower_left] # ['left_foot']
        lower_body_right_key_body_names = [k for k in self.agent_properties.key_body_names if k in jb_names_dict.rigid_body.lower_right] # ['right_foot']

        # This is only for amp obs local keyxyz positions , not to be used else where
        dof_key_xyz_pos_indxs_set_left_upper = [i for n in upper_body_left_key_body_names
                                                    for i in key_body_names.index(n)*3 + np.arange(3)]
        dof_key_xyz_pos_indxs_set_right_upper = [i for n in upper_body_torso_key_body_names + upper_body_right_key_body_names
                                                    for i in key_body_names.index(n)*3 + np.arange(3)]
        dof_key_xyz_pos_indxs_set_lower = [i for n in lower_body_left_key_body_names + lower_body_right_key_body_names
                                                    for i in key_body_names.index(n)*3 + np.arange(3)]
        assert len(np.unique(dof_key_xyz_pos_indxs_set_right_upper + dof_key_xyz_pos_indxs_set_left_upper + dof_key_xyz_pos_indxs_set_lower)) == len(self._key_body_ids)*3

        # ['right_hand', 'left_hand', 'right_foot', 'left_foot', 'sword', 'shield']
        self.obs_mask_utility_indexes = {
                                            "dof_joint_names": dof_joint_names,
                                            "dof_vel_indxs_dict": dof_vel_indxs_dict,
                                            "dof_pos_obs_indxs_dict": dof_pos_obs_indxs_dict,
            
                                            "dof_vel_indxs_set_left_upper": np.array(dof_vel_indxs_set_left_upper),
                                            "dof_vel_indxs_set_right_upper": np.array(dof_vel_indxs_set_right_upper),
                                            "dof_vel_indxs_set_lower": np.array(dof_vel_indxs_set_lower),
                                            "dof_vel_indxs_set_upper": np.array(dof_vel_indxs_set_left_upper + dof_vel_indxs_set_right_upper),

                                            "dof_obs_indxs_set_left_upper": np.array(dof_obs_indxs_set_left_upper),
                                            "dof_obs_indxs_set_right_upper": np.array(dof_obs_indxs_set_right_upper),
                                            "dof_obs_indxs_set_lower": np.array(dof_obs_indxs_set_lower),
                                            "dof_obs_indxs_set_upper": np.array(dof_obs_indxs_set_left_upper + dof_obs_indxs_set_right_upper),

                                            "dof_xyz_pos_indxs_set_left_upper": np.array(dof_xyz_pos_indxs_set_left_upper),
                                            "dof_xyz_pos_indxs_set_right_upper": np.array(dof_xyz_pos_indxs_set_right_upper),
                                            "dof_xyz_pos_indxs_set_lower": np.array(dof_xyz_pos_indxs_set_lower),
                                            "dof_xyz_pos_indxs_set_upper": np.array(dof_xyz_pos_indxs_set_left_upper + dof_xyz_pos_indxs_set_right_upper),
                                            
                                            "dof_key_xyz_pos_indxs_set_left_upper": np.array(dof_key_xyz_pos_indxs_set_left_upper),
                                            "dof_key_xyz_pos_indxs_set_right_upper": np.array(dof_key_xyz_pos_indxs_set_right_upper),
                                            "dof_key_xyz_pos_indxs_set_lower": np.array(dof_key_xyz_pos_indxs_set_lower),
                                            
                                        "upper_rigid_body_indxs": np.array(upper_rigid_body_indxs),
                                        "lower_rigid_body_indxs": np.array(lower_rigid_body_indxs),
                                            }

        # ------------------- HARD CODED RANDOM MOTIONS -------------------
        # root_pos *     root_rot *             root_vel *            root_local_ang_vel_obs 0    ___________.      ________.      ___________.
        # root_pos *     root_rot *             root_vel *            root_local_ang_vel_obs 0    [1] u_body_r.    [2] u_body_l.

        # ------------------- NATURAL COMPOSITION OF MOTIONS ---------------
        # [1] root_pos   [1] root_rot *         [1] root_vel *        [1] root_local_ang_vel_obs  [2] u_body_r.    [3] u_body_l.   [1] l_body

        # ------------------- NATURAL FULL LOOKHEADS ---------------------
        # root_pos.      root_rot.              root_vel.             root_local_ang_vel_obs.     dof_pos_obs.      dof_vel.       dof_xyz.

        # ------------------- NATURAL PARTIAL LOOKHEADS ---------------------
        # root_pos.      root_rot.              root_vel.             root_local_ang_vel_obs.     ___________.      ________.      ___________.
        # ___________.   __________________.   _________________.     ______________________.     _l_body____.      _l_body_.      _l_body____.
        # ___________.   __________________.   _________________.     ______________________.     _u_body_r__.      _u_body_r.     _u_body_r__.
        # ___________.   __________________.   _________________.     ______________________.     _u_body_l__.      _u_body_l.     _u_body_l__.
        # ___________.   __________________.   _________________.     ______________________.     [1] u_body_r.    [2] u_body_l     [3] l_body.
        # ___________.   __________________.   _________________.     ______________________.     [1] hand, foot, sword shield (Keypos only)

    def _init_lookahead_mask_pool(self):

        len_key_body_ids = len(self._key_body_ids)
        root_obs_offset = 0
        dof_pos_obs_offset = 13 
        dof_vel_obs_offset = 13 + self._dof_obs_size
        local_body_xyz_pos_obs_offset = 13 + self._dof_obs_size + self._dof_offsets[-1]
        glboal_body_xyz_pos_obs_offset = 13 + self._dof_obs_size + self._dof_offsets[-1] + 3 * self.num_bodies

        lower_body_pos_obs_indxs = dof_pos_obs_offset + self.obs_mask_utility_indexes["dof_obs_indxs_set_lower"]
        upper_body_right_pos_obs_indxs = dof_pos_obs_offset + self.obs_mask_utility_indexes["dof_obs_indxs_set_right_upper"]
        upper_body_left_pos_obs_indxs = dof_pos_obs_offset + self.obs_mask_utility_indexes["dof_obs_indxs_set_left_upper"]

        lower_body_vel_obs_indxs = dof_vel_obs_offset + self.obs_mask_utility_indexes["dof_vel_indxs_set_lower"]
        upper_body_right_vel_obs_indxs = dof_vel_obs_offset + self.obs_mask_utility_indexes["dof_vel_indxs_set_right_upper"]
        upper_body_left_vel_obs_indxs = dof_vel_obs_offset + self.obs_mask_utility_indexes["dof_vel_indxs_set_left_upper"]

        local_lower_body_xyz_obs_indxs = local_body_xyz_pos_obs_offset + self.obs_mask_utility_indexes["dof_xyz_pos_indxs_set_lower"]
        local_upper_body_right_xyz_obs_indxs = local_body_xyz_pos_obs_offset + self.obs_mask_utility_indexes["dof_xyz_pos_indxs_set_right_upper"]
        local_upper_body_left_xyz_obs_indxs = local_body_xyz_pos_obs_offset + self.obs_mask_utility_indexes["dof_xyz_pos_indxs_set_left_upper"]

        global_lower_body_xyz_obs_indxs = glboal_body_xyz_pos_obs_offset + self.obs_mask_utility_indexes["dof_xyz_pos_indxs_set_lower"]
        global_upper_body_right_xyz_obs_indxs = glboal_body_xyz_pos_obs_offset + self.obs_mask_utility_indexes["dof_xyz_pos_indxs_set_right_upper"]
        global_upper_body_left_xyz_obs_indxs = glboal_body_xyz_pos_obs_offset + self.obs_mask_utility_indexes["dof_xyz_pos_indxs_set_left_upper"]

        assert len(np.unique(np.concatenate((lower_body_pos_obs_indxs, upper_body_right_pos_obs_indxs, upper_body_left_pos_obs_indxs)))) == self._dof_obs_size
        assert len(np.unique(np.concatenate((lower_body_vel_obs_indxs, upper_body_right_vel_obs_indxs, upper_body_left_vel_obs_indxs)))) == self._dof_offsets[-1]
        assert len(np.unique(np.concatenate((local_lower_body_xyz_obs_indxs, local_upper_body_right_xyz_obs_indxs, local_upper_body_left_xyz_obs_indxs)))) == self.num_bodies*3
        assert len(np.unique(np.concatenate((global_lower_body_xyz_obs_indxs, global_upper_body_right_xyz_obs_indxs, global_upper_body_left_xyz_obs_indxs)))) == self.num_bodies*3

        all_obs_lk_indxs = torch.arange(self._lookahead_obs_dim).cuda()
        all_root_obs_lk_indxs = torch.arange(13).cuda()
        all_dof_pos_obs_lk_indxs = all_obs_lk_indxs[dof_pos_obs_offset:dof_vel_obs_offset]
        all_dof_vel_obs_lk_indxs = all_obs_lk_indxs[dof_vel_obs_offset:local_body_xyz_pos_obs_offset]
        all_local_dof_xyz_pos_obs_lk_indxs = all_obs_lk_indxs[local_body_xyz_pos_obs_offset:glboal_body_xyz_pos_obs_offset]
        all_global_dof_xyz_pos_obs_lk_indxs = all_obs_lk_indxs[glboal_body_xyz_pos_obs_offset:]
        all_global_key_dof_xyz_pos_obs_lk_indxs = glboal_body_xyz_pos_obs_offset + \
                                             self._key_body_ids.repeat_interleave(3)*3 + \
                                             torch.arange(3).repeat(len_key_body_ids).view(-1).cuda()
        all_local_key_dof_xyz_pos_obs_lk_indxs = local_body_xyz_pos_obs_offset + \
                                             self._key_body_ids.repeat_interleave(3)*3 + \
                                             torch.arange(3).repeat(len_key_body_ids).view(-1).cuda()

        lower_body_lk_obs_indxs = torch.LongTensor(np.concatenate((all_root_obs_lk_indxs.cpu(), lower_body_pos_obs_indxs, lower_body_vel_obs_indxs, local_lower_body_xyz_obs_indxs))).cuda()
        upper_body_right_lk_obs_indxs = torch.LongTensor(np.concatenate((upper_body_right_pos_obs_indxs, upper_body_right_vel_obs_indxs, local_upper_body_right_xyz_obs_indxs))).cuda()
        upper_body_left_lk_obs_indxs = torch.LongTensor(np.concatenate((upper_body_left_pos_obs_indxs, upper_body_left_vel_obs_indxs, local_upper_body_left_xyz_obs_indxs))).cuda()
        upper_body_lk_obs_indxs = torch.LongTensor(np.concatenate((upper_body_right_lk_obs_indxs.cpu(), upper_body_left_lk_obs_indxs.cpu()))).cuda()
        local_upper_body_left_xyz_lk_obs_indxs = torch.LongTensor(local_upper_body_left_xyz_obs_indxs).cuda()
        local_upper_body_right_xyz_lk_obs_indxs = torch.LongTensor(local_upper_body_right_xyz_obs_indxs).cuda()
        local_lower_body_xyz_lk_obs_indxs = torch.LongTensor(local_lower_body_xyz_obs_indxs).cuda()

        # Define Masks
        self.lookahead_mask_pool = torch.ones((8, self._lookahead_obs_dim)).type(torch.bool).cuda()
        self.lookahead_mask_pool[0, torch.cat((all_root_obs_lk_indxs, all_dof_pos_obs_lk_indxs,
                                                all_dof_vel_obs_lk_indxs, all_local_dof_xyz_pos_obs_lk_indxs))] = False # Mask global joint xyz positions , Root obs + joint angle + joint angle vel + local joint xyz 
        self.lookahead_mask_pool[1, torch.cat((all_root_obs_lk_indxs,))] = False # Root obs only
        self.lookahead_mask_pool[2, torch.cat((all_root_obs_lk_indxs, all_dof_pos_obs_lk_indxs, all_dof_vel_obs_lk_indxs))] = False # Root Obs and Joint Angles
        self.lookahead_mask_pool[3, torch.cat((all_global_dof_xyz_pos_obs_lk_indxs,))] = False # Global joint xyz only
        self.lookahead_mask_pool[4, torch.cat((all_global_key_dof_xyz_pos_obs_lk_indxs,))] = False # Global key joint xyz only
        self.lookahead_mask_pool[5, torch.cat((all_root_obs_lk_indxs, all_local_dof_xyz_pos_obs_lk_indxs))] = False # Root obs + local joint xyz only
        self.lookahead_mask_pool[6, torch.cat((all_root_obs_lk_indxs, all_local_key_dof_xyz_pos_obs_lk_indxs))] = False # Root obs + local joint key xyz only
        self.lookahead_mask_pool[7, torch.cat((all_root_obs_lk_indxs, local_upper_body_left_xyz_lk_obs_indxs, local_upper_body_right_xyz_lk_obs_indxs, ))] = False # Root obs + Upper Body. 
        # self.lookahead_mask_pool[7, torch.cat((all_root_obs_lk_indxs, local_lower_body_xyz_lk_obs_indxs,))] = False # Root obs + Lower Body.

        self._root_obs_conditioning_mask_pool_indx = 1
        self._compose_compatible_lookahead_mask_pool_indx = 5
        self._upper_body_conditioning_compatible_mask_pool_indx = 7

        # make upper_body_lookahead_mask
        # upper_body_lookahead_indxs = self.obs_mask_utility_indexes["all_root_obs_lk_indxs"].tolist() + \
        #                              self.obs_mask_utility_indexes["local_upper_body_right_xyz_lk_obs_indxs"].tolist() + \
        #                              self.obs_mask_utility_indexes["local_upper_body_left_xyz_lk_obs_indxs"].tolist()
        # self.upper_body_lookahead_mask = torch.ones(self._lookahead_obs_dim).type(torch.bool).cuda()
        # self.upper_body_lookahead_mask[upper_body_lookahead_indxs] = False # Do not mask upper body lookahead and root (note xyz are still masked because it has orientation information.)

        # Define Sample Weights
        self.lookahead_mask_pool_sample_weights = torch.ones(len(self.lookahead_mask_pool)).cuda()
        self.lookahead_mask_pool_sample_weights[0] = 1 # All Mocap details 
        self.lookahead_mask_pool_sample_weights[1] = int(self._enable_lk_channel_mask) # Root Obs Only 
        self.lookahead_mask_pool_sample_weights[2] = int(self._enable_lk_channel_mask) # Root Obs + Joint Angles
        self.lookahead_mask_pool_sample_weights[3] = int(self._enable_lk_channel_mask) + int(self._enable_lk_jl_mask) # Global joint xyz only
        self.lookahead_mask_pool_sample_weights[4] = int(self._enable_lk_jl_mask) # Global key joint xyz only
        self.lookahead_mask_pool_sample_weights[5] = int(self._enable_lk_channel_mask) # Root obs + local joint xyz only
        self.lookahead_mask_pool_sample_weights[6] = int(self._enable_lk_jl_mask) # Root obs + local joint key xyz only
        self.lookahead_mask_pool_sample_weights[7] = int(self._enable_lk_channel_mask)  # Root obs + Upper Body.
        # self.lookahead_mask_pool_sample_weights[4] = 2 # For Composite motions, we want to sample root + local joint xyz more often

        self.lookahead_mask_pool_compose_compatible_sample_weights = torch.zeros(len(self.lookahead_mask_pool)).cuda()
        self.lookahead_mask_pool_compose_compatible_sample_weights[1] = 1
        self.lookahead_mask_pool_compose_compatible_sample_weights[5] = 1
        self.lookahead_mask_pool_compose_compatible_sample_weights[7] = 1

        self._lookahead_factor_dims = [len(all_root_obs_lk_indxs), 
                                       len(all_dof_pos_obs_lk_indxs),
                                       len(all_dof_vel_obs_lk_indxs),
                                       len(all_local_dof_xyz_pos_obs_lk_indxs), 
                                       len(all_global_dof_xyz_pos_obs_lk_indxs)]
        assert len(self._lookahead_factor_dims) == self.num_lookahead_channels

        # update mask utility indexes
        self.obs_mask_utility_indexes.update({"all_obs_lk_indxs": torch.cat((all_obs_lk_indxs,)).cpu().numpy(),
                                              "all_root_obs_lk_indxs": torch.cat((all_root_obs_lk_indxs,)).cpu().numpy(),
                                              "lower_body_lk_obs_indxs": lower_body_lk_obs_indxs.cpu().numpy(),
                                              "upper_body_right_lk_obs_indxs": upper_body_right_lk_obs_indxs.cpu().numpy(),
                                              "upper_body_left_lk_obs_indxs": upper_body_left_lk_obs_indxs.cpu().numpy(),
                                              "upper_body_lk_obs_indxs": upper_body_lk_obs_indxs.cpu().numpy(),
                                              "local_upper_body_right_xyz_lk_obs_indxs": local_upper_body_right_xyz_obs_indxs,
                                              "local_upper_body_left_xyz_lk_obs_indxs": local_upper_body_left_xyz_obs_indxs,
                                              "all_dof_pos_obs_lk_indxs": all_dof_pos_obs_lk_indxs.cpu().numpy(),
                                              "all_global_dof_xyz_pos_obs_lk_indxs": all_global_dof_xyz_pos_obs_lk_indxs.cpu().numpy(),
                                              "all_local_key_dof_xyz_pos_obs_lk_indxs": all_local_key_dof_xyz_pos_obs_lk_indxs.cpu().numpy()})

        # ------------------- HARD CODED RANDOM MOTIONS -------------------
        # root_pos *     root_rot *             root_vel *            root_local_ang_vel_obs 0    ___________.      ________.      ___________.
        # root_pos *     root_rot *             root_vel *            root_local_ang_vel_obs 0    [1] u_body_r.    [2] u_body_l.

        # ------------------- NATURAL COMPOSITION OF MOTIONS ---------------
        # [1] root_pos   [1] root_rot *         [1] root_vel *        [1] root_local_ang_vel_obs  [2] u_body_r.    [3] u_body_l.   [1] l_body

        # ------------------- NATURAL FULL LOOKHEADS ---------------------
        # root_pos.      root_rot.              root_vel.             root_local_ang_vel_obs.     dof_pos_obs.      dof_vel.       dof_xyz.

        # ------------------- NATURAL PARTIAL LOOKHEADS ---------------------
        # root_pos.      root_rot.              root_vel.             root_local_ang_vel_obs.     ___________.      ________.      ___________.
        # ___________.   __________________.   _________________.     ______________________.     _l_body____.      _l_body_.      _l_body____.
        # ___________.   __________________.   _________________.     ______________________.     _u_body_r__.      _u_body_r.     _u_body_r__.
        # ___________.   __________________.   _________________.     ______________________.     _u_body_l__.      _u_body_l.     _u_body_l__.
        # ___________.   __________________.   _________________.     ______________________.     [1] u_body_r.    [2] u_body_l     [3] l_body.
        # ___________.   __________________.   _________________.     ______________________.     [1] hand, foot, sword shield (Keypos only)

    def _init_amp_obs_mask_pool(self):
        def minus_operator(t1,t2):
            # Create a tensor to compare all values at once
            compareview = t2.repeat(t1.shape[0],1).T
            # Non Intersection
            return t1[(compareview != t1).T.prod(1)==1]

        root_obs_offset = 0
        dof_pos_obs_offset = 13 
        dof_vel_obs_offset = 13 + self._dof_obs_size
        keypos_obs_offset = 13 + self._dof_obs_size + self._dof_offsets[-1]
        len_key_body_ids = len(self._key_body_ids)

        lower_body_pos_obs_indxs = dof_pos_obs_offset + self.obs_mask_utility_indexes["dof_obs_indxs_set_lower"]
        upper_body_right_pos_obs_indxs = dof_pos_obs_offset + self.obs_mask_utility_indexes["dof_obs_indxs_set_right_upper"]
        upper_body_left_pos_obs_indxs = dof_pos_obs_offset + self.obs_mask_utility_indexes["dof_obs_indxs_set_left_upper"]

        lower_body_vel_obs_indxs = dof_vel_obs_offset + self.obs_mask_utility_indexes["dof_vel_indxs_set_lower"]
        upper_body_right_vel_obs_indxs = dof_vel_obs_offset + self.obs_mask_utility_indexes["dof_vel_indxs_set_right_upper"]
        upper_body_left_vel_obs_indxs = dof_vel_obs_offset + self.obs_mask_utility_indexes["dof_vel_indxs_set_left_upper"]

        lower_body_key_xyz_obs_indxs = keypos_obs_offset + self.obs_mask_utility_indexes["dof_key_xyz_pos_indxs_set_lower"]
        upper_body_right_key_xyz_obs_indxs = keypos_obs_offset + self.obs_mask_utility_indexes["dof_key_xyz_pos_indxs_set_right_upper"]
        upper_body_left_key_xyz_obs_indxs = keypos_obs_offset + self.obs_mask_utility_indexes["dof_key_xyz_pos_indxs_set_left_upper"]

        assert len(np.unique(np.concatenate((lower_body_pos_obs_indxs, upper_body_right_pos_obs_indxs, upper_body_left_pos_obs_indxs)))) == self._dof_obs_size
        assert len(np.unique(np.concatenate((lower_body_vel_obs_indxs, upper_body_right_vel_obs_indxs, upper_body_left_vel_obs_indxs)))) == self._dof_offsets[-1]
        assert len(np.unique(np.concatenate((lower_body_key_xyz_obs_indxs, upper_body_right_key_xyz_obs_indxs, upper_body_left_key_xyz_obs_indxs)))) == len_key_body_ids*3

        all_obs_indxs = torch.arange(self._amp_obs_dim).cuda()
        root_amp_obs_indxs = torch.arange(13).cuda()
        lower_body_amp_obs_indxs = torch.LongTensor(np.concatenate((root_amp_obs_indxs.cpu(), lower_body_pos_obs_indxs, lower_body_vel_obs_indxs, lower_body_key_xyz_obs_indxs))).cuda()
        upper_body_right_amp_obs_indxs = torch.LongTensor(np.concatenate((upper_body_right_pos_obs_indxs, upper_body_right_vel_obs_indxs, upper_body_right_key_xyz_obs_indxs))).cuda()
        upper_body_left_amp_obs_indxs = torch.LongTensor(np.concatenate((upper_body_left_pos_obs_indxs, upper_body_left_vel_obs_indxs, upper_body_left_key_xyz_obs_indxs))).cuda()

        assert len(torch.unique(torch.cat((root_amp_obs_indxs, lower_body_amp_obs_indxs, upper_body_right_amp_obs_indxs, upper_body_left_amp_obs_indxs)))) == self._amp_obs_dim

        self.amp_obs_mask_pool = torch.ones((5, self._amp_obs_dim)).type(torch.bool).cuda()
        self.amp_obs_mask_pool[0, torch.cat((all_obs_indxs,))] = False # Root obs + joint angle + joint xyz 
        self.amp_obs_mask_pool[1, torch.cat((root_amp_obs_indxs,))] = False # Root obs 
        self.amp_obs_mask_pool[2, torch.cat((lower_body_amp_obs_indxs,))] = False # lower body amp obs only
        self.amp_obs_mask_pool[3, torch.cat((upper_body_right_amp_obs_indxs,))] = False # upper right
        self.amp_obs_mask_pool[4, torch.cat((upper_body_left_amp_obs_indxs,))] = False # upper left

        self.amp_obs_mask_pool_sample_weights = torch.ones(len(self.amp_obs_mask_pool)).cuda()

        self.obs_mask_utility_indexes.update({"amp_obs_indxs": torch.cat((all_obs_indxs,)).cpu().numpy(),
                                              "root_amp_obs_indxs": torch.cat((root_amp_obs_indxs,)).cpu().numpy(),
                                              "lower_body_amp_obs_indxs": lower_body_amp_obs_indxs.cpu().numpy(),
                                              "upper_body_right_amp_obs_indxs": upper_body_right_amp_obs_indxs.cpu().numpy(),
                                              "upper_body_left_amp_obs_indxs": upper_body_left_amp_obs_indxs.cpu().numpy(),})

        # All Obs             | root_height.   root_local_rot_obs.   root_local_vel_obs.   root_local_ang_vel_obs.   dof_pos_obs.    dof_vel.   dof_local_key_xyz.
        # Root Obs            |root_height.   root_local_rot_obs.   root_local_vel_obs.   root_local_ang_vel_obs.   ___________.    ________.   _________________.
        # lower body          |___________.   __________________.   _________________.    ______________________.    _l_body___.    _l_body_.   _l_body__________.
        # upper body right    |___________.   __________________.   _________________.    ______________________.   _u_body_r__.    _u_body_r.  _u_body_r________.
        # upper body left     |___________.   __________________.   _________________.    ______________________.   _u_body_l__.    _u_body_l.  _u_body_l________.

    def _sample_lookahead_mask_pool_indxs_for_composed_motions(self, n):
        if self._enable_lookahead_mask and n > 0:
            return torch.multinomial(self.lookahead_mask_pool_compose_compatible_sample_weights, num_samples=n, replacement=True)
        else:
            return torch.ones(n).type(torch.LongTensor).cuda() * self._compose_compatible_lookahead_mask_pool_indx

    def _sample_lookahead_mask_pool_indxs(self, n):
        if self._enable_lookahead_mask:
            return torch.multinomial(self.lookahead_mask_pool_sample_weights, num_samples=n, replacement=True)
        else:
            return torch.zeros(n).type(torch.LongTensor).cuda()

    def _sample_amp_obs_mask_pool_indxs(self, n):
        # if self._enable_lookahead_mask:
        return torch.multinomial(self.amp_obs_mask_pool_sample_weights, num_samples=n, replacement=True)
        # else:
        #     return torch.zeros(n).type(torch.LongTensor).cuda()

    # def _build_pd_action_offset_scale(self):
    #     super()._build_pd_action_offset_scale()
    #     return

    def _build_pd_action_offset_scale(self):
        super()._build_pd_action_offset_scale()

        if "digit" in self._motion_file:
            self._pd_action_offset = self._pd_action_offset + torch.FloatTensor([0.2554, -0.0362, 0.0785, # Left leg
                                                                                -0.0192, 0.0722, -0.0174,
                                                                                -0.0880, -0.1098,
                                                                                -0.2057, 0.9051, 0.0004, 0.6, # Left arm
                                                                                -0.2535, 0.0308, -0.0867, # Right leg
                                                                                0.0220, -0.0714, 0.0162,
                                                                                0.0921, 0.1256,
                                                                                0.2057, -0.9051, 0.0, -0.6, # Right arm
                                                                                ]).to(self.device)
            print("PD Offset changed for digit")

        # self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        return

    # RobustAdditionalCodeBlock
    # -------------------------------------------------------
    @property
    def lookahead_tar_flat_obs(self):
        return self._tar_flat_lookahead_obs 

    @property
    def lookahead_tar_flat_dim(self):
        return self._tar_flat_lookahead_obs.size(-1)

    def localize_lookahead_obs(self, lookahead_obs_flat_batch, lookahead_obs_mask,  num_stacks):
        assert num_stacks == self._lookahead_timesteps, "This logic is not defined yet."

        if len(lookahead_obs_flat_batch.shape) == 2:
            pass
        else:
            assert False, "This logic is not defined yet."

        # num_stacks = 10
        bs = lookahead_obs_flat_batch.size(0)

        lookahead_root_xyz_obs = lookahead_obs_flat_batch.view(bs, num_stacks, -1)[:,:,0:3].clone()
        lookahead_root_xyz_obs[:,:,:2] = 0 # Mask out the x and y dimension of the root observation

        lookahead_root_pose_obs = lookahead_obs_flat_batch.view(bs, num_stacks, -1)[:,:,3:13]

        lookahead_localized_obs = lookahead_obs_flat_batch.view(bs, num_stacks, -1)[:,:,13:-self.num_bodies*3]

        lookahead_glboal_pos = lookahead_obs_flat_batch.view(bs, num_stacks, -1)[:,:,-self.num_bodies*3:]
        lookahead_glboal_pos_unrolled = lookahead_glboal_pos.view(bs, num_stacks, -1, 3)
        lookahead_root_xyz = lookahead_glboal_pos_unrolled[:,0:1,0:1,:].clone()#. (batch,1,1, 3)
        localized_lookahead_unrolled = lookahead_glboal_pos_unrolled - lookahead_root_xyz
        localized_lookahead_unrolled[:,0,0,2] = lookahead_root_xyz[:,0,0,2] # give the z dimension back to the root

        lookahead_unrolled = torch.cat((lookahead_root_xyz_obs.view(bs,num_stacks,3),
                                        lookahead_root_pose_obs.view(bs, num_stacks, 10),
                                        lookahead_localized_obs.view(bs,num_stacks,-1), 
                                        localized_lookahead_unrolled.view(bs, num_stacks, -1)), dim = -1)

        # Apply Lookahead obs mask
        lookahead_unrolled[lookahead_obs_mask.unsqueeze(-2).repeat(1,num_stacks,1)] = 0
        return lookahead_unrolled.view(bs, -1)

    def _sample_demo_motions(self, n):
        if self._motion_is_sampled_by_frames:
            motion_frames = torch.multinomial(self._demo_motion_frames_weights, num_samples=n, replacement=True)
            motion_ids = self.frame_to_motion_id[motion_frames]         # motion_times = self.frame_to_motion_times[motion_frames]
        else:
            motion_ids = torch.multinomial(self._demo_motion_weights, num_samples=n, replacement=True)

        return motion_ids

    def _sample_start_state_motions(self, n):
        if self._motion_is_sampled_by_frames:
            motion_frames = torch.multinomial(self._start_state_motion_frames_weights, num_samples=n, replacement=True)
            motion_ids = self.frame_to_motion_id[motion_frames]         # motion_times = self.frame_to_motion_times[motion_frames]
        else:
            motion_ids = torch.multinomial(self._start_state_motion_weights, num_samples=n, replacement=True)

        return motion_ids
    # -------------------------------------------------------

    def _override_full_state_with_predefined_root_pose(self, full_state_obs, random_root_pose):
        # Note that everything else except the pre defined root pose will be set to zero here.
        # Because other will not be relevant and will need to be masked out in lookahead anyways.
        # This is done to avoid any confusion in the model.
        bs1 = full_state_obs.size(0)
        bs2 = random_root_pose.size(0)
        assert bs1 == bs2, "Batch size of full state obs and random root pose should be same."

        full_state_obs[:, :13] = random_root_pose[:, 0:13]
        full_state_obs[:, 13:] = 0

        return full_state_obs

    def _init_tar_amp_keypos_obs(self, env_ids):
        """
        Computes the full flat lookahead of keypos. Note that this need not be just called at the beginning of the episode, but can be called at any time step.
        """
        demo_idxs = self._global_demo_start_motion_ids[env_ids]
        time_step_idxs = self.global_demo_start_motion_time_steps[env_ids]

        len_env_ids = len(env_ids)
        num_lookahead = 10

        q_ts_idxs = (time_step_idxs + self.progress_buf[env_ids] + 1) % self._motion_max_steps[demo_idxs]  # num_envs X 1

        f_idxs = self._motion_lib.demo_store["length_starts"][demo_idxs[:, None]] + q_ts_idxs[:, None] + torch.arange(num_lookahead).cuda()  # num_envs X num_lookahead

        tar_full_states = self._motion_lib.demo_store["full_state_obs"].transpose(0, 1)[f_idxs.view(-1)].cuda()

        # Rotate target as specified by the demo rotations
        rotation_quat = self._global_demo_start_rotations[env_ids].repeat_interleave(10, dim = 0) # Todo Change the hardcoding here
        rotated_tar_full_states = self._rotate_full_state_obs(rotate_demo_by=rotation_quat,
                                                              full_state_obs=tar_full_states)
        self._tar_flat_full_state_obs[env_ids] = rotated_tar_full_states.view(len_env_ids, -1)
        self._tar_flat_amp_obs[env_ids] = rotated_tar_full_states[:, self._full_state_amp_obs_indxs].view(len_env_ids, -1)
        # self._tar_flat_lookahead_obs[env_ids] = rotated_tar_full_states[:, self._full_state_lookahead_obs_indxs].view(len_env_ids, -1)

        # OverRide full state for dedicated env ids
        # full state obs = torch.cat((root_pos, root_rot, root_vel, root_ang_vel, root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_pos, dof_obs, dof_vel, flat_local_key_pos, non_localized_body_pos), dim=-1)
        for i in range(num_lookahead):
            self._tar_flat_full_state_obs[self._random_dir_speed_env_idxs, i*self._full_state_obs_dim:i*self._full_state_obs_dim+10] = self._global_random_dir_speed_lookahead_bucket[self._random_dir_speed_env_idxs, :10]
            self._tar_flat_full_state_obs[self._random_dir_speed_env_idxs, i*self._full_state_obs_dim+23 : i*self._full_state_obs_dim+26] = self._global_random_dir_speed_lookahead_bucket[self._random_dir_speed_env_idxs, 10:13]
            self._tar_flat_full_state_obs[self._random_dir_speed_env_idxs, (i+1)*self._full_state_obs_dim-self.num_bodies*3:(i+1)*self._full_state_obs_dim] = 0 # zero out everything global xyz positions

        compose_reset_env_idxs = torch.LongTensor(np.intersect1d(self._compose_demo_targets_env_idxs.cpu().numpy(), env_ids.cpu().numpy())).cuda()
        if len(compose_reset_env_idxs) > 0:
            for i in range(num_lookahead):
                t_indxs = compose_reset_env_idxs
                s_indxs = (compose_reset_env_idxs + 1)%self.num_envs
                source_full_state = self._tar_flat_full_state_obs[s_indxs, i*self._full_state_obs_dim:(i+1)*self._full_state_obs_dim]
                target_full_state = self._tar_flat_full_state_obs[t_indxs, i*self._full_state_obs_dim:(i+1)*self._full_state_obs_dim]
                self._tar_flat_full_state_obs[compose_reset_env_idxs, i*self._full_state_obs_dim: (i+1)*self._full_state_obs_dim] = \
                                                            self._compose_full_state_obs(source_full_state_obs = source_full_state,
                                                                                        target_full_state_obs = target_full_state,
                                                                                        to_compose_split = "upper_body")

        # Calculate Lookahead obs
        self._tar_flat_lookahead_obs[env_ids] = self._tar_flat_full_state_obs[env_ids].view(-1, self._full_state_obs_dim)[:, self._full_state_lookahead_obs_indxs].view(len_env_ids, -1)

        # Apply Lookahead masks for partial guidance.
        # for i in range(num_lookahead):
        #     tmp_tensor = self._tar_flat_lookahead_obs[env_ids, i*self._lookahead_obs_dim:(i+1)*self._lookahead_obs_dim].clone()
        #     tmp_tensor[self._global_demo_lookahead_mask[env_ids]] = 0
        #     self._tar_flat_lookahead_obs[env_ids, i*self._lookahead_obs_dim:(i+1)*self._lookahead_obs_dim] = tmp_tensor

    def _compose_full_state_obs(self, source_full_state_obs, target_full_state_obs, to_compose_split = "upper_body"):
        _target_full_state_obs = target_full_state_obs.clone()
        # obs = torch.cat((root_pos, root_rot, root_vel, root_ang_vel, root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_pos, dof_obs, dof_vel, flat_local_key_pos, non_localized_body_pos), dim=-1)
        if to_compose_split == "upper_body":
            # zero out the non localized body xyz positions
            _target_full_state_obs[:, -self.num_bodies*3:] = 0 
            # zero out the localized body joints # Including angular velocity
            _target_full_state_obs[:, 26:-self.num_bodies*3*2] = 0 

            local_xyz_indices = self._full_state_split_indxs["dof_local_xyz"].clone()[self.obs_mask_utility_indexes["dof_xyz_pos_indxs_set_upper"]]
            _target_full_state_obs[:,local_xyz_indices] = source_full_state_obs[:,local_xyz_indices]

            dof_obs_indxs = self._full_state_split_indxs["dof_pos_obs"].clone()[self.obs_mask_utility_indexes["dof_obs_indxs_set_upper"]]
            _target_full_state_obs[:,dof_obs_indxs] = source_full_state_obs[:,dof_obs_indxs]

            dof_vel_indxs = self._full_state_split_indxs["dof_vel"].clone()[self.obs_mask_utility_indexes["dof_vel_indxs_set_upper"]]
            _target_full_state_obs[:,dof_vel_indxs] = source_full_state_obs[:,dof_vel_indxs]

        else:
            assert False, "This logic is not defined yet."
        return _target_full_state_obs

    def _compute_curr_tar_amp_keypos_obs(self):

        demo_idxs = self._global_demo_start_motion_ids
        time_step_idxs = self.global_demo_start_motion_time_steps

        len_env_ids = len(demo_idxs)
        num_lookahead = 10

        q_ts_idxs = (time_step_idxs + self.progress_buf + num_lookahead + 1) % (self._motion_max_steps[demo_idxs]) # num_envs X 1

        f_idxs = self._motion_lib.demo_store["length_starts"][demo_idxs] + q_ts_idxs.cuda()  # num_envs X num_lookahead

        # Get Full State Observation
        to_set_tar_full_state_obs = self._motion_lib.demo_store["full_state_obs"].transpose(0, 1)[f_idxs.view(-1)].view(len_env_ids, -1).cuda()

        # Rotate target as specified by the demo rotations
        rotated_to_set_tar_full_state_obs = self._rotate_full_state_obs(rotate_demo_by = self._global_demo_start_rotations, full_state_obs = to_set_tar_full_state_obs)# ToDo Fix This

        # Set Values
        self._tar_flat_full_state_obs[:,-self._full_state_obs_dim:] = rotated_to_set_tar_full_state_obs[:, :]
        self._tar_flat_amp_obs[:,-self._amp_obs_dim:] = rotated_to_set_tar_full_state_obs[:, self._full_state_amp_obs_indxs]

        # Over Ride Full state for dedicated environments
        self._tar_flat_full_state_obs[self._random_dir_speed_env_idxs, -self._full_state_obs_dim:-self._full_state_obs_dim+10] = self._global_random_dir_speed_lookahead_bucket[self._random_dir_speed_env_idxs, :10]
        self._tar_flat_full_state_obs[self._random_dir_speed_env_idxs, -self._full_state_obs_dim+23:-self._full_state_obs_dim + 26] = self._global_random_dir_speed_lookahead_bucket[self._random_dir_speed_env_idxs, 10:13] # Match local angular velocity
        self._tar_flat_full_state_obs[self._random_dir_speed_env_idxs, -self.num_bodies*3 :] = 0 # global xyz positions do not make sense anymore 
        # Note localized linear velocity and orientation is not VALID. make sure the lookahead mask takes care of this.
        # Make sure that lookahead masks take care of this so that it is not input to the policy.
        # The reward is calculated based on root pose, local angular velocity and local xyz which are all honored here.
        # assert torch.sum(self._global_demo_lookahead_mask[self._random_dir_speed_env_idxs, -self.num_bodies*3:]) == 0

        # Compose the target upper body with the source upper body
        if len(self._compose_demo_targets_env_idxs) > 0:
            t_indxs = self._compose_demo_targets_env_idxs
            s_indxs = (self._compose_demo_targets_env_idxs + 1)%self.num_envs
            source = self._tar_flat_full_state_obs[s_indxs, -self._full_state_obs_dim:]
            target = self._tar_flat_full_state_obs[t_indxs, -self._full_state_obs_dim:]
            self._tar_flat_full_state_obs[self._compose_demo_targets_env_idxs, -self._full_state_obs_dim:] = \
                                            self._compose_full_state_obs(source_full_state_obs = source,
                                                                        target_full_state_obs = target,
                                                                        to_compose_split = "upper_body")

        self._tar_flat_lookahead_obs[:, -self._lookahead_obs_dim:] = self._tar_flat_full_state_obs[:, -self._full_state_obs_dim+self._full_state_lookahead_obs_indxs]

        # Apply lookahead Mask for partial guidance
        # flat_to_mask_indxs = self._global_demo_lookahead_mask.nonzero()
        # flat_to_mask_indxs_row = flat_to_mask_indxs[:,0]
        # flat_to_mask_indxs_col = flat_to_mask_indxs[:,1] + self._lookahead_obs_dim*(num_lookahead-1)
        # self._tar_flat_lookahead_obs[flat_to_mask_indxs_row,flat_to_mask_indxs_col] = 0

        # The demos will be looped back to the start of the motion
        looped_env_idxs = torch.where(q_ts_idxs <= 11)[0]
        if looped_env_idxs.size(0) > 0:
            self._init_tar_amp_keypos_obs(looped_env_idxs)

    def _shift_tar_amp_keypos_obs(self, env_ids=None):
        # Shift the target padding #############################################
        self._tar_flat_amp_obs = torch.roll(self._tar_flat_amp_obs, -self._amp_obs_dim)
        self._tar_flat_lookahead_obs = torch.roll(self._tar_flat_lookahead_obs, -self._lookahead_obs_dim)
        self._tar_flat_full_state_obs = torch.roll(self._tar_flat_full_state_obs, -self._full_state_obs_dim)
        return

    def _rotate_keypos_obs(self, rotate_demo_by, keypos_obs):
        # Rotate the agent rotation to the target rotation padding #############
        keypos_obs_dim = keypos_obs.shape[-1]
        num_bodies = keypos_obs_dim//3
        rotate_by = rotate_demo_by.repeat_interleave(num_bodies, dim = 0) #Todo Change the hardcoding here
        keypos_obs = quat_rotate(rotate_by, keypos_obs.reshape(len(keypos_obs)*num_bodies,3)).view(-1, num_bodies*3)
        return keypos_obs

    def _rotate_root_obs(self, rotate_demo_by, amp_obs):
        # ASSUMES THAT first 13 elements are root pos and rot
        tar_root_pos = amp_obs[:, 0:3]
        tar_root_rot = amp_obs[:, 3:7]
        tar_root_vel = amp_obs[:, 7:10]
        tar_root_ang_vel = amp_obs[:, 10:13]

        rotate_demo_by = self._global_demo_start_rotations.to(tar_root_rot.device)
        new_tar_root_pos = quat_rotate(rotate_demo_by, tar_root_pos) # Added for Imitation
        new_tar_root_rot = quat_mul(rotate_demo_by, tar_root_rot) # Added for Imitation
        new_tar_root_vel = quat_rotate(rotate_demo_by, tar_root_vel) # Added for Imitation
        new_tar_root_ang_vel = quat_rotate(rotate_demo_by, tar_root_ang_vel) # Added for Imitation

        amp_obs[:, 0:3] = new_tar_root_pos # Added for Imitation
        amp_obs[:, 3:7] = new_tar_root_rot # Added for Imitation
        amp_obs[:, 7:10] = new_tar_root_vel # Added for Imitation
        amp_obs[:, 10:13] = new_tar_root_ang_vel # Added for Imitation

        return amp_obs

    def _rotate_amp_obs(self, rotate_demo_by, amp_obs):
        # Rotate the agent rotation to the target rotation padding #############
        # No need to rotate amp obs now, everything is already in local co ordinate frame.
        # Height, local orientation, local angular velocity, local velocity
        # tar_root_pos = amp_obs[:, 0:3]
        # tar_root_rot = amp_obs[:, 3:7]
        # tar_root_vel = amp_obs[:, 7:10]
        # tar_root_ang_vel = amp_obs[:, 10:13]

        # rotate_demo_by = self._global_demo_start_rotations.to(tar_root_rot.device)
        # new_tar_root_pos = quat_rotate(rotate_demo_by, tar_root_pos) # Added for Imitation
        # new_tar_root_rot = quat_mul(rotate_demo_by, tar_root_rot) # Added for Imitation
        # new_tar_root_vel = quat_rotate(rotate_demo_by, tar_root_vel) # Added for Imitation
        # new_tar_root_ang_vel = quat_rotate(rotate_demo_by, tar_root_ang_vel) # Added for Imitation

        # amp_obs[:, 0:3] = new_tar_root_pos # Added for Imitation
        # amp_obs[:, 3:7] = new_tar_root_rot # Added for Imitation
        # amp_obs[:, 7:10] = new_tar_root_vel # Added for Imitation
        # amp_obs[:, 10:13] = new_tar_root_ang_vel # Added for Imitation

        return amp_obs
        ########################################################################################################

    def _rotate_full_state_obs(self, rotate_demo_by, full_state_obs):
        split_state_indxs = ObsBuilder.get_full_body_state_index_dict(self._dof_obs_size, self._dof_offsets[-1], self.num_bodies, self._key_body_ids)

        full_state_obs[:,split_state_indxs["root_pos"]] = quat_rotate(rotate_demo_by, full_state_obs[:,split_state_indxs["root_pos"]])
        full_state_obs[:,split_state_indxs["root_rot"]] = quat_mul(rotate_demo_by, full_state_obs[:,split_state_indxs["root_rot"]])
        full_state_obs[:,split_state_indxs["root_vel"]] = quat_rotate(rotate_demo_by, full_state_obs[:,split_state_indxs["root_vel"]])
        full_state_obs[:,split_state_indxs["root_ang_vel"]] = quat_rotate(rotate_demo_by, full_state_obs[:,split_state_indxs["root_ang_vel"]])
        full_state_obs[:,split_state_indxs["dof_xyz"]] = self._rotate_keypos_obs(rotate_demo_by, full_state_obs[:,split_state_indxs["dof_xyz"]])

        return full_state_obs

    def _reset_demo_tar_reset_steps(self, env_ids):
        if self._uniform_targets:
            self._demo_tar_reset_steps[env_ids] = torch.randint_like(self._demo_tar_reset_steps[0], 
                                                        low=self._demo_tar_reset_steps_min, 
                                                        high=self._demo_tar_reset_steps_max).repeat(len(env_ids))
        else:    
            self._demo_tar_reset_steps[env_ids] = torch.randint_like(self._demo_tar_reset_steps[env_ids], 
                                                            low=self._demo_tar_reset_steps_min, 
                                                            high=self._demo_tar_reset_steps_max)
        return

    def _update_demo_targets(self):
        if self._switch_demos_within_episode:
            need_switch_envs = self._demo_tar_reset_steps <= self.progress_buf

            need_update = torch.any(need_switch_envs)
            if (need_update):
                switch_demo_at_env_ids = need_switch_envs.nonzero(as_tuple=False).flatten()
                self._reset_demo_targets(switch_demo_at_env_ids)
                # self._reset_demo_lookahead_masks(switch_demo_at_env_ids)
                self._init_tar_amp_keypos_obs(switch_demo_at_env_ids)

                if self._uniform_targets:
                    self._demo_tar_reset_steps[switch_demo_at_env_ids] += torch.randint_like(self._demo_tar_reset_steps[0],
                                                                                low=self._demo_tar_reset_steps_min,
                                                                                high=self._demo_tar_reset_steps_max).repeat(len(switch_demo_at_env_ids))
                else:
                    self._demo_tar_reset_steps[switch_demo_at_env_ids] += torch.randint_like(self._demo_tar_reset_steps[switch_demo_at_env_ids],
                                                                                    low=self._demo_tar_reset_steps_min, 
                                                                                    high=self._demo_tar_reset_steps_max)

                if (self.viewer):
                    self._change_char_color(switch_demo_at_env_ids)
        return

    def _reset_demo_targets(self, env_ids, hard_start_demo_at_agent_sart=False):
        if len(env_ids) == 0 :
            return

        num_reset_envs = env_ids.shape[0]
        motion_ids = self._sample_demo_motions(num_reset_envs)
        if (self._demo_init == HumanoidAMPRobust.StateInit.Random or self._demo_init == HumanoidAMPRobust.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids).to(self.device)
        elif (self._demo_init == HumanoidAMPRobust.StateInit.Start):
            motion_times = torch.zeros(num_reset_envs, device=self.device)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        motion_rot = self._sample_random_rotation_quat_for_demo(num_reset_envs)

        if  self._uniform_targets:
            motion_ids = motion_ids[0:1].repeat(num_reset_envs)
            motion_times = motion_times[0:1].to(self.device).repeat(num_reset_envs)
            motion_rot = motion_rot[0:1].to(self.device).repeat(num_reset_envs,1)

        if self._use_predefined_targets:
            motion_ids = self._predefined_target_demo_start_motion_ids[env_ids]
            motion_times = self._predefined_target_demo_start_motion_times[env_ids]
            motion_rot = self._predefined_target_demo_start_rotations[env_ids]

        # motion_rot = torch.nn.functional.normalize(torch.cat((torch.zeros(len(env_ids), 2), torch.rand(len(env_ids), 2)), dim = -1), dim=1).to(self.device)
        self._global_demo_start_motion_ids[env_ids] = motion_ids
        self._global_demo_start_motion_times[env_ids] = motion_times
        self._global_demo_start_rotations[env_ids] = motion_rot

        # Demo Start = Agent Start
        if hard_start_demo_at_agent_sart:
            start_demo_at_agent_start_mask = torch.ones(self.num_envs).cuda()
        else:
            start_demo_at_agent_start_mask = torch.rand(self.num_envs).cuda() < self._start_demo_at_agent_state_prob

        reset_called_at_mask = torch.zeros(self.num_envs).type(torch.LongTensor).cuda()
        reset_called_at_mask[env_ids] = 1

        reset_demo_at_mask = (start_demo_at_agent_start_mask * reset_called_at_mask).type(torch.bool)
        self._global_demo_start_motion_ids[reset_demo_at_mask] = self._global_agent_start_motion_ids[reset_demo_at_mask]
        self._global_demo_start_motion_times[reset_demo_at_mask] = self._global_agent_start_motion_times[reset_demo_at_mask] + self.dt
        self._global_demo_start_rotations[reset_demo_at_mask] = self._global_agent_start_rotations[reset_demo_at_mask]

        # Reset _random_dir_speed Bucket
        self._global_random_dir_speed_lookahead_bucket[env_ids] = self._sample_random_lookahead_diff_orient_speed(num_reset_envs).cuda() #RobustAdditionalCodeLine

        if  self._start_demo_at_agent_state_prob > 0.0:
            # To be activated only if agent start prob is different from demo sampling.
            # this over rides the motion id if the agent was inititated in a trajectory that is not allowed for imitation
            allowed_idxs_for_demo = torch.nonzero(self._demo_motion_weights)
            violation_idxs = torch.nonzero(~torch.isin(self._global_demo_start_motion_ids, allowed_idxs_for_demo)).view(-1) 
            if len(violation_idxs) > 0:
                self._global_demo_start_motion_ids[violation_idxs] = self._sample_demo_motions(len(violation_idxs))

        return 

    def _reset_demo_lookahead_masks(self, env_ids):
        num_reset_envs = env_ids.shape[0]

        # Update mappings for reset env indexes
        if self._fixed_demo_lookahead_mask_env2idxmap_flag:
            self._global_demo_lookahead_mask_env2idxmap[env_ids] = ((env_ids/2)%len(self.lookahead_mask_pool)).type(torch.LongTensor).cuda()
        else:
            self._global_demo_lookahead_mask_env2idxmap[env_ids] = self._sample_lookahead_mask_pool_indxs(len(env_ids))

        ############################################################################################################################################
        # demo_lookahead_mask_env2idxmap Exception 1: _random_dir_speed_env_idxs must be set to _root_obs_conditioning_mask_pool_indx
        # Reset lookahead masks for _random_dir_speed | Mask everything but the root pose
        rand_dir_speed_reset_env_idxs = torch.LongTensor(np.intersect1d(self._random_dir_speed_env_idxs.cpu().numpy(), env_ids.cpu().numpy())).cuda()
        self._global_demo_lookahead_mask_env2idxmap[rand_dir_speed_reset_env_idxs] = self._root_obs_conditioning_mask_pool_indx # second element as the root pose only lookahead mask
        ############################################################################################################################################

        ############################################################################################################################################
        # Exception 2: _random_dir_speed_upper_body_env_idxs must be set to _upper_body_conditioning_compatible_mask_pool_indx
        rand_dir_speed_upper_body_reset_env_idxs = torch.LongTensor(np.intersect1d(self._random_dir_speed_upper_body_env_idxs.cpu().numpy(), env_ids.cpu().numpy())).cuda()
        self._global_demo_lookahead_mask_env2idxmap[rand_dir_speed_upper_body_reset_env_idxs] = self._upper_body_conditioning_compatible_mask_pool_indx
        ############################################################################################################################################

        ############################################################################################################################################
        # Exception 3: _compose_demo_targets_env_idxs must be set to sampled _root_obs_conditioning_mask_pool_indx
        # Reset lookahead mask for compose
        compose_demo_reset_env_idxs = torch.LongTensor(np.intersect1d(self._compose_demo_targets_env_idxs.cpu().numpy(), env_ids.cpu().numpy())).cuda()
        self._global_demo_lookahead_mask_env2idxmap[compose_demo_reset_env_idxs] = self._sample_lookahead_mask_pool_indxs_for_composed_motions(len(compose_demo_reset_env_idxs)) #self._compose_compatible_lookahead_mask_pool_indx
        ############################################################################################################################################

        ############################################################################################################################################
        # Set new lookahead masks | Channel block masking
        self._global_demo_lookahead_mask[env_ids] = self.lookahead_mask_pool[self._global_demo_lookahead_mask_env2idxmap[env_ids]]
        ############################################################################################################################################

        ############################################################################################################################################
        if self._enable_lk_jl_mask:
            # Apply Random perturbation to the , Intra Channel Masking
            backup_mask = self._global_demo_lookahead_mask.clone()
            random_mask = self._sample_lookahead_joint_level_mask(len(env_ids))
            self._global_demo_lookahead_mask[env_ids] = (random_mask | self._global_demo_lookahead_mask[env_ids]).bool()

            # Dont mess with the upper body reset env idxs
            self._global_demo_lookahead_mask[rand_dir_speed_upper_body_reset_env_idxs] = backup_mask[rand_dir_speed_upper_body_reset_env_idxs]
        ############################################################################################################################################

        ############################################################################################################################################
        # Sanity Checks
        if len(rand_dir_speed_upper_body_reset_env_idxs) >0:
            lookahead_mask_pool_1_num = torch.unique(torch.sum(self._global_demo_lookahead_mask[rand_dir_speed_upper_body_reset_env_idxs], dim = -1)).view(-1)
            assert lookahead_mask_pool_1_num.item() == torch.sum(self.lookahead_mask_pool[self._upper_body_conditioning_compatible_mask_pool_indx]).item(), "Lookahead mask pool 1 is not set correctly"
        ############################################################################################################################################

        if  self._uniform_targets:
            self._global_demo_lookahead_mask[env_ids] = self._global_demo_lookahead_mask[env_ids[0]].unsqueeze(0).clone()

        if self._use_predefined_lookahead_mask:
            self._global_demo_lookahead_mask[env_ids] = self._predefined_lookahead_mask.unsqueeze(0).clone()

    def _sample_lookahead_joint_level_mask(self, num_samples):
        # Generate Random Mask
        mask_joint_prob = torch.rand(num_samples).unsqueeze(-1).cuda()
        mask = torch.zeros(num_samples,self._lookahead_obs_dim).type(torch.bool).cuda()
        joint_mask = torch.rand(num_samples,self.num_bodies).cuda() < mask_joint_prob

        if self._use_predefined_jl_mask:
            joint_mask[:,::] = False
            if self._predefined_jl_mask_joint_prob == 0:
                pass
            elif self._predefined_jl_mask_joint_prob == 0.25:
                joint_mask[:,::3] = True
            elif self._predefined_jl_mask_joint_prob == 0.5:
                joint_mask[:,::2] = True
            elif self._predefined_jl_mask_joint_prob == 0.75:
                joint_mask[:,::2] = True
                joint_mask[:,::3] = True

        body_mask = (joint_mask).repeat_interleave(3, dim = -1)

        # Make Sure to mask only the xyz and global xyz input.
        mask[:, :-2*self.num_bodies*3] = False
        mask[:, -2*self.num_bodies*3:-self.num_bodies*3] = body_mask
        mask[:, -2*self.num_bodies*3:-2*self.num_bodies*3 + 3] = False # Dont Mask Root
        mask[:, -self.num_bodies*3:] = body_mask
        mask[:, -self.num_bodies*3:-self.num_bodies*3 + 3] = False # Dont Mask Root

        # With 50% prob no random masking
        to_exclude_indxs = torch.nonzero(torch.rand(num_samples)>self._jl_mask_prob).view(-1)
        mask[to_exclude_indxs] = False

        return mask

    def _reset_amp_obs_mask(self, env_ids):
        ############################################################################################################################################
        # Sample Random Amp obs mask pool
        amp_obs_mask_pool_indxs = self._sample_amp_obs_mask_pool_indxs(len(env_ids))
        self._global_amp_obs_mask[env_ids] = self.amp_obs_mask_pool[amp_obs_mask_pool_indxs]
        # self.lookahead_mask_pool[torch.randint(0,len(self.lookahead_mask_pool), (len(env_ids),))]
        ############################################################################################################################################

        return

    def post_physics_step(self):
        self.extras["prev_lookahead_mask"] = self._global_demo_lookahead_mask.clone()
        self.extras["prev_tar_amp_obs"] = self._tar_flat_amp_obs.clone() #RobustAddtionalCodeLine
        self.extras["prev_tar_flat_lookahead_obs"] = self._tar_flat_lookahead_obs.clone() #RobustAddtionalCodeLine
        self.extras["prev_tar_lookahead_obs_mask"] = self._global_demo_lookahead_mask.clone() #RobustAddtionalCodeLine
        self.extras["prev_tar_full_state_obs"] = self._tar_flat_full_state_obs.clone() #RobustAddtionalCodeLine

        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._update_hist_amp_obs()
        self._compute_amp_observations()
        self._shift_tar_amp_keypos_obs()
        self._compute_curr_tar_amp_keypos_obs()
        self._compute_reward(self.actions)
        self._compute_reset()
        self._update_demo_targets()

        self.extras["terminate"] = self._terminate_buf
        self.extras["amp_obs"] = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["tar_amp_obs"] = self._tar_flat_amp_obs.clone() #RobustAddtionalCodeLine
        self.extras["tar_flat_lookahead_obs"] = self._tar_flat_lookahead_obs.clone() #RobustAddtionalCodeLine
        self.extras["tar_flat_full_state_obs"] = self._tar_flat_full_state_obs.clone() #RobustAddtionalCodeLine
        self.extras["tar_lookahead_mask"] = self._global_demo_lookahead_mask.clone()

        # debug viz
        if self.viewer and self.flags["debug_viz"]:
            self._update_debug_viz()

        self._prev_action_buf[:] = self.actions
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):
        """
        Used for Training the Discriminator
        """

        if (self._amp_obs_demo_buf is None):
            # Allocate Buffer for AMP Observations
            self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), 
                                                 device=self.device, dtype=torch.float32)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)

        # motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_ids = self._sample_demo_motions(num_samples)  # RobustAdditionalCodeLine

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        motion_ids, motion_times0 = motion_ids.to(self.device), motion_times0.to(self.device)

        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids.to(self._motion_lib._device), 
                                                motion_times.to(self._motion_lib._device))
        
        full_state_obs_demo, full_state_obs_info = ObsBuilder.get_full_state_obs_from_motion_frame(self, motion_ids, motion_times)
        amp_obs_demo = full_state_obs_demo[:, self._full_state_amp_obs_indxs]
        assert amp_obs_demo.size(-1) == self._amp_obs_dim, f"Assertion failed found size: {amp_obs_demo.size(-1)}"

        rand_rot = self._sample_random_rotation_quat_for_demo(len(amp_obs_demo)//self._num_amp_obs_steps).repeat_interleave(self._num_amp_obs_steps, dim = 0)
        return self._rotate_amp_obs(rotate_demo_by = rand_rot, amp_obs=amp_obs_demo)

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/amp_humanoid_sword_shield.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/smpl_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 69 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/digit-v3-open-chain.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 24 + 3 * num_key_bodies # just a place holder this is not correct
        elif (asset_file == "mjcf/digit-v3-open-chain_with_foot_spheres.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 24 + 3 * num_key_bodies # just a place holder this is not correct
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        return

    def _sample_random_rotation_quat_for_demo(self, num_samples):
        if self._state_demo_rotate:
            return quat_from_angle_axis(angle = torch.randint(0,360, size = (num_samples,)).cuda(), axis = torch.FloatTensor([[0,0,1]]).cuda())
        else:
            return quat_from_angle_axis(angle = torch.zeros(num_samples).cuda(), axis = torch.FloatTensor([[0,0,1]]).cuda())

    def _sample_unit_vectors(self, num_samples):
        return torch.nn.functional.normalize(torch.rand(num_samples, 2).cuda(), dim=1)

    def _sample_random_lookahead_diff_orient_speed(self, num_samples):
        r_angles = torch.randint(0,360, size = (num_samples,)).cuda()
        rand_theta = 2 * np.pi * torch.rand(num_samples, device=self.device) - np.pi

        if self._uniform_targets:
            r_angles[:] = r_angles[0].clone()
            rand_theta[:] = rand_theta[0].clone()

        lookahead = torch.zeros(num_samples, self._lookahead_obs_dim).cuda()
        # Random Height
        lookahead[:,2] = self._rigid_body_pos[:num_samples, 0, 2] if self._req_height is None else self._req_height
        # Random Orientation
        lookahead[:,3:7] = quat_from_angle_axis(angle = r_angles, axis = torch.FloatTensor([[0,0,1]]).cuda())
        # Random velocity
        rand_face_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        rand_speed = self._random_speed_min + torch.rand(num_samples, 1).cuda() * (self._random_speed_max - self._random_speed_min)
        lookahead[:,7:9] = rand_speed * rand_face_dir / torch.norm(rand_face_dir, dim = -1).unsqueeze(-1)
        return lookahead

    def _sample_random_rotation_quat_for_agent(self, num_samples):
        if self._state_init_rotate:
            return quat_from_angle_axis(angle = torch.randint(0,360, size = (num_samples,)).cuda(), axis = torch.FloatTensor([[0,0,1]]).cuda())
        else:
            return quat_from_angle_axis(angle = torch.zeros(num_samples).cuda(), axis = torch.FloatTensor([[0,0,1]]).cuda())

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)

        # RobustAdditionalCodeBlock
        # ------------------------------------------------------------
        # Setup Demo Ids and Times
        if len(env_ids) >0:
            self._reset_demo_targets(env_ids)
            self._reset_demo_lookahead_masks(env_ids)
            self._reset_amp_obs_mask(env_ids)
            self._reset_demo_tar_reset_steps(env_ids)
            self._init_tar_amp_keypos_obs(env_ids)
        # ------------------------------------------------------------
        return

    def _reset_envs_with_manual_targets(self, env_ids, demo_tar_m_ids, demo_tar_m_ts):
        if len(env_ids) ==0:
            return 

        assert self._uniform_targets == False , "logic not defined yet"

        self._reset_envs(env_ids)

        # self._reset_actors(env_ids)
        self._reset_ref_state_init(env_ids, demo_tar_m_ids, demo_tar_m_ts)
        self._reset_env_tensors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)

        self._init_amp_obs(env_ids)

        self._reset_demo_targets(env_ids, hard_start_demo_at_agent_sart=True)
        self._reset_demo_lookahead_masks(env_ids)
        self._reset_amp_obs_mask(env_ids)
        self._reset_demo_tar_reset_steps(env_ids)
        self._init_tar_amp_keypos_obs(env_ids)

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMPRobust.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMPRobust.StateInit.Start
              or self._state_init == HumanoidAMPRobust.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMPRobust.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

        # Default is basically zero motion time
        self._global_agent_start_motion_ids[env_ids] = 0 # RobustAddtionalCodeLine
        self._global_agent_start_motion_times[env_ids] = 0 # RobustAddtionalCodeLine
        self._global_agent_start_rotations[env_ids] = torch.cat((torch.zeros(len(env_ids), 3), torch.ones(len(env_ids), 1)), dim = -1).to(self.device) # RobustAddtionalCodeLine
        return

    def _reset_ref_state_init(self, env_ids, manual_motion_ids = None, manual_motion_timesteps = None):
        num_envs = env_ids.shape[0]

        # motion_ids = self._motion_lib.sample_motions(num_envs)
        if manual_motion_ids is not None and manual_motion_timesteps is not None:
            motion_ids = manual_motion_ids
            motion_times = manual_motion_timesteps
        else:
            motion_ids = self._sample_start_state_motions(num_envs) # RobustAddtionalCodeLine

            if (self._state_init == HumanoidAMPRobust.StateInit.Random
                or self._state_init == HumanoidAMPRobust.StateInit.Hybrid):
                motion_times = self._motion_lib.sample_time(motion_ids).to(self.device)
            elif (self._state_init == HumanoidAMPRobust.StateInit.Start):
                motion_times = torch.zeros(num_envs, device=self.device)
            else:
                assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids.to(self._motion_lib._device), motion_times.to(self._motion_lib._device))

        # RobustAdditionalCodeBlock
        # ------------------------------------------------------------
        rand_rot = self._sample_random_rotation_quat_for_agent(len(motion_ids)).to(self.device)
        zero_rot = torch.cat((torch.zeros(len(motion_ids), 3), torch.ones(len(motion_ids), 1)), dim = -1).to(self.device)

        agent_rot = zero_rot

        if self._state_init_rotate:
            # Randomize The Direction of Humanoid as well
            p = 0.5  # The probability of setting a motion ID to a random direction
            rand_rot_mask = torch.rand(len(motion_ids)) < p
            agent_rot[rand_rot_mask] = rand_rot[rand_rot_mask]

        # Track which state and time step was the agent set to in the start
        self._global_agent_start_motion_ids[env_ids] = motion_ids.to(self.device) # RobustAddtionalCodeLine
        self._global_agent_start_motion_times[env_ids] = motion_times.to(self.device) # RobustAddtionalCodeLine
        self._global_agent_start_rotations[env_ids] = agent_rot.to(self.device)
        # ------------------------------------------------------------

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=torch_utils.quat_mul(agent_rot.to(root_rot.device), root_rot), 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids.to(self.device)
        self._reset_ref_motion_times = motion_times.to(self.device)
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)

        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        # amp_motion_ids = self._global_demo_start_motion_ids[env_ids].repeat_interleave(len(time_steps)).unsqueeze(-1).repeat(1,32) # RobustAdditionalLine

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        full_state_obs_demo, full_state_obs_info = ObsBuilder.get_full_state_obs_from_motion_frame(self, motion_ids, motion_times)
        amp_obs_demo = full_state_obs_demo[:, self._full_state_amp_obs_indxs]
        assert amp_obs_demo.size(-1) == self._amp_obs_dim, f"Assertion failed found size: {amp_obs_demo.size(-1)}"

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos.to(self._humanoid_root_states.device)
        self._humanoid_root_states[env_ids, 3:7] = root_rot.to(self._humanoid_root_states.device)
        self._humanoid_root_states[env_ids, 7:10] = root_vel.to(self._humanoid_root_states.device)
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel.to(self._humanoid_root_states.device)

        self._dof_pos[env_ids] = dof_pos.to(self._dof_pos.device)
        self._dof_vel[env_ids] = dof_vel.to(self._dof_vel.device)
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    # RobustAdditionalCodeBlock
    # ------------------------------------------------------------
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            env_ids = self._all_env_ids

        if len(env_ids) == 0:
            return

        self._compute_full_state_observations(env_ids)
        self._curr_amp_obs_buf[env_ids] = self._curr_full_state_obs_buf[env_ids][:, self._full_state_amp_obs_indxs]
        self._curr_keypos_obs_buf[env_ids] = self._curr_full_state_obs_buf[env_ids][:, self._full_state_keypos_obs_indxs]

        # self._compute_lookahead_observations_from_full_state(env_ids)
        # self._compute_amp_observations_from_full_state(env_ids)
        # self._compute_keypos_observations_from_full_state(env_ids)

        return

    def _compute_full_state_observations(self, env_ids = None):
        all_body_pos = self._rigid_body_pos
        env_ids = self._all_env_ids if env_ids is None else env_ids
        self._curr_full_state_obs_buf[env_ids] = ObsBuilder.build_full_state_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                                    self._rigid_body_rot[env_ids][:, 0, :],
                                                                                    self._rigid_body_vel[env_ids][:, 0, :],
                                                                                    self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                                    self._dof_pos[env_ids], 
                                                                                    self._dof_vel[env_ids], 
                                                                                    all_body_pos[env_ids],
                                                                                    self.obs_flags["local_root_obs"], self.obs_flags["root_height_obs"], 
                                                                                    self._dof_obs_size, self._dof_offsets).to(self.device)

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.set_char_color(rand_col, env_ids)
        return

    @property
    def global_demo_start_motion_time_steps(self):
        return (self._global_demo_start_motion_times/self.dt).type(torch.long)

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        obs_size += 6 # oreintation observation
        obs_size += self.lk_embedding_dim
        if self._enable_body_pos_obs:
            obs_size += self._num_bodies * 3 # keypos observation
        return obs_size

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)

        humanoid_obs = self._compute_humanoid_obs(env_ids)
        dir_obs = self._compute_direction_obs(env_ids)

        demo_clip_idxs = self._global_demo_start_motion_ids[env_ids]
        demo_progress_idxs = (self.global_demo_start_motion_time_steps[env_ids] + self.progress_buf[env_ids] + 1) % self._motion_max_steps[demo_clip_idxs]

        demo_clip_idxs = demo_clip_idxs.unsqueeze(-1).repeat(1,self.lk_embedding_dim//2) # progress idx is repeated half of the times, use the first to get embedding.
        demo_progress_idxs = demo_progress_idxs.unsqueeze(-1).repeat(1,self.lk_embedding_dim//2) # progress idx is repeated half of times, use the first to get embedding.

        if self._enable_body_pos_obs:
            body_positions = self._rigid_body_pos[env_ids,:,:].view(len(env_ids), -1)
        else:
            body_positions = torch.FloatTensor([]).cuda()
        obs = torch.cat([demo_clip_idxs, demo_progress_idxs, dir_obs, humanoid_obs, body_positions], dim=-1)
        self.obs_buf[env_ids] = obs
        return

    def _compute_direction_obs(self, env_ids):
        root_rot = self._rigid_body_rot[:, 0, :]
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)

        if env_ids is None: 
            return root_rot_obs[:]
        else:
            return root_rot_obs[env_ids]

    def _compute_reward(self, actions):
        imit_rewards, info = RewardUtils.compute_imit_reward_using_full_body_state(full_body_state=self._curr_full_state_obs_buf , 
                                                                        tar_body_state=self.extras["prev_tar_full_state_obs"][:,:self._full_state_obs_dim], # holds next 10 timesteps, so we take the first 1. 
                                                                        lookahead_obs=self.extras["prev_tar_flat_lookahead_obs"][:, :self._lookahead_obs_dim], # holds next 10 timesteps, so we take the first 1. 
                                                                        lookahead_obs_mask=self.extras["prev_tar_lookahead_obs_mask"][:, :self._lookahead_obs_dim], # holds next 10 timesteps, so we take the first 1. 
                                                                        dof_obs_size=self._dof_obs_size,
                                                                        num_dofs=self._dof_offsets[-1],
                                                                        num_bodies=self.num_bodies, 
                                                                        key_body_ids=self._key_body_ids,
                                                                        helper_indexes=self.obs_mask_utility_indexes,
                                                                        ret_info = True,
                                                                        reward_config = {"use_shifted_xyz_reward": self.reward_properties.use_shifted_xyz_reward,
                                                                                         "keypos_reward_only": self.reward_properties.keypos_reward_only,
                                                                                         "keypos_big_weight_reward": self.reward_properties.keypos_big_weight_reward,
                                                                                         "lookahead_obs_split_indxs_dict": self._lookahead_obs_split_indxs, 
                                                                                         "dof_xyz_reward_weight":self.reward_properties.dof_xyz_reward_w}) # Added for Imitation

        energy_penalty = -0.0002*self._penalty_multiplyer*torch.sum(torch.abs(self.dof_force_tensor), dim = -1)
        control_penalty = -0.01*self._penalty_multiplyer*torch.sum(torch.abs(self._prev_action_buf - actions), dim = -1)
        all_penalties = energy_penalty + control_penalty

        self.rew_buf[:] = imit_rewards # this is used to return reward during the step.
        self.extras['energy_penalty'] = torch.clamp(all_penalties, min=-25, max=0).clone().unsqueeze(-1)  # It is used in buffer reward calculation. extras will be passed as info. 
        
        return

    # OVer load to add some cache mechanisms.
    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = InvKinMotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self._motion_lib_device)

        if self._motion_is_sampled_by_frames:
            self._demo_motion_frames_weights = torch.ones((self._motion_lib._motion_num_frames.sum(),), device=self.device)
            self._start_state_motion_frames_weights = torch.ones((self._motion_lib._motion_num_frames.sum(),), device=self.device)

            self.frame_to_motion_id = torch.zeros((self._motion_lib._motion_num_frames.sum(),), device=self.device, dtype=torch.long)
            # self.frame_to_motion_times = torch.zeros((self._motion_lib._motion_num_frames.sum(),), device=self.device, dtype=torch.float)
            for i in range(len(self._motion_lib._motion_num_frames)):
                self.frame_to_motion_id[self._motion_lib._motion_num_frames[:i].sum():self._motion_lib._motion_num_frames[:i+1].sum()] = i
                # self.frame_to_motion_times[self._motion_lib._motion_num_frames[:i].sum():self._motion_lib._motion_num_frames[:i+1].sum()] = torch.arange(self._motion_lib._motion_num_frames[i], device=self.device) * self._motion_lib._motion_dt[i]

        return

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.pre_render_hook_fxn()
            self.render()
            self.gym.simulate(self.sim)
        return

    def pre_render_hook_fxn(self):
        if hasattr(self, 'pre_render_hook_obj'):
            if hasattr(self.pre_render_hook_obj, 'pre_render_hook_fxn'):
                self.pre_render_hook_obj.pre_render_hook_fxn()
        return None
