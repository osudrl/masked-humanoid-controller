# Imports

# Isaac Gym Imports
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

# Python Imports
import torch

# Env Imports
from env.tasks.humanoid_im_getup import HumanoidAMPGetupRobust

# Utils Imports
from utils import torch_utils


# -------------------------------------------------------------------
# --------------- Mimic Viz Wrapper ---------------------------------
# -------------------------------------------------------------------

TAR_HEADING_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2
TAR_BODY_ACTOR_ID_START = 3

class HumanoidAMPGetupRobustTaskViz(HumanoidAMPGetupRobust):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # self._enable_task_viz = cfg["env"]["enableTaskViz"]

        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self._enable_task_viz = True
        self.num_envs = cfg["env"]["numEnvs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        if not self.headless:
            self._build_marker_state_tensors()
            print("MArkers Created")

        self.real_traj = False
        self.show_traj = False
        self.show_root_traj = False
        self.show_headings = False
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._heading_marker_handles = []
            self._face_marker_handles = []
            self._body_marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        # self._reset_task_viz(env_ids)
        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _draw_task(self):
        self._update_marker_tensors()
        self._set_marker_positions()

        return

    def _set_marker_positions(self):
        marker_ids = torch.cat([self._heading_marker_actor_ids, self._face_marker_actor_ids, self._body_marker_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                        gymtorch.unwrap_tensor(marker_ids), len(marker_ids))
        
    def _update_marker_tensors(self):
        ####################################################################################        
        viz_tar_dir  = (self._tar_flat_lookahead_obs[:,7:9]/torch.norm(self._tar_flat_lookahead_obs[:,7:9], dim = -1).unsqueeze(-1))
        ####################################################################################

        if self.show_headings:
            humanoid_root_pos = self._humanoid_root_states[..., 0:3]
            self._heading_marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + viz_tar_dir
            self._heading_marker_pos[..., 2] = 0.0

            heading_theta = torch.atan2(viz_tar_dir[..., 1], viz_tar_dir[..., 0])
            heading_axis = torch.zeros_like(self._heading_marker_pos)
            heading_axis[..., -1] = 1.0
            heading_q = quat_from_angle_axis(heading_theta, heading_axis)
            self._heading_marker_rot[:] = heading_q

            ####################################################################################
            heading_rot = torch_utils.calc_heading_quat(self._tar_flat_lookahead_obs[:,3:7]) 
            facing_dir = torch.zeros_like(self._tar_flat_lookahead_obs[:,:3])
            facing_dir[..., 0] = 1.0
            viz_tar_facing_dir = quat_rotate(heading_rot, facing_dir)[:, :2]
            ###############################################################################

            self._face_marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + viz_tar_facing_dir
            self._face_marker_pos[..., 2] = 0.0

            face_theta = torch.atan2(viz_tar_facing_dir[..., 1], viz_tar_facing_dir[..., 0])
            face_axis = torch.zeros_like(self._face_marker_pos)
            face_axis[..., -1] = 1.0
            face_q = quat_from_angle_axis(face_theta, heading_axis)
            self._face_marker_rot[:] = face_q
        else:
            self._heading_marker_pos[:] = 1000
            self._face_marker_pos[:] = 1000



        ##### Render Body Ids. ########
        # Here show root traj means it is not localized. # Todo rename variables
        if self.show_traj or self.show_root_traj:
            # self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)
        
            try:
                tar_lookahead_obs = self.extras["prev_tar_flat_lookahead_obs"][:, :self._lookahead_obs_dim]
            except:
                tar_lookahead_obs = self._tar_flat_lookahead_obs.clone()[:, :self._lookahead_obs_dim] 

            root_pos = tar_lookahead_obs[:, self._lookahead_obs_split_indxs["root_pos"]]
            
            # _track_bodies_id = 
            if self.show_root_traj:
                _track_bodies_id = torch.arange(self._rigid_body_pos.size(-2))
                curr_pos = self._rigid_body_pos[:,_track_bodies_id,:].clone()
                tar_pos = tar_lookahead_obs[:,self._lookahead_obs_split_indxs["dof_xyz"]].view(self.num_envs,-1 ,3)[:,_track_bodies_id,:].clone()
                shfited_tar_pos = tar_pos 
                
                root_shift = self._rigid_body_pos[:,0,:2].clone()
                curr_root = tar_pos[:,0,:2].clone()
                shfited_tar_pos[:,:,:2] = tar_pos[:,:,:2] - curr_root.unsqueeze(1) + root_shift.unsqueeze(1)
            else:
                _track_bodies_id = torch.arange(self._rigid_body_pos.size(-2))
                curr_pos = self._rigid_body_pos[:,_track_bodies_id,:].clone()
                tar_local_pos = tar_lookahead_obs[:,self._lookahead_obs_split_indxs["dof_local_xyz"]].view(self.num_envs,-1 ,3)[:,_track_bodies_id,:].clone()
                
                root_rot = self._rigid_body_rot[:, 0, :]
                heading_rot = torch_utils.calc_heading_quat(root_rot)
                tar_local_pos = quat_rotate(heading_rot.repeat_interleave(self.num_bodies, dim = 0),
                                            tar_local_pos.view(-1,3)).view(self.num_envs,-1,3)
                
                root_shift = curr_pos[:,0,:].clone()
                curr_root = tar_local_pos[:,0,:].clone()
                shfited_tar_pos = tar_local_pos - curr_root.unsqueeze(1) + root_shift.unsqueeze(1)

                # shfited_tar_pos = quat_rotate(root_rot.repeat_interleave(len(_track_bodies_id), dim = 0),
                #                 shfited_tar_pos.view(-1,3)).view(self.num_envs,len(_track_bodies_id),3)

            # Rotate shifted_tar_pos to current robot heading
            ##############################################
            # root_rot = self._rigid_body_rot[:, 0, :]
            # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
            
            # heading_rot_expand = heading_rot.unsqueeze(-2)
            # heading_rot_expand = heading_rot_expand.repeat((1, shfited_tar_pos.shape[1], 1))
            # flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
            # shfited_tar_pos = quat_rotate(flat_heading_rot, shfited_tar_pos.view(-1,3)).view(self.num_envs,-1,3)
            
            ##############################################
            
            ## Only update the tracking points. 
            if self.real_traj:
                self._body_marker_pos[:] = 1000
                
            self._body_marker_pos[..., _track_bodies_id, :] = shfited_tar_pos[..., _track_bodies_id, :]

            try:
                tar_lookahead_obs_mask_1 = self.extras["prev_tar_lookahead_obs_mask"][:,self._lookahead_obs_split_indxs["dof_xyz"]].type(torch.bool).view(self.num_envs, -1, 3)
                tar_lookahead_obs_mask_2 = self.extras["prev_tar_lookahead_obs_mask"][:,self._lookahead_obs_split_indxs["dof_local_xyz"]].type(torch.bool).view(self.num_envs, -1, 3)
                tar_lookahead_obs_mask = tar_lookahead_obs_mask_1 * tar_lookahead_obs_mask_2
            except: 
                tar_lookahead_obs_mask = torch.zeros_like(tar_lookahead_obs[:,self._lookahead_obs_split_indxs["dof_xyz"]]).type(torch.bool).view(self.num_envs, -1, 3)
            self._body_marker_pos[tar_lookahead_obs_mask] = 1000

        else:
            self._body_marker_pos[:] = 1000

        # ######### Heading debug #######
        # points = self.init_root_points()
        # base_quat = self._rigid_body_rot[0, 0:1]
        # base_quat = remove_base_rot(base_quat)
        # heading_rot = torch_utils.calc_heading_quat(base_quat)
        # show_points = quat_apply(heading_rot.repeat(1, points.shape[0]).reshape(-1, 4), points) + (self._rigid_body_pos[0, 0:1]).unsqueeze(1)
        # self._marker_pos[:] = show_points[:, :self._marker_pos.shape[1]]
        # ######### Heading debug #######

        # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(self._body_marker_actor_ids), len(self._body_marker_actor_ids))

        return

    def _load_marker_asset(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = "mjcf/heading_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._heading_marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = "mjcf/traj_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._body_marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)


        return
  
    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        default_pose.p.z = 0.0
        
        marker_handle = self.gym.create_actor(env_ptr, self._heading_marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._heading_marker_handles.append(marker_handle)
        
        face_marker_handle = self.gym.create_actor(env_ptr, self._heading_marker_asset, default_pose, "face_marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, face_marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.8))
        self._face_marker_handles.append(face_marker_handle)

        default_pose = gymapi.Transform()
        # self.num_bodies = 17
        for i in range(self.num_bodies):
            marker_handle = self.gym.create_actor(env_ptr, self._body_marker_asset, default_pose, "marker", self.num_envs + 10, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
            # if i in self._track_bodies_id:
            # else:
            #     self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0))
            self._body_marker_handles[env_id].append(marker_handle)

        
        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._heading_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_HEADING_ACTOR_ID, :]
        self._heading_marker_pos = self._heading_marker_states[..., :3]
        self._heading_marker_rot = self._heading_marker_states[..., 3:7]
        self._heading_marker_actor_ids = self._humanoid_actor_ids + TAR_HEADING_ACTOR_ID

        self._face_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_FACING_ACTOR_ID, :]
        self._face_marker_pos = self._face_marker_states[..., :3]
        self._face_marker_rot = self._face_marker_states[..., 3:7]
        self._face_marker_actor_ids = self._humanoid_actor_ids + TAR_FACING_ACTOR_ID

        self._body_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_BODY_ACTOR_ID_START:(TAR_BODY_ACTOR_ID_START + self.num_bodies), :]
        self._body_marker_pos = self._body_marker_states[..., :3]
        self._body_marker_rotation = self._body_marker_states[..., 3:7]

        self._body_marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._body_marker_handles, dtype=torch.int32, device=self.device)
        self._body_marker_actor_ids = self._body_marker_actor_ids.flatten()

        return
