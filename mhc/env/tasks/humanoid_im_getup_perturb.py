# Imports

# Isaac Gym Imports
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

# Env Imports
from env.tasks.humanoid_im_getup_task_viz import HumanoidAMPGetupRobustTaskViz


# -------------------------------------------------------------------
# --------------- Perturb Mimic Viz Wrapper ---------------------------------
# -------------------------------------------------------------------
PERTURB_OBJS = [
    ["small", 1],
    ["small", 2],
    ["small", 3],
    ["small", 90],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],   
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],    
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 1],
    ["small", 3],
    ["small", 2],
    ["small", 2],
    ["small", 3],
    ["small", 90],
    ["small", 1],
    ["small", 1],
    ["large", 90],
    ["large", 1],
    ["large", 1],
    ["large", 1],
    ["large", 1],
    ["large", 1],
    ["large", 1],
]

class HumanoidAMPGetupPerturb(HumanoidAMPGetupRobustTaskViz):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40
        assert(self._proj_warmup_steps < self._proj_steps)

        self._build_proj_tensors()
        self._calc_perturb_times()


        # Placeholder Variables
        self.real_traj = False
        self.show_traj = False
        self.show_root_traj = False
        self.show_headings = False

        self._enable_perturbation = True 
        self._last_toggle_timestep = 0
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        self._proj_handles = []
        self._load_proj_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_proj(env_id, env_ptr)
        return

    def _load_proj_asset(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]

        small_asset_file = "mjcf/block_projectile.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 200.0
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)
        
        large_asset_file = "mjcf/block_projectile_large.urdf"
        large_asset_options = gymapi.AssetOptions()
        large_asset_options.angular_damping = 0.01
        large_asset_options.linear_damping = 0.01
        large_asset_options.max_angular_velocity = 100.0
        large_asset_options.density = 100.0
        large_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._large_proj_asset = self.gym.load_asset(self.sim, asset_root, large_asset_file, large_asset_options)
        return

    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 200 + i
            default_pose.p.z = 1
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), col_group, col_filter, segmentation_id)
            self._proj_handles.append(proj_handle)

        return

    # def _build_body_ids_tensor(self, env_ptr, actor_handle, body_names):
    #     env_ptr = self.envs[0]
    #     actor_handle = self.humanoid_handles[0]
    #     body_ids = []

    #     for body_name in body_names:
    #         body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
    #         assert(body_id != -1)
    #         body_ids.append(body_id)

    #     body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
    #     return body_ids

    def _build_proj_tensors(self):
        num_actors = self.get_num_actors_per_env()
        num_objs = self._get_num_objs()
        self._proj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., (num_actors - num_objs):, :]
        
        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, device=self.device, dtype=torch.int32)
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._proj_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]
        
        return

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_OBJS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._proj_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset_v2(self.reset_buf, self.progress_buf,
                                                               self._contact_forces, self._contact_body_ids,
                                                               self._rigid_body_pos, self.max_episode_length,
                                                               self._enable_early_termination, self._termination_heights)
        return

    def post_physics_step(self):
        self._update_proj()
        super().post_physics_step()
        return
    
    def _get_num_objs(self):
        return len(PERTURB_OBJS)

    def _update_proj(self):
        
        curr_timestep = self.progress_buf.cpu().numpy()[0]
        curr_timestep = curr_timestep % (self._perturb_timesteps[-1] + 1)
        perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]
        
        if (len(perturb_step) > 0 and self._enable_perturbation):
            # print("Perturbation Called")
            perturb_id = perturb_step[0]
            n = self.num_envs
            humanoid_root_pos = self._humanoid_root_states[..., 0:3]

            rand_theta = torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device)
            rand_theta *= 2 * np.pi
            rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_dist_min
            pos_x = rand_dist * torch.cos(rand_theta)
            pos_y = -rand_dist * torch.sin(rand_theta)
            pos_z = (self._proj_h_max - self._proj_h_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_h_min
            
            self._proj_states[..., perturb_id, 0] = humanoid_root_pos[..., 0] + pos_x
            self._proj_states[..., perturb_id, 1] = humanoid_root_pos[..., 1] + pos_y
            self._proj_states[..., perturb_id, 2] = pos_z
            self._proj_states[..., perturb_id, 3:6] = 0.0
            self._proj_states[..., perturb_id, 6] = 1.0
            
            tar_body_idx = np.random.randint(self.num_bodies)
            tar_body_idx = 1

            launch_tar_pos = self._rigid_body_pos[..., tar_body_idx, :]
            launch_dir = launch_tar_pos - self._proj_states[..., perturb_id, 0:3]
            launch_dir += 0.1 * torch.randn_like(launch_dir)
            launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
            launch_speed = (self._proj_speed_max - self._proj_speed_min) * torch.rand_like(launch_dir[:, 0:1]) + self._proj_speed_min
            launch_vel = launch_speed * launch_dir
            launch_vel[..., 0:2] += self._rigid_body_vel[..., tar_body_idx, 0:2]
            self._proj_states[..., perturb_id, 7:10] = launch_vel
            self._proj_states[..., perturb_id, 10:13] = 0.0

            marker_ids = torch.cat([self._heading_marker_actor_ids, self._face_marker_actor_ids, self._body_marker_actor_ids], dim=0)
            marker_and_proj_ids = _proj_actor_ids = torch.cat([marker_ids, self._proj_actor_ids], dim=0)

            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                         gymtorch.unwrap_tensor(marker_and_proj_ids),
                                                         len(marker_and_proj_ids))
        else:
            marker_ids = torch.cat([self._heading_marker_actor_ids, self._face_marker_actor_ids, self._body_marker_actor_ids], dim=0)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                         gymtorch.unwrap_tensor(marker_ids),
                                                         len(marker_ids))

        return

    def _draw_task(self):
        self._update_marker_tensors()
        # self._set_marker_positions() # Dont set marker positions here 
        return

@torch.jit.script
def compute_humanoid_reset_v2(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    
    terminated = torch.zeros_like(reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
