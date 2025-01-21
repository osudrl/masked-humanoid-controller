# Imports

# Isaac Gym Imports
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

# Python Imports
import wandb
from types import MethodType

# AMP Imports
from learning.amp_players import AMPPlayerContinuous

# Env Imports
from env.tasks.humanoid_im import HumanoidAMPRobust

# RL Games Imports
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import players
from rl_games.algos_torch.running_mean_std import RunningMeanStd

# Utils Imports
from utils.benchmark_helper import generate_cache_for_env_vars
from utils.benchmark_helper import restore_env_vars_from_cache
from utils.benchmark_helper import benchmark_policy


# --------------------------------------------
# -------------------Play---------------------
# --------------------------------------------

class AMPPlayer2(AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        self.num_envs = self.env.task.num_envs
        self.num_demo_envs = 1
        self.motion_files = [f.split('/')[-1] for f in self.env.task._motion_lib._motion_files]
        self.motion_lengths = self.env.task._motion_lib._motion_lengths
        self.n_motions = len(self.motion_lengths)

        cam_pos = gymapi.Vec3(0, -6.0, 3.0)
        cam_target = gymapi.Vec3(0, 0, 2.0)
        self.env.task.gym.viewer_camera_look_at(self.env.task.viewer, None, cam_pos, cam_target)

        # Set visualization artifacts off
        self._render_demo = False
        self._render_demo_composition = False
        self._render_target_markers = False
        self._render_anchored_demo = False
        self._speed_adaptive_color = False
        self._to_anchor_env_idx = 0
        self._anchor_to_env_idx = 1
        self._render_shifted_demo = False
        self._dynamic_weighting = False
        self._zap_start = 10
        self._zap_end = 60

        self._prev_demo_id = None
        self._prev_demo_color = (0,1,0)
        self._masked_out_color = (0,0,0)

        self._render_color_for_speed = False

        self.benchmark_policy = MethodType(benchmark_policy, self)

        self.env.task.pre_render_hook_obj = self 
        return


    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape_wo_mask']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return
    
    def localize_lookahead_obs(self, lookahead_obs_flat_batch, lookahead_obs_mask):
        return self.env.task.localize_lookahead_obs(lookahead_obs_flat_batch, 
                                                    lookahead_obs_mask = lookahead_obs_mask,
                                                    num_stacks= self.env.task._lookahead_timesteps)

    def _load_config_params(self, config):
        super()._load_config_params(config)

    def _build_net_config(self):
        config = super()._build_net_config()
        config['lookahead_tar_flat_obs_shape'] = (self.env.task.lookahead_tar_flat_dim,)
        config['lookahead_tar_mask_shape'] = (self.env.task._lookahead_obs_dim,)
        config['amp_mask_shape'] = (self.env.task._amp_obs_dim,)
        config['lookahead_timesteps'] = self.env.task._lookahead_timesteps
        config['amp_input_shape'] = (self.env.amp_observation_space.shape[0] + self.env.task._amp_obs_dim, )
        config['amp_input_shape_wo_mask'] = (self.env.amp_observation_space.shape[0],) 
        config['lookahead_factor_dims'] = self.env.task._lookahead_factor_dims
        return config

    def _preproc_obs(self, obs_batch):
        lk_embedding_dim  = self.env.task.lk_embedding_dim
        new_obs_batch = super()._preproc_obs(obs_batch)
        new_obs_batch[...,:lk_embedding_dim] = obs_batch[...,:lk_embedding_dim] # dont change the indexes        
        return new_obs_batch
    
    def obs_to_torch(self, obs):
        obs_dict = super().obs_to_torch(obs)
        obs_dict['lookahead_tar_obs'] = self.env.task.lookahead_tar_flat_obs  
        obs_dict['amp_obs'] = self.env.task._curr_amp_obs_buf  
        obs_dict['keypos_obs'] = self.env.task._curr_keypos_obs_buf
        obs_dict['full_state_obs'] = self.env.task._curr_full_state_obs_buf
        obs_dict['lookahead_tar_mask'] = self.env.task._global_demo_lookahead_mask
        obs_dict['amp_obs_mask'] = self.env.task._global_amp_obs_mask
        return obs_dict

    
    @torch.no_grad()
    def get_action(self, obs_dict, is_determenistic=False):

        obs = obs_dict['obs']
        lookahead_tar_obs = self.localize_lookahead_obs(obs_dict['lookahead_tar_obs'], obs_dict['lookahead_tar_mask'])
        lookahead_tar_mask = obs_dict['lookahead_tar_mask']

        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)

        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states,
            'lookahead_tar_obs': lookahead_tar_obs,
            'lookahead_tar_mask': lookahead_tar_mask
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = torch.squeeze(current_action.detach())

        return  players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))


    def _amp_debug(self, info):
        return None
    
    @torch.no_grad()
    def _eval_disc(self, amp_obs, amp_obs_mask):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs, amp_obs_mask)
    
    def _reshape_amp_obs_mask(self, amp_obs, amp_obs_mask):
        if len(amp_obs.shape) == 2:
            amp_obs_mask_reshaped = amp_obs_mask.unsqueeze(0).repeat(amp_obs.shape[0], 1)
        elif len(amp_obs.shape) == 3:
            amp_obs_mask_reshaped = amp_obs_mask.unsqueeze(0).unsqueeze(0).repeat(amp_obs.shape[0], amp_obs.shape[1], 1)
        else:
            assert False
        return amp_obs_mask_reshaped

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)

        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        
        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint['model']
        
        # Hacks to make the pretrained model work with the current model
        l = list(pretrained_dict.keys())
        for k in l:
            if "actor_cnn.hist_embedding" in k:
                pretrained_dict[k.replace("actor_cnn.hist_embedding","lookahead_tar_enc")] = pretrained_dict[k]

        print("Parameters Omitted: ", [k for k, v in pretrained_dict.items() if k not in model_dict])
        print("Parameters Missing: ", [k for k, v in model_dict.items() if k not in pretrained_dict])
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)  

    #####################################################################
    ###=========================Buffer Logic =========================###
    #####################################################################
    def create_env_buffer(self, size, num_envs = None, num_keypos = 6, action_horizon = 10):
        play_steps = size
        num_envs = num_envs or self.env.num_envs
        action_dim = self.env.task.get_action_size()
        obs_dim = self.env.task.get_obs_size()
        keypos_obs_dim = self.env.task.num_bodies*3
        full_state_dim = self.env.task._full_state_obs_dim
        amp_obs_dim = self.env.task._num_amp_obs_per_step
        num_keypos = len(self.env.task._key_body_ids)

        buffer = {"obs":  torch.zeros((play_steps, num_envs, obs_dim) ).to("cpu"),
                "next_obs":  torch.zeros((play_steps, num_envs, obs_dim)).to("cpu"),

                "amp_obs":  torch.zeros((play_steps, num_envs, amp_obs_dim)).to("cpu"),
                "next_amp_obs":  torch.zeros((play_steps, num_envs, amp_obs_dim)).to("cpu"),
                
                "full_state_obs": torch.zeros((play_steps, num_envs, full_state_dim)).to("cpu"),
                "next_full_state_obs": torch.zeros((play_steps, num_envs, full_state_dim)).to("cpu"),

                "keypos_obs":  torch.zeros((play_steps, num_envs, keypos_obs_dim)).to("cpu"),
                "next_keypos_obs":  torch.zeros((play_steps, num_envs, keypos_obs_dim)).to("cpu"),

                "actions": torch.zeros((play_steps, num_envs, action_dim)).to("cpu"),
                "rewards": torch.zeros((play_steps, num_envs)).to("cpu"),
                "dones": torch.zeros((play_steps, num_envs)).to("cpu"),
                "terminated": torch.zeros((play_steps, num_envs)).to("cpu"),

                "root_pos": torch.zeros((play_steps, num_envs, 3), dtype=torch.float32, device="cpu"),
                "root_rot": torch.zeros((play_steps, num_envs, 4), dtype=torch.float32, device="cpu"),
                "root_vel": torch.zeros((play_steps, num_envs, 3), dtype=torch.float32, device="cpu"),
                "root_ang_vel": torch.zeros((play_steps, num_envs, 3), dtype=torch.float32, device="cpu"),
                "dof_pos": torch.zeros((play_steps, num_envs, action_dim), dtype=torch.float32, device="cpu"),
                "dof_vel": torch.zeros((play_steps, num_envs, action_dim), dtype=torch.float32, device="cpu"),
                "local_key_pos": torch.zeros((play_steps, num_envs, num_keypos * 3), dtype=torch.float32, device="cpu"),
                
                "demo_lengths": torch.zeros((play_steps, num_envs), dtype=torch.float32, device="cpu"),
                # "target_keypos_obs_lookahead": torch.zeros((play_steps, num_envs, keypos_obs_dim*lookahead_timesteps), dtype=torch.float32, device="cpu"),
                # "target_amp_obs_lookahead": torch.zeros((play_steps, num_envs, amp_obs_dim*lookahead_timesteps), dtype=torch.float32, device="cpu"),
                # "extended_keypos_obs_lookahead": torch.zeros((play_steps, num_envs, keypos_obs_dim*(lookahead_timesteps+action_horizon) ), dtype=torch.float32, device="cpu"),
                }
        
        return buffer


    #####################################################################
    ###=======================Collect Dataset ========================###
    #####################################################################
    def visualize(self, demo_idx_pool = [1], random_start = False, render_demo = True,
                        render_target_markers = True, mimic_random_root_poses = False,
                        enable_lookahead_mask = False,
                        upper_body_conditioning = False, 
                        compose_demo = False,
                        uniform_targets = True, 
                        show_traj = False, 
                        show_headings = False, 
                        show_root_traj = False,
                        random_joints_mask = False,
                        random_channels = False,
                        key_joints_only = False, 
                        collect_dataset = False):
        
        cached_vars = generate_cache_for_env_vars(self.env.task)

        self._render_demo = render_demo
        self._render_target_markers = render_target_markers

        # Reset ############################################################################################################
        self.env.task._demo_motion_weights[:] = 0
        self.env.task._demo_motion_weights[demo_idx_pool] = 1
        self.env.task._demo_motion_weights = self.env.task._demo_motion_weights / torch.sum(self.env.task._demo_motion_weights)
        # self.env.task._fixed_demo_lookahead_mask_env2idxmap_flag = True if enable_lookahead_mask else False
        # ################################################################################

        # Reset Tensors
        self.env.task._random_dir_speed_lookahead_share = 0 #cfg["env"]["random_dir_speed_lookahead_share"]

        self.env.task._demo_env_offset = self.env.task._init_random_dir_env_indexes(start_at = 0)
        self.env.task._random_dir_speed_mimic_upper_prob = 0
        self.env.task._init_random_dir_upper_env_indexes()
        self._compose_demo_targets_share = 0
        self.env.task._init_compose_env_indexes(start_at = self.env.task._demo_env_offset)
        
        if random_start:
            self.env.task._start_state_motion_weights[:] = 1/len(self.env.task._start_state_motion_weights)
            self.env.task._state_init = HumanoidAMPRobust.StateInit.Random
            self.env.task._demo_init = HumanoidAMPRobust.StateInit.Start
            self.env.task._state_init_rotate = True
            self.env.task._state_demo_rotate = True
            self.env.task._fall_init_prob = 0.1
            self.env.task._start_demo_at_agent_state_prob = 0
            self.env.task._uniform_targets = uniform_targets
            self.env.task._switch_demos_within_episode = True
            self.env.task._demo_tar_reset_steps_max = 100
            self.env.task._demo_tar_reset_steps_min = 50
            self.env.task._enable_lookahead_mask = enable_lookahead_mask
            self.env.task.lookahead_mask_pool_sample_weights[:] = 1 if enable_lookahead_mask else 0
            self.env.task.lookahead_mask_pool_sample_weights[0] = 1 # First one is the default mask full information
        else:
            self.env.task._start_state_motion_weights[:] = 0
            self.env.task._start_state_motion_weights[demo_idx_pool] = 1/len(demo_idx_pool)
            self.env.task._state_init = HumanoidAMPRobust.StateInit.Start
            self.env.task._demo_init = HumanoidAMPRobust.StateInit.Start
            self.env.task._state_init_rotate = False
            self.env.task._state_demo_rotate = False
            self.env.task._fall_init_prob = 0
            self.env.task._start_demo_at_agent_state_prob = 0
            self.env.task._uniform_targets = uniform_targets
            self.env.task._switch_demos_within_episode = False
            self.env.task._enable_lookahead_mask = enable_lookahead_mask
            self.env.task.lookahead_mask_pool_sample_weights[:] = 1 if enable_lookahead_mask else 0
            self.env.task.lookahead_mask_pool_sample_weights[0] = 1 # First one is the default mask full information
        
        if mimic_random_root_poses:
            self.env.task._random_dir_speed_lookahead_share = 1 #cfg["env"]["random_dir_speed_lookahead_share"]
            self.env.task._demo_env_offset = self.env.task._init_random_dir_env_indexes(start_at = 0)
            self.env.task._random_dir_speed_mimic_upper_prob = 1 if upper_body_conditioning else 0
            self.env.task._init_random_dir_upper_env_indexes()
            self._compose_demo_targets_share = 0
            self.env.task._init_compose_env_indexes(self.env.task._demo_env_offset)


            self.env.task._random_dir_speed_env_idxs = torch.arange(self.num_envs).type(torch.LongTensor).cuda()
            self.env.task._demo_tar_reset_steps_max = 151
            self.env.task._demo_tar_reset_steps_min = 150
            self.env.task._switch_demos_within_episode = True

            self.env.task.show_traj = False
            self.env.task.show_headings = True
            self.env.task.show_root_traj = True

            self._render_color_for_speed = True # Activates logic for random speed. 
        else:
            self.env.task._random_dir_speed_env_idxs = torch.zeros([0]).type(torch.LongTensor).cuda()
            self.env.task.show_traj = show_traj
            self.env.task.show_headings = show_headings
            self.env.task.show_root_traj = show_root_traj

            self._render_color_for_speed = False

        if compose_demo:
            self.env.task._enable_lookahead_mask = True
            self.env.task.lookahead_mask_pool_sample_weights[:] = 0 # First one is the default mask full information
            self.env.task.lookahead_mask_pool_sample_weights[self.env.task._compose_compatible_lookahead_mask_pool_indx] = 1 # First one is the default mask full information
            if self.env.task.num_envs == 3:
                self.env.task._compose_demo_targets_env_idxs = torch.arange(1, 1+1).type(torch.LongTensor).cuda() # Last Environment cannot be composed.
                self.env.task._predefined_target_demo_start_motion_ids[[0,1]] = demo_idx_pool[0]
                self.env.task._predefined_target_demo_start_motion_ids[2] = demo_idx_pool[1]
            else:
                # Set Tensors According to desired composition
                self.env.task._compose_demo_targets_env_idxs = torch.arange(2, self.num_envs).type(torch.LongTensor).cuda() # Last Environment cannot be composed.
                source_demo_ids = torch.arange(0,self.num_envs,2).type(torch.LongTensor).cuda()
                target_demo_ids = torch.arange(1,self.num_envs,2).type(torch.LongTensor).cuda()
                self.env.task._predefined_target_demo_start_motion_ids[source_demo_ids] = demo_idx_pool[0]
                self.env.task._predefined_target_demo_start_motion_ids[target_demo_ids] = demo_idx_pool[1]
            self.env.task._use_predefined_targets = True
            self._render_demo = False
            self._render_demo_composition = True

        if random_joints_mask:
            self.env.task._enable_lk_jl_mask = True
            self.env.task._jl_mask_prob = 1  
            self.env.task._enable_lookahead_mask = True
            self.env.task.lookahead_mask_pool_sample_weights[:] = 0 if enable_lookahead_mask else 0
            self.env.task.lookahead_mask_pool_sample_weights[7] = 1 # First one is the default mask full information
            self.env.task.lookahead_mask_pool_sample_weights[5] = 1 # First one is the default mask full information
            self.env.task.lookahead_mask_pool_sample_weights[3] = 1 # First one is the default mask full information

        if random_channels:
            self.env.task.lookahead_mask_pool_sample_weights[:] = 0 if enable_lookahead_mask else 0
            self.env.task.lookahead_mask_pool_sample_weights[0] = 1 # First one is the default mask full information
            self.env.task.lookahead_mask_pool_sample_weights[2] = 1 # First one is the default mask full information
            self.env.task.lookahead_mask_pool_sample_weights[3] = 1 # First one is the default mask full information
            self.env.task.lookahead_mask_pool_sample_weights[5] = 1 # First one is the default mask full information
        

        if key_joints_only: 
            self.env.task._enable_lk_jl_mask = False
            self.env.task._jl_mask_prob = 0
            self.env.task._enable_lookahead_mask = True
            self.env.task.lookahead_mask_pool_sample_weights[:] = 0 if enable_lookahead_mask else 0
            self.env.task.lookahead_mask_pool_sample_weights[4] = 1 # First one is the default mask full information
            self.env.task.lookahead_mask_pool_sample_weights[6] = 1 # First one is the default mask full information

        self.env_reset(torch.arange(self.env.task.num_envs).cuda())
        ###################################################################################################################


        # Reset Environment before visualization
        self.env.task._reset_envs(torch.arange(self.num_envs).cuda())
        self.env.task.gym.simulate(self.env.task.sim)
        # self.env.task.gym.render(self.env.task.sim)
        
        if collect_dataset:
            buffer = self.collect_dataset(self.env.task.max_episode_length)
        else:
            self.games_num = 1
            self.run()
            buffer = None

        # Set visualization artifacts off
        self._render_demo = False
        self._render_target_markers = False
        self._set_char_color((0,0,1), torch.arange(self.env.num_envs).cuda())


        self._render_demo_composition = False

        # Restore Vars
        restore_env_vars_from_cache(self.env.task, cached_vars)

        return buffer

    def _set_char_color(self, r, env_ids = None):
        task = self.env.task

        if env_ids is None:
            env_ids = torch.arange(task.num_envs).cuda()
            
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col[:] = r
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        task.set_char_color(rand_col, env_ids)
        return
    

    def _set_char_color_for_body_subset(self, r, env_ids = None, body_subset = "upper_rigid_body_indxs"):
        task = self.env.task

        if env_ids is None:
            env_ids = torch.arange(task.num_envs).cuda()
            
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col[:] = r
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        
        col = rand_col
        for env_id in env_ids:
            env_ptr = task.envs[env_id]
            handle = task.humanoid_handles[env_id]

            for j in task.obs_mask_utility_indexes[body_subset]:
                task.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return
    
    def speed_to_color(self, speed):
        """
        Converts a speed value (0 to 1) to a color gradient between blue and red.

        :param speed: A float value between 0 and 1.
        :return: A tuple representing the RGB color.
        """
        # Ensure speed is within the range [0, 1]
        speed = max(0, min(speed, 1))

        # Linearly interpolate between blue and red
        red = speed  # Red increases with speed
        blue = 1 - speed  # Blue decreases with speed
        green = 0  # Green always zero in blue-to-red gradient

        # # Example usage
        # speed_example = 0.5  # Speed value example
        # color = speed_to_color(speed_example)
        # print(f"Color for speed {speed_example}: {color}")

        return (red, green, blue)
        
    def pre_render_hook_fxn(self):
        if self._render_demo:
            viz_at_idxs = torch.LongTensor([0]).cuda()
            self._set_env_to_targets(set_at_env_ids = viz_at_idxs, 
                                     target_of_env_ids = torch.LongTensor([1]).cuda(),
                                     anchor_root = self._render_anchored_demo, 
                                     shift_root= self._render_shifted_demo)
            
            char_color_map = {0: (0,0,1),                     # Blue [All Mocap details]
                              1: (192/255, 192/255, 192/255), # Grey (Root obs only)
                              2: (1,1,0),                     # Yellow (Root Obs + Joint Angles)
                              3: (0,0,1),                     # Red (Global joint xyz only) 
                              4: (192/255, 192/255, 192/255), # Grey (Joint Key XYZ only)
                              5: (148/255, 0, 211/255),       # Purple (Root obs + local joint xyz only)
                              6: (192/255, 192/255, 192/255), # Grey (Joint Key XYZ only)
                              7: (192/255, 192/255, 192/255)  # Grey (Joint Key XYZ only)
                              }
            
            # self._set_char_color((0,0,1), torch.arange(1,self.env.num_envs).cuda())
            l = len(self.env.task.lookahead_mask_pool)
            mask_len_ref = torch.sum(self.env.task.lookahead_mask_pool, dim = -1)
            mask_len = torch.sum(self.env.task._global_demo_lookahead_mask, dim = -1)
            for i in range(len(mask_len_ref)):
                ref = mask_len_ref[i]
                try:
                    self._set_char_color(char_color_map[i], torch.arange(self.env.num_envs).cuda()[mask_len==ref])
                except:
                    print("Char Color map, index not found ?, setting default blue")
                    self._set_char_color(char_color_map[0], torch.arange(self.env.num_envs).cuda())

            # Activate if we want to change color after each switch.
            curr_demo_id = self.env.task._global_demo_start_motion_ids[0].item()
            if self._prev_demo_id is None:
                self._prev_demo_id = curr_demo_id
                self._prev_demo_color = (0,1,0)
            if self._prev_demo_id != curr_demo_id:
                new_color = (0.3, 0.6, 0.3) if self._prev_demo_color == (0,1,0) else (0,1,0)
                self._prev_demo_color = new_color
                self._prev_demo_id = curr_demo_id
                
            self._set_char_color(self._prev_demo_color, torch.LongTensor([0]).cuda())

        if self._render_color_for_speed:
            for env_index in range(self.env.num_envs):
                # Extract the x and y components of the speed from the lookahead data
                speed_x = self.env.task._tar_flat_full_state_obs[env_index, 7]
                speed_y = self.env.task._tar_flat_full_state_obs[env_index, 8]
                # Calculate the magnitude of the speed vector
                speed_magnitude = torch.sqrt(speed_x**2 + speed_y**2).item()

                if self._speed_adaptive_color:
                    self._set_char_color(self.speed_to_color((speed_magnitude-1)/2), torch.LongTensor([env_index]).cuda())
                else:
                    self._set_char_color((0,0,1), torch.LongTensor([env_index]).cuda())
                
            if self._render_anchored_demo:
                self._set_char_color((0,1,0), torch.LongTensor([0]).cuda())
                self._set_char_color_for_body_subset(self._masked_out_color, torch.LongTensor([0]).cuda(), body_subset = "lower_rigid_body_indxs")

        if self._render_demo_composition:
            self._set_env_to_targets(set_at_env_ids = torch.LongTensor([0,2]).cuda(), 
                                     target_of_env_ids = torch.LongTensor([0,2]).cuda(),
                                     to_anchor_env_ids= torch.LongTensor([1]).cuda(), # Note that this is the index of set_at env_ids
                                     anchor_at_env_ids = torch.LongTensor([0]).cuda(), 
                                     shift_to_env_id = 1)
            
            char_color_map = {0: (0,0,1), # Blue (All Obs)
                              1: (1,0,0), # Red (Root obs only)
                              2: (1,1,0), # Yellow (Joint XYZ only)
                              3: (1,0.5,0), # Orange (Joint Key XYZ only) 
                              4: (148/255, 0, 211/255), # Purple (Joint Key XYZ only)
                              5: (75/255, 0, 130/255) # Purple (Joint Key XYZ only)
                              }
            # self._set_char_color((0,0,1), torch.arange(1,self.env.num_envs).cuda())
            l = len(self.env.task.lookahead_mask_pool)
            mask_len_ref = torch.sum(self.env.task.lookahead_mask_pool, dim = -1)
            mask_len = torch.sum(self.env.task._global_demo_lookahead_mask, dim = -1)
            for i in range(len(mask_len_ref)):
                ref = mask_len_ref[i]
                self._set_char_color(char_color_map[(i%2)*2], torch.arange(self.env.num_envs).cuda()[mask_len==ref])

            self._set_char_color((0,1,0), torch.LongTensor([0,2]).cuda())
            self._set_char_color_for_body_subset(self._masked_out_color, torch.LongTensor([0]).cuda(), body_subset = "upper_rigid_body_indxs")
            self._set_char_color_for_body_subset(self._masked_out_color, torch.LongTensor([2]).cuda(), body_subset = "lower_rigid_body_indxs")

    def restore_from_wandb(self, wandb_path):
        if wandb_path != "None":
            api = wandb.Api()
            file_name = wandb_path.split("/")[-1]
            run = api.run("/".join(wandb_path.split("/")[:-1]))
            run.file(file_name).download(root="output", replace = True)
            self.restore(f"output/{file_name}")
            print("Restore Succesful")
        else:
            print("No restore path provided")


    def _set_env_to_targets(self, set_at_env_ids, target_of_env_ids, anchor_root = False,
                             to_anchor_env_ids = None,
                             anchor_at_env_ids = None,
                             shift_to_env_id = None,
                             shift_root = False):
        """
        Resets environment state for given env ids and sets the character color to green.
        """


        task = self.env.task

        target_demo_idxs = task._global_demo_start_motion_ids[target_of_env_ids]
        target_demo_time_steps = task.global_demo_start_motion_time_steps[target_of_env_ids]
        target_demo_rotations = task._global_demo_start_rotations[target_of_env_ids]

        progress_steps = task.progress_buf[target_of_env_ids]

        # Calculate time step indices
        q_ts_idxs = ((target_demo_time_steps + progress_steps) % task._motion_max_steps[target_demo_idxs]).view(-1)
        # Calculate final indices
        f_idxs = task._motion_lib.demo_store["length_starts"][target_demo_idxs] + q_ts_idxs
        
        #######################################################################
        tar_demo_root_pos = task._motion_lib.demo_store["root_pos"].transpose(0, 1)[f_idxs].cuda()
        tar_demo_root_rot = task._motion_lib.demo_store["root_rot"].transpose(0, 1)[f_idxs].cuda()
        tar_demo_dof_pos = task._motion_lib.demo_store["dof_pos"].transpose(0, 1)[f_idxs].cuda()
        tar_demo_root_vel = task._motion_lib.demo_store["root_vel"].transpose(0, 1)[f_idxs].cuda()
        tar_demo_root_ang_vel = task._motion_lib.demo_store["root_ang_vel"].transpose(0, 1)[f_idxs].cuda()
        tar_demo_dof_vel = task._motion_lib.demo_store["dof_vel"].transpose(0, 1)[f_idxs].cuda()


        # Rotate the agent rotation to the target rotation padding #############
        tar_demo_root_pos = quat_rotate(target_demo_rotations, tar_demo_root_pos)
        tar_demo_root_rot = quat_mul(target_demo_rotations, tar_demo_root_rot)
        tar_demo_root_vel = quat_rotate(target_demo_rotations, tar_demo_root_vel)
        tar_demo_root_ang_vel = quat_rotate(target_demo_rotations, tar_demo_root_ang_vel)
        #######################################################################

        #### Shift or anchor logic #####
        t_env_id = target_of_env_ids[0].item()
        curr_demo_idx = target_demo_idxs[0].item()
        curr_demo_time_step = q_ts_idxs[0].item()
        if not hasattr(task, "_prev_to_render_mid"): 
            task._prev_to_render_mid = curr_demo_idx
            task._prev_to_render_mstep = curr_demo_time_step
            task.target_fixed_offset_x = 0
            task.target_fixed_offset_y = 0
        
        if task._prev_to_render_mid != curr_demo_idx or task._prev_to_render_mstep != (curr_demo_time_step-1):
            if shift_to_env_id is not None:
                t_env_id = shift_to_env_id
            task.target_fixed_offset_x = task._humanoid_root_states[t_env_id, 0].item() - tar_demo_root_pos[0,0].item()
            task.target_fixed_offset_y = task._humanoid_root_states[t_env_id, 1].item() - tar_demo_root_pos[0,1].item()

        if anchor_root:
            tar_demo_root_pos = task._humanoid_root_states[target_of_env_ids, :3] # 0 is root index
        elif shift_root or shift_to_env_id is not None:
            tar_demo_root_pos[:, 0] += task.target_fixed_offset_x  # Adjust x position
            tar_demo_root_pos[:, 1] += task.target_fixed_offset_y  # Adjust y position
        
        if anchor_at_env_ids is not None and to_anchor_env_ids is not None:
            tar_demo_root_pos[to_anchor_env_ids, :] = task._humanoid_root_states[anchor_at_env_ids, :3] # 0 is root index

        task._prev_to_render_mid = curr_demo_idx
        task._prev_to_render_mstep = curr_demo_time_step

        # Reset environment state for the given env_ids
        task._set_env_state(
            env_ids=set_at_env_ids,
            root_pos=tar_demo_root_pos,
            root_rot=tar_demo_root_rot,
            dof_pos=tar_demo_dof_pos,
            root_vel=tar_demo_root_vel,
            root_ang_vel=tar_demo_root_ang_vel,
            dof_vel=tar_demo_dof_vel
        )

        env_ids_int32 = self.env.task._humanoid_actor_ids[set_at_env_ids]
        self.env.task.gym.set_actor_root_state_tensor_indexed(self.env.task.sim,
                                                        gymtorch.unwrap_tensor(self.env.task._root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.env.task.gym.set_dof_state_tensor_indexed(self.env.task.sim,
                                                gymtorch.unwrap_tensor(self.env.task._dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        # task._reset_env_tensors(set_at_env_ids)
        # task._refresh_sim_tensors()

