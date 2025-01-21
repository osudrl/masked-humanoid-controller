# Imports

# Isaac Gym Imports
from isaacgym import gymapi
from isaacgym.torch_utils import *

# Python Imports
import yaml
import os
import copy
import pickle as pk
import torch
import numpy as np
import wandb
from datetime import datetime
from types import MethodType
from munch import Munch
from tqdm import tqdm
import time 

# AMP Imports
from learning.amp_models import ModelAMPContinuous

# Env Imports
from env.tasks.vec_task_wrappers import VecTaskPythonWrapper
from env.tasks.humanoid_im import ObsBuilder
from env.tasks.humanoid_im import RewardUtils
from env.tasks.humanoid_im import DemoStoreUtils
from env.tasks.humanoid_im import HumanoidAMPRobust
from env.tasks.humanoid_im_getup import HumanoidAMPGetupRobust
from env.tasks.humanoid_im_getup_task_viz import HumanoidAMPGetupRobustTaskViz
from env.tasks.humanoid_im_getup_perturb import HumanoidAMPGetupPerturb

# Learning Imports
from learning.mhc_agent import AMPAgent2
from learning.mhc_models import ModelAMPContinuousv1
from learning.mhc_network_builder import AMPBuilderV1
from learning.mhc_players import AMPPlayer2

# RL Games Imports
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.common import tr_helpers

# Utils Imports
from utils.benchmark_helper import generate_cache_for_env_vars
from utils.benchmark_helper import restore_env_vars_from_cache
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task


# --------------------------------------------
# ---------------House Keeping----------------
# --------------------------------------------
def create_rlgpu_env(args, cfg, cfg_train, **kwargs):
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    
    frames = kwargs.pop('frames', 1)
    # if frames > 1:
    #     env = wrappers.FrameStack(env, frames, False)
    return env

class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return

class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    try:
        print("Creating Task")
        print(args.task)
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        warn_task_name()
    env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    return task, env

vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(args, cfg, cfg_train, **kwargs),
    'vecenv_type': 'RLGPU'})

# --------------------------------------------
# ----- Custom overrides/ Housekeeping--------
# --------------------------------------------

from utils.extract_keypoints import extract_keypoints_from_text, extract_keypoints_from_video, extract_data_from_vr

def setup_base_config(args):
    cfg, cfg_train, logdir = load_cfg(args)
    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    cfg_train['params']['network']['mlp']['units'] = [int(x) for x in args.mlp_units.split(',')]
    cfg_train['params']['network']['mlp']['activation'] = "silu" 
    cfg_train['params']['network']['disc']['units'] = [int(x) for x in args.disc_units.split(',')]
    cfg_train['params']['network']['lk_encoder'] = {}
    cfg_train['params']['network']['lk_encoder']['units'] = [int(x) for x in args.lk_encoder_units.split(',')]
    cfg_train['params']['network']['lk_encoder']['embedding_dim'] = args.lk_embedding_dim

    cfg_train['params']['config']['task_reward_w'] = args.task_reward_w # Done for imit rewards to be activated. 
    cfg_train['params']['config']['disc_reward_w'] = 1-args.task_reward_w # Done for imit rewards to be activated. 
    cfg_train["params"]["config"]['max_epochs'] = int(1000) if args.debug else int(args.max_epochs)
    cfg_train['params']['config']['train_dir'] = args.output_path
    cfg_train['params']['config']['save_frequency'] = 50
    
    
    cfg['env']['motion_file'] = args.motion_file
    cfg["env"]["stateRotate"] = args.init_state_rotate
    cfg["env"]["stateDemoRotate"] = args.demo_rotate
    cfg["env"]["stateInit"] = args.init_state_sampling
    cfg["env"]["demoInit"] = "Random" #HumanoidAMPRobust.StateInit.Random
    cfg["env"]["demoMotionIds"] = args.demo_motion_ids # Motion Ids to mimic
    cfg["env"]["hybridInitProb"] = 0.9 # This is ref state init prob
    cfg["env"]["controlFrequencyInv"] = args.ctrl_freq_int
    cfg["env"]["enableEarlyTermination"] = not args.disable_early_termination
    cfg['env']['envSpacing'] = args.env_spacing
    cfg["env"]["ENERGY_PENALTY_ACTIVATED"] = args.energy_penalty
    cfg["env"]["MOTIONLIB_DEVICE"] = args.motion_lib_device
    cfg["env"]["CACHE_HORIZON"] = 350
    cfg["env"]["heightOffset"] = args.height_offset
    cfg["env"]["penaltyMultiplyer"] = args.penalty_multiplyer
    cfg["env"]["switchDemosWithinEpisode"] = args.switch_demos_within_episode
    if args.key_body_names != "":
        cfg["env"]["keyBodies"] = args.key_body_names.split(',')
    cfg["env"]["enableBodyPosObs"] = args.enable_body_pos_obs
    cfg["env"]["keyposRewardOnly"] = args.keypos_reward_only
    cfg["env"]["motion_is_sampled_by_frames"] = args.motion_is_sampled_by_frames
    cfg["env"]["useOnlyKeyposForAMP"] = args.use_only_keypos_for_amp
    cfg["env"]["random_dir_speed_lookahead_share"] = args.random_dir_speed_lookahead_share
    cfg["env"]["compose_demo_targets_share"] = args.compose_demo_targets_share
    cfg["env"]["enable_lookahead_mask"] = args.enable_lookahead_mask
    cfg["env"]["enable_lk_jl_mask"] = args.enable_lk_jl_mask
    cfg["env"]["enable_lk_channel_mask"] = args.enable_lk_channel_mask
    cfg["env"]["use_predefined_demo_weights_for_sampling"] = args.use_predefined_demo_weights_for_sampling
    cfg["env"]["use_shifted_xyz_reward"] = args.use_shifted_xyz_reward
    cfg["env"]["start_demo_at_agent_state_prob"] = args.start_demo_at_agent_state_prob
    cfg["env"]["lk_embedding_dim"] = args.lk_embedding_dim
    cfg["env"]["keypos_big_weight_reward"] = args.keypos_big_weight_reward
    cfg["env"]["dof_xyz_reward_w"] = args.dof_xyz_reward_w

    cfg["env"]["disable_lk_mask_0"] = args.disable_lk_mask_0
    cfg["env"]["disable_lk_mask_1"] = args.disable_lk_mask_1
    cfg["env"]["disable_lk_mask_2"] = args.disable_lk_mask_2
    cfg["env"]["disable_lk_mask_3"] = args.disable_lk_mask_3
    cfg["env"]["disable_lk_mask_4"] = args.disable_lk_mask_4
    cfg["env"]["disable_lk_mask_5"] = args.disable_lk_mask_5
    cfg["env"]["disable_lk_mask_6"] = args.disable_lk_mask_6

    cfg_train['params']['network']['disable_lookahead_mask_in_obs'] = args.disable_lookahead_mask_in_obs
    cfg_train['params']['network']['disable_multi_mask_amp'] = args.disable_multi_mask_amp

    return cfg, cfg_train, logdir

def get_extra_parameters():
    extra_parameters = [
        {"name": "--wandb_project", "type": str, "default": "catchup_amp_reallusion_v1", "help": "Initialize a new run"},

        {"name": "--imit_from_flat_buffer", "action": "store_true", "default": False, "help": "Switch demos within episode"},
        {"name": "--flat_buffer_path", "type": str, "default": "None", "help": "Path for saving Weights & Biases files"},

        {"name": "--train_run", "action": "store_true", "default": False, "help": "Initialize a new training run"},
        {"name": "--visualize_policy", "action": "store_true", "default": False, "help": "Apply to test environment visualization"},
        {"name": "--benchmark_policy", "action": "store_true", "default": False, "help": "Apply to test environment visualization"},
        {"name": "--visualize_video", "action": "store_true", "default": False, "help": "Apply to load video and visualize"},
        {"name": "--visualize_text2motion", "action": "store_true", "default": False, "help": "Apply to load video and visualize"},
        {"name": "--visualize_vr_motion", "action": "store_true", "default": False, "help": "Apply to load video and visualize"},
        {"name": "--collect_dataset", "action": "store_true", "default": False, "help": "Initialize data collection for a new run"},
        {"name": "--eval_policy", "action": "store_true", "default": False, "help": "Initialize evaluation of a policy"},
        {"name": "--motion_lib_device", "type": str, "default": "cuda:0", "help": "Device to use for motion library operations"},
        
        {"name": "--benchmark_type", "type": str, "default": "mimicry", "help": "Device to use for motion library operations"},
        {"name": "--benchmark_dataset", "type": str, "default": "default", "help": "Device to use for motion library operations"},
        {"name": "--benchmark_player", "type": str, "default": "default", "help": "Device to use for motion library operations"},
        # {"name": "--benchmark_type", "type": str, "default": "mimicry_wplookahead", "help": "Device to use for motion library operations"},

        {"name": "--heading_fsm", "action": "store_true", "default": False, "help": "Heading FSM"},
        {"name": "--location_fsm", "action": "store_true", "default": False, "help": "Heading FSM"}, 
        
        {"name": "--debug", "action": "store_true", "default": False, "help": "Enable debug mode"},
        {"name": "--max_epochs", "type": int, "default": 200000, "help": "Number of epochs to run"},
        {"name": "--xbox_interactive", "action": "store_true", "default": False, "help": "Enable interactive mode for testing"},
        {"name": "--interactive", "action": "store_true", "default": False, "help": "Enable interactive mode for testing"},
        {"name": "--interactive_set", "action": "store_true", "default": False, "help": "Enable interactive mode for testing"},
        {"name": "--env_spacing", "type": float, "default": 5, "help": "Weight for task reward"},
        {"name": "--dynamic_weighting", "action": "store_true", "default": False, "help": "Apply to test environment dynamic weighting"},
        {"name": "--dynamic_weighting_start_epoch", "type": int, "default": 5000, "help": "Start dynamic weighting after this epoch"},
        {"name": "--motion_is_sampled_by_frames", "action": "store_true", "default": False, "help": "Sample motions at frame level"},
        {"name": "--disable_lookahead_mask_in_obs", "action": "store_true", "default": False, "help": "Sample motions at frame level"},

        {"name": "--disable_lk_mask_0", "action": "store_true", "default": False, "help": "Deactivate samling of lookahead 0"},
        {"name": "--disable_lk_mask_1", "action": "store_true", "default": False, "help": "Deactivate samling of lookahead 1"},
        {"name": "--disable_lk_mask_2", "action": "store_true", "default": False, "help": "Deactivate samling of lookahead 2"},
        {"name": "--disable_lk_mask_3", "action": "store_true", "default": False, "help": "Deactivate samling of lookahead 3"},
        {"name": "--disable_lk_mask_4", "action": "store_true", "default": False, "help": "Deactivate samling of lookahead 4"},
        {"name": "--disable_lk_mask_5", "action": "store_true", "default": False, "help": "Deactivate samling of lookahead 5"},
        {"name": "--disable_lk_mask_6", "action": "store_true", "default": False, "help": "Deactivate samling of lookahead 6"},

        {"name": "--demo_motion_ids", "type": str, "default": "1", "help": "Motion IDs to imitate for demo"},
        {"name": "--mlp_units", "type": str, "default": "1024,512", "help": "MLP units for model initialization"},
        {"name": "--disc_units", "type": str, "default": "1024,512", "help": "Discriminator units for model initialization"},
        {"name": "--lk_encoder_units", "type": str, "default": "1024,1024", "help": "MLP units for model initialization"},
        {"name": "--lk_embedding_dim", "type": int, "default": 64, "help": "lookahead embedding dim"},
        {"name": "--wandb_path", "type": str, "default": "None", "help": "Path for saving Weights & Biases files"},
        {"name": "--task_reward_w", "type": float, "default": 0.5, "help": "Weight for task reward"},
        {"name": "--dof_xyz_reward_w", "type": int, "default": 25, "help": "Weight for task reward"},
        
        {"name": "--ctrl_freq_int", "type": int, "default": 2, "help": "Control frequency interval"},
        {"name": "--energy_penalty", "action": "store_true", "default": False, "help": "Enable energy penalty"},
        {"name": "--disable_early_termination", "action": "store_true", "default": False, "help": "Disable early termination"},
        {"name": "--enable_body_pos_obs", "action": "store_true", "default": False, "help": "Switch demos within episode"},
        {"name": "--enable_lookahead_mask", "action": "store_true", "default": False, "help": "Switch demos within episode"},
        {"name": "--enable_lk_channel_mask", "action": "store_true", "default": False, "help": "Enable lookahead channel mask"},
        {"name": "--enable_lk_jl_mask", "action": "store_true", "default": False, "help": "Enable lookahead joint random mask"},
        {"name": "--keypos_reward_only", "action": "store_true", "default": False, "help": "Track only absolute key body positions"},
        {"name": "--keypos_big_weight_reward", "action": "store_true", "default": False, "help": "Track only absolute key body positions with higher weights"},

        {"name": "--height_offset", "type": float, "default": 0, "help": "Dimension of AMP observations"},
        {"name": "--random_dir_speed_lookahead_share", "type": float, "default": 0, "help": "Dimension of AMP observations"},
        {"name": "--compose_demo_targets_share", "type": float, "default": 0, "help": "Perecentage of share for composed demo targets"},
        {"name": "--amp_obs_dim", "type": int, "default": 153, "help": "Dimension of AMP observations"},
        {"name": "--keypos_obs_dim", "type": int, "default": 17*3, "help": "Dimension of keypose observations"},
        {"name": "--penalty_multiplyer", "type": int, "default": 1, "help": "Dimension of keypose observations"},
        {"name": "--init_state_sampling", "type": str, "default": "Hybrid", "help": "Motion IDs to imitate for demo"},
        {"name": "--init_state_rotate", "action": "store_true", "default": False, "help": "Switch demos within episode"},
        {"name": "--demo_rotate", "action": "store_true", "default": False, "help": "Switch demos within episode"},
        {"name": "--switch_demos_within_episode", "action": "store_true", "default": False, "help": "Switch demos within episode"},
        {"name": "--key_body_names","type": str, "default":"", "help": "Path for saving Weights & Biases files"},
        {"name": "--use_only_keypos_for_amp", "action": "store_true", "default": False, "help": "Switch demos within episode"},
        {"name": "--use_predefined_demo_weights_for_sampling", "action": "store_true", "default": False, "help": "Activate pre defined weights for sampling"},
        {"name": "--honor_mask_for_rewards_at_benchmark", "action": "store_true", "default": False, "help": "Activate pre defined weights for sampling"},

        {"name": "--start_demo_at_agent_state_prob", "type": float, "default": 0.5, "help": "Dimension of AMP observations"},

        {"name": "--use_shifted_xyz_reward", "action": "store_true", "default": False, "help": "Activate pre defined weights for sampling"},
        {"name": "--disable_multi_mask_amp", "action": "store_true", "default": False, "help": "Activate pre defined weights for sampling"},
   ]
      

    return extra_parameters


# python ase/train_catchup_amp_llc_v5.py --visualize_policy --num_envs 2 --wandb_path dacmdp/catchup_amp_reallusion_llc_v4_debug/4qrihd7h/latest_model.pth --env_spacing 1.5 --demo_motion_ids 9999 --mlp_units 1024,512 --disc_units 1024,512 --lk_encoder_units 1024,1024 --lk_embedding_dim 64 --max_epochs 200000 --disable_early_termination --motion_lib_device cpu --cfg_env ase/data/cfg/humanoid_ase_sword_shield_catchup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid_catchup.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --wandb_project catchup_amp_reallusion_llc_stable_v2_debug --energy_penalty --penalty_multiplyer 2 --switch_demos_within_episode --init_state_rotate --demo_rotate --honor_mask_for_rewards_at_benchmark

def custom_update_camera(self):
    self.to_track_env_id = 1 if self.num_envs >1  else 0
    self.gym.refresh_actor_root_state_tensor(self.sim)
    char_root_pos = self._humanoid_root_states[self.to_track_env_id, 0:3].cpu().numpy()
    char_root_pos[0] = char_root_pos[0] + 3

    cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
    cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
    cam_delta = cam_pos - self._cam_prev_char_pos

    new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
    new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                            char_root_pos[1] + cam_delta[1], 
                            cam_pos[2])

    self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

    self._cam_prev_char_pos[:] = char_root_pos
    return

def change_color_based_on_speed(player):
    for env_index in range(player.env.num_envs):
        # Extract the x and y components of the speed from the lookahead data
        speed_x = player.env.task._tar_flat_full_state_obs[env_index, 7]
        speed_y = player.env.task._tar_flat_full_state_obs[env_index, 8]
        # Calculate the magnitude of the speed vector
        speed_magnitude = torch.sqrt(speed_x**2 + speed_y**2).item()

        player._set_char_color(player.speed_to_color((speed_magnitude-1)/2), torch.LongTensor([env_index]).cuda())


def custom_render_demo(player):
    self = player
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


def read_joystick_commands(player):

    self = player.env.task
    all_env_ids = player.env.task._all_env_ids

    # Speed, Target Heading and Direction
    # --------------------------------------------------------------------------
    # Use Joystick X and Y to set Target Heading and Direction
    agent_height = 0.85 - 0.5*joy.LeftTrigger
    agent_heading_dir = torch.FloatTensor([[joy.LeftJoystickX + 1e-6, joy.LeftJoystickY]]).cuda()
    agent_vel_dir = torch.FloatTensor([[joy.RightJoystickX + 1e-6, joy.RightJoystickY]]).cuda()
    agent_heading_dir = agent_heading_dir/torch.norm(agent_heading_dir, dim = -1).unsqueeze(-1)
    agent_heading_theta = torch.atan2(agent_heading_dir[:,1], agent_heading_dir[:,0])
    
    # Setup the commands in the lookahead directly
    lookahead = torch.zeros(player.env.task.num_envs, player.env.task._lookahead_obs_dim).cuda()
    lookahead[:,2] = agent_height # Fixed Height
    lookahead[:,3:7] = quat_from_angle_axis(angle = agent_heading_theta.cuda(),
                                                axis = torch.FloatTensor([[0,0,1]]).cuda()) 
    lookahead[:,7:9] = agent_vel_dir / torch.norm(agent_vel_dir, dim = -1).unsqueeze(-1)
    player.env.task._global_random_dir_speed_lookahead_bucket[:] = lookahead
    # -------------------------------------------------------------------------------------    

    
    # Upper Body 
    # ----------------------------------------------------------------------------------------------
    # seed the motion id if not already set
    # Increment or decrement the motion id based on the joystick buttons
    self.imit_joystick_motion_id = getattr(self, "imit_joystick_motion_id", 0)
    if joy.Y:
        self.imit_joystick_motion_id += 1
        self._global_demo_start_rotations[all_env_ids] = self._sample_random_rotation_quat_for_demo(len(all_env_ids))
    if joy.A:
        self.imit_joystick_motion_id -= 1
        self._global_demo_start_rotations[all_env_ids] = self._sample_random_rotation_quat_for_demo(len(all_env_ids))
    if joy.B:
        self.progress_buf[:] = 9999999
    self.imit_joystick_motion_id = self.imit_joystick_motion_id % self._motion_lib.num_motions()


    # Motion Sets
    # ----------------------------------------------------------------------------------------------
    leftBumperMotionSet = [2, 5, 24, 26, 30]    # Motion Set 0 
    rightBumperMotionSet = [17, 15, 16, 18, 34]  # Motion Set 1
    defaultMotionSet = [65, 69, 70, 73, 68]  # Motion Set 2 for default case
    # -----


    # Normal Xbox Setup
    # --------------------------------------------------------------------------------------------------------------
    player._render_demo = False
    self.show_traj = False
    self.show_root_traj = False
    self._random_dir_speed_env_idxs = all_env_ids
    self._random_dir_speed_upper_body_env_idxs = all_env_ids
    self._global_demo_lookahead_mask[all_env_ids] = self.lookahead_mask_pool[self._upper_body_conditioning_compatible_mask_pool_indx]
    # --------------------------------------------------------------------------------------------------------------

    # Calculate index based on the right trigger's value
    # Assume rightTrigger is a value between 0 and 1
    # Determine which motion set to use based on bumper state
    index = int(5 * joy.RightTrigger)
    
    mode = f"{int(joy.RightBumper)}{int(joy.LeftBumper)}"
    motion_set = None
    if mode == "01":
        self._global_demo_start_motion_ids[:] = rightBumperMotionSet[index]
    elif mode == "10":
        self._global_demo_start_motion_ids[:] = leftBumperMotionSet[index]
    elif mode == "00":  # Default case when neither or both bumpers are pressed
        self._global_demo_start_motion_ids[:] = defaultMotionSet[index]
    elif mode == "11":
        # Imit Setup
        # --------------------------------------------------------------------------------------------------------------
        self._random_dir_speed_env_idxs = torch.arange(0).type(torch.LongTensor).cuda()
        self._random_dir_speed_upper_body_env_idxs = torch.arange(0).type(torch.LongTensor).cuda()
        player._render_demo = True
        player._render_shifted_demo = True
        self.show_traj = False
        self.show_root_traj = False
        self._global_demo_start_motion_ids[:] = self.imit_joystick_motion_id
        self._global_demo_lookahead_mask[all_env_ids] = self.lookahead_mask_pool[0]
        # --------------------------------------------------------------------------------------------------------------

    # Enable Perturbation
    # --------------------------------------------------------------------------------------------------------------    
    if joy.X:
        # print("Joystick X pressed", self.progress_buf[0], self._last_toggle_timestep)
        if abs(self.progress_buf[0] - self._last_toggle_timestep) > 50:
            self._enable_perturbation = not self._enable_perturbation
            self._last_toggle_timestep = self.progress_buf[0].clone()
    # --------------------------------------------------------------------------------------------------------------


    change_color_based_on_speed(player)
    custom_render_demo(player)


    print("Dir: {:6.2f} {:6.2f}, Vel:{:6.2f} {:6.2f} Height:{:6.2f} Upper Motion Id:{:6.2f} Motion Set:{:6.2f} {:6.2f}  Perturb: {:6.2f}, Imit Motion: {:6.2f}".format(
            joy.LeftJoystickX,
            joy.LeftJoystickY,
            joy.RightJoystickX,
            joy.RightJoystickY,
            joy.LeftTrigger,
            int(10 * joy.RightTrigger / 2),
            joy.RightBumper,
            joy.LeftBumper,
            self._enable_perturbation,
            self.imit_joystick_motion_id)
        , end="\r")
            

if __name__ == "__main__":
    # Get Arguments
    set_np_formatting()
    extra_parameters = get_extra_parameters()
    args = get_args(extra_parameters=extra_parameters)
    args.demo_motion_ids = [int(x) for x in args.demo_motion_ids.split(',')]
    args.task = 'HumanoidAMPGetupRobustTaskViz'
        
    args.seed = 5940
    args.checkpoint = ''

    if len(args.demo_motion_ids) == 1 and args.demo_motion_ids[0] == 9999:
        num_motions = len(yaml.safe_load(open(args.motion_file, 'r'))['motions'])
        args.demo_motion_ids = list(range(num_motions))

    cfg, cfg_train, logdir = setup_base_config(args)
    cfg["env"]["CACHE_HORIZON"] = 3005

    # rl_games expecting config later
    config = copy.deepcopy(cfg_train['params']['config'])

    # Setup Player
    network = AMPBuilderV1()
    network.load(cfg_train['params']['network'])
    config['network'] = ModelAMPContinuous(network)
    player = AMPPlayer2(config)
    player.env.task._update_camera = MethodType(custom_update_camera, player.env.task)

    # Restore from wandb
    run = wandb.init(entity = "dacmdp",
                    project="amp_debug",
                    name = "VisualizeRun" + datetime.now().strftime("_%y%m%d-%H%M%S"))
    player.restore_from_wandb(args.wandb_path)

    # Seeding Run Variables
    player.games_num= 1
    player.env.task.max_episode_length = 10
    player.env.reset(torch.arange(player.env.task.num_envs).cuda())
    player.run()


    render_demo=True
    render_target_markers = True
    
    
    from utils.xbone import XboxController
    joy = XboxController()

    def custom_pre_render_hook_fxn(self):
        read_joystick_commands(player)
        
    player.env.task.pre_render_hook_fxn = MethodType(custom_pre_render_hook_fxn, player.env.task)

    # Reset ############################################################################################################
    player.env.task._state_init = HumanoidAMPRobust.StateInit.Start
    player.env.task._demo_init = HumanoidAMPRobust.StateInit.Start
    player.env.task._start_state_motion_weights[:] = 0
    player.env.task._start_state_motion_weights[65] = 1
    player.env.task._demo_motion_weights[:] = 0
    player.env.task._demo_motion_weights[65] = 1
    # self.env.task._fixed_demo_lookahead_mask_env2idxmap_flag = True if enable_lookahead_mask else False
    # ################################################################################

    # Make all env do random dir
    player.env.task._random_dir_speed_lookahead_share = 1 #cfg["env"]["random_dir_speed_lookahead_share"]
    player.env.task._demo_env_offset = player.env.task._init_random_dir_env_indexes(start_at = 0)
    player.env.task._random_dir_speed_mimic_upper_prob = 1
    player.env.task._init_random_dir_upper_env_indexes()
    player._compose_demo_targets_share = 0
    player.env.task._init_compose_env_indexes(start_at = player.env.task._demo_env_offset)
    player.env.task._random_dir_speed_env_idxs = torch.arange(player.num_envs).type(torch.LongTensor).cuda()
    player.env.task._random_dir_speed_upper_body_env_idxs = torch.arange(player.num_envs).type(torch.LongTensor).cuda()
    
    # Reset Tensors
    player.env.task._state_init_rotate = False
    player.env.task._state_demo_rotate = True
    player.env.task._fall_init_prob = 0
    player.env.task._start_demo_at_agent_state_prob = 0
    player.env.task._uniform_targets = True
    player.env.task._switch_demos_within_episode = False
    player.env.task._enable_lookahead_mask = True
    player.env.task.lookahead_mask_pool_sample_weights[:] = 0
    player.env.task.lookahead_mask_pool_sample_weights[0] = 1 # First one is the default mask full information
    player.env.task._switch_demos_within_episode = False

    # Task Specific
    player.env.task.show_traj = False
    player.env.task.show_headings = True
    player.env.task.show_root_traj = False
    player._render_color_for_speed = True # Activates logic for random speed. 


    player.env_reset(torch.arange(player.env.task.num_envs).cuda())
    ###################################################################################################################
    
    for _ in range(10000):
        player.env.task.max_episode_length = 5000
        player.games_num = 1
        player.run()