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

def load_humanoid_keypoint_data_to_sword_shield_demo_store(player, keypoint_data_dict):
    """
    player: AMP player
    video_keypos: Dict(Tensors) each element is of dimensions {"motion_name": Tensor([num_frames x num_bodies x 3]}
    num_bodies = 24 for amass

    the list will be added sequentially so the first element will be demo 0, second element will be demo 1, etc.
    """

    motion_lib = player.env.task._motion_lib
    task = player.env.task

    for demo_id, (name, keypoint_data) in enumerate(keypoint_data_dict.items()):
        print("Warning: This will override the demo_id: ", demo_id, " in the demo_store. And it may also affect other demo ids.")
        print("Overiding demo_id: ", demo_id, " with name: ", name, " with video_keypos of shape: ", keypoint_data.shape)

        print(keypoint_data.shape)
        keypoint_data = torch.FloatTensor(keypoint_data)

        motion_lib.demo_store["demo_names"][demo_id] = name

        if demo_id == 0:
            motion_lib.demo_store["length_starts"][demo_id] = 0
        else:
            motion_lib.demo_store["length_starts"][demo_id] = motion_lib.demo_store["length_starts"][demo_id-1] + task._motion_max_steps[demo_id-1]
        
        len_of_video = keypoint_data.shape[0]
        keypos_obs_dim =  motion_lib.demo_store["keypos_obs"].shape[0]
        fill_start_idx = motion_lib.demo_store["length_starts"][demo_id]
        fill_end_idx = fill_start_idx + len_of_video

        keypoint_data_flat = keypoint_data.view(-1, keypos_obs_dim).transpose(0,1)
        keypos_indxs = task._full_state_split_indxs["dof_xyz"]

        motion_lib.demo_store["full_state_obs"][keypos_indxs, fill_start_idx:fill_end_idx] = keypoint_data_flat

        task._motion_max_steps[demo_id] = len_of_video
        
        print("Demo Id {} has {} frames".format(demo_id, task._motion_max_steps[demo_id]))
        print("Length Starts:", motion_lib.demo_store["length_starts"][demo_id])
        print("Max steps:",task._motion_max_steps[demo_id])

def visualize_keypoint_dict(keypoint_dict, mask, player, motion_id, collect_dataset = False):
    cached_vars = generate_cache_for_env_vars(player.env.task)

    load_humanoid_keypoint_data_to_sword_shield_demo_store(player, keypoint_dict)

    # import time
    # time.sleep(5)
    for _ in range(1):
        i = motion_id 
        print(list(keypoint_dict.keys())[i])
        demo_idx_pool = [i]

        player._render_demo = False
        player._render_target_markers = True

        # Reset ############################################################################################################
        player.env.task._demo_motion_weights[:] = 0
        player.env.task._demo_motion_weights[demo_idx_pool] = 1
        player.env.task._demo_motion_weights = player.env.task._demo_motion_weights / torch.sum(player.env.task._demo_motion_weights)
        # player.env.task._fixed_demo_lookahead_mask_env2idxmap_flag = True if enable_lookahead_mask else False
        # ################################################################################

        # Reset Tensors
        player.env.task._random_dir_speed_lookahead_share = 0 #cfg["env"]["random_dir_speed_lookahead_share"]

        player.env.task._demo_env_offset = player.env.task._init_random_dir_env_indexes(start_at = 0)
        player.env.task._random_dir_speed_mimic_upper_prob = 0
        player.env.task._init_random_dir_upper_env_indexes()
        player._compose_demo_targets_share = 0
        player.env.task._init_compose_env_indexes(start_at = player.env.task._demo_env_offset)


        player.env.task._start_state_motion_weights[:] = 0
        player.env.task._start_state_motion_weights[demo_idx_pool] = 1/len(demo_idx_pool)
        player.env.task._state_init = HumanoidAMPRobust.StateInit.Start
        player.env.task._demo_init = HumanoidAMPRobust.StateInit.Start
        player.env.task._state_init_rotate = False
        player.env.task._state_demo_rotate = False
        player.env.task._fall_init_prob = 0
        player.env.task._start_demo_at_agent_state_prob = 0
        player.env.task._uniform_targets = True
        player.env.task._switch_demos_within_episode = False

        player.env.task._random_dir_speed_env_idxs = torch.zeros([0]).type(torch.LongTensor).cuda()
        player.env.task.show_traj = False
        player.env.task.show_root_traj = True
        player.env.task.show_headings = False

        player._render_color_for_speed = False

        player.env.task._enable_lk_jl_mask = False
        player.env.task._jl_mask_prob = 0
        player.env.task._enable_lookahead_mask = True
        # player.env.task.lookahead_mask_pool_sample_weights[:] = 0 
        # player.env.task.lookahead_mask_pool_sample_weights[3] = 1 # Global Lookahead
        player.env.task._use_predefined_lookahead_mask = True
        player.env.task._predefined_lookahead_mask = torch.ones(player.env.task._lookahead_obs_dim).type(torch.bool).cuda()
        player.env.task._predefined_lookahead_mask[-51:] = mask.repeat_interleave(3)

        player.env_reset(torch.arange(player.env.task.num_envs).cuda())
        ###################################################################################################################

        player.env.task.max_episode_length = torch.max(player.env.task._motion_max_steps[demo_idx_pool]) - 10
        player.games_num = 1
        if collect_dataset:
            buffer = player.collect_dataset(player.env.task.max_episode_length)
        else:
            player.run()
            buffer = None
    restore_env_vars_from_cache(player.env.task, cached_vars)

    return buffer


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

if __name__ == "__main__":
    # Get Arguments
    set_np_formatting()
    extra_parameters = get_extra_parameters()
    args = get_args(extra_parameters=extra_parameters)
    # ENERGY_PENALTY_ACTIVATED = args.energy_penalty
    # MOTIONLIB_DEVICE = args.motion_lib_device
    args.demo_motion_ids = [int(x) for x in args.demo_motion_ids.split(',')]
    
    if args.train_run:
        args.task = 'HumanoidAMPGetupRobust'
    else:
        args.task = 'HumanoidAMPGetupRobustTaskViz'
    #     # args.task = 'HumanoidAMPGetupPerturb'
        
    # args.cfg_env = 'ase/data/cfg/humanoid_ase_sword_shield_getup.yaml'
    # args.cfg_train = 'ase/data/cfg/train/rlg/amp_humanoid_catchup.yaml'
    # args.motion_file = 'ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml'
    args.seed = 5940
    # args.test = True if args.visualize_policy or args.collect_dataset else False
    args.checkpoint = ''

    if len(args.demo_motion_ids) == 1 and args.demo_motion_ids[0] == 9999:
        num_motions = len(yaml.safe_load(open(args.motion_file, 'r'))['motions'])
        args.demo_motion_ids = list(range(num_motions))

    cfg, cfg_train, logdir = setup_base_config(args)
    # How To Run:
    # Reallusion Dataset Training
    
    # python ase/train_catchup_amp_llc_v5.py --train_run --headless --demo_motion_ids 9999 --num_envs 4096 --mlp_units 1024,512 --disc_units 1024,512 --lk_encoder_units 1024,1024 --lk_embedding_dim 64 --max_epochs 500000 --disable_early_termination --energy_penalty --motion_lib_device cpu --cfg_env ase/data/cfg/humanoid_ase_sword_shield_catchup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid_catchup.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --wandb_project catchup_amp_reallusion_llc_v4_debug --benchmark_policy --dof_xyz_reward_w 60 --keypos_big_weight_reward --init_state_rotate --demo_rotate --switch_demos_within_episode --enable_lookahead_mask --enable_lk_channel_mask --enable_lk_jl_mask --random_dir_speed_lookahead_share 0.25 --compose_demo_targets_share 0.25


    if args.train_run:
        # --------------------------------------------
        # ------------------train---------------------
        # --------------------------------------------
        config = copy.deepcopy(cfg_train['params']['config'])

        if len(args.demo_motion_ids) > 1:
            motion_name = f"Train_MIdSetOf_{len(args.demo_motion_ids)}"
        else:
            motion_name = f"Train_MId_{args.demo_motion_ids[0]}"
        args.experiment = 'AMP_' + motion_name + datetime.now().strftime("_%y%m%d-%H%M%S")

        wandb.init(
            # entity = 'intellabs-eai-aal-motion',
            entity = 'calebpersonal',
            # set the wandb project where this run will be logged
            project = args.wandb_project if not args.debug else "catchup_amp_debug",
            name = args.experiment,
            
            # track hyperparameters and run metadata
            config={
            "info": "v1: disc sn",
            "motion": motion_name,
            "seed": cfg_train['params']['seed'],
            "numAMPObsSteps": cfg['env']['numAMPObsSteps'],
            "ac": cfg_train['params']['network']['mlp']['units'],
            "disc": cfg_train['params']['network']['disc']['units'],
            "lr_rl": config['learning_rate'],
            "lr_disc": config['learning_rate'],
            "mini_epochs": config['mini_epochs'],
            "disc_gp": config['disc_grad_penalty'],
            "disc_rew_scale": config['disc_reward_scale'],
            "task_reward_w": config['task_reward_w'],
            "disc_reward_w": config['disc_reward_w'],
            "MotionFile": args.motion_file.split("/")[-1].split(".")[0], 
            "MotionIds": args.demo_motion_ids,
            "NumEnvs":args.num_envs,
            "robust_start": True,
            "controlFrequencyInv": cfg["env"]["controlFrequencyInv"],
            "enableEarlyTermination": cfg["env"]["enableEarlyTermination"],
            "StateInit": cfg["env"]["stateInit"],
            "robsut_rotation":cfg["env"]["stateRotate"],
            "cfg_train_dump":cfg_train,
            "cfg_dump": cfg,
            "height_offset": args.height_offset
            },
            sync_tensorboard=True,
        )
        print("Saving Python File to Wandb")
        wandb.save("ase/train_catchup_amp_llc_v5.py", base_path = "ase", policy = "now")

        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        network = AMPBuilderV1()
        network.load(cfg_train['params']['network'])
        config['network'] = ModelAMPContinuousv1(network)
        config['features'] = {}
        config['features']['observer'] = RLGPUAlgoObserver()
        config['full_experiment_name'] = args.experiment

        agent = AMPAgent2(base_name='run', config=config)
        agent.restore_from_wandb(args.wandb_path)
        agent.benchmark_policy_flag = args.benchmark_policy

        # eval_agent(agent, 0)
        # agent.save = MethodType(custom_save, agent)
        agent._dynamic_weighting = args.dynamic_weighting
        agent._dynamic_weighting_start_epoch = args.dynamic_weighting_start_epoch

        print("Device Motionlib", agent.vec_env.env.task._motion_lib._device)
        print("Device demo_store", agent.vec_env.env.task._motion_lib.demo_store["amp_obs"].device)

        agent.train()

        artifacts_model = wandb.Artifact(name=motion_name, type='model')
        artifacts_model.add_file(os.path.join(agent.nn_dir, agent.config['name']+'.pth'))
        wandb.log_artifact(artifacts_model)
        wandb.finish()
        # --------------------------------------------
        # --------------------------------------------

    if args.visualize_policy or args.benchmark_policy:
        from munch import Munch
        from tqdm import tqdm
        import time

        cfg["env"]["CACHE_HORIZON"] = 3005

        # rl_games expecting config later
        config = copy.deepcopy(cfg_train['params']['config'])
        # config = copy.deepcopy(cfg_train['params'])

        network = AMPBuilderV1()
        network.load(cfg_train['params']['network'])
        config['network'] = ModelAMPContinuous(network)

        # Dummy Setup
        player = AMPPlayer2(config)
        # player.restore(args.checkpoint)

        run = wandb.init(entity = "calebpersonal",
                        project="amp_debug",
                        name = "VisualizeRun" + datetime.now().strftime("_%y%m%d-%H%M%S"))
        player.restore_from_wandb(args.wandb_path)

        def _update_camera(self):
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
        player.env.task._update_camera = MethodType(_update_camera, player.env.task)

        # Seeding Run Variables
        player.env.task.max_episode_length = 10
        player.games_num= 1
        player.env.reset(torch.arange(player.env.task.num_envs).cuda())
        player.run()

        # # Replace Demo Store
        # Activate if visualizing ASE Mimicry
        if args.benchmark_dataset == "default":
            pass
        else:
            backup_store = copy.deepcopy(player.env.task._motion_lib.demo_store)
            tmp_demo_store = pk.load(open("saved_models/data/ase_demo_store_87_eplen300.pkl", "rb"))
            player.env.task._motion_lib.demo_store = tmp_demo_store.demo_store
            player.env.task._motion_max_steps[:] = 280
            print("Demo Store Loaded from pre defined file")

        if args.benchmark_player == "default":
            pass
        elif args.benchmark_player == "ase":
            from rl_games.common import env_configurations
            from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
            import copy

            from learning.ase_players import ASEPlayer
            from learning.ase_models import ModelASEContinuous
            from learning.ase_network_builder import ASEBuilder
            from utils.parse_task import parse_task
            class ASEPlayerNoEnv(ASEPlayer):
                def __init__(self, config, env):
                    self.env = env
                    env_info = env_configurations.get_env_info(self.env)
                    self.env_info = copy.deepcopy(env_info)
                    config['env_info'] = copy.deepcopy(env_info)
                    super().__init__(config)
                    return
                
                def _build_net_config(self):
                    self.obs_shape = ( self.obs_shape[0] - 70, * self.obs_shape[1:])
                    config = super()._build_net_config()
                    return config

                def get_action(self, obs_dict, amp_lookahead, is_determenistic=False):
                    self._ase_latents = self._eval_enc(amp_lookahead)
                    obs_dict["obs"] = obs_dict["obs"][:, 70:]
                    return super().get_action(obs_dict)

                def _update_latents(self):
                    return
            
            # ase_args = get_args()
            import pickle as pk 
            ase_args = pk.load(open('ase/data/models/ase_player_default_args.pk', 'rb'))
            ase_args.task = args.task
            ase_args.num_envs = args.num_envs
            ase_args.cfg_env = 'ase/data/cfg/humanoid_ase_sword_shield_getup.yaml'
            ase_args.cfg_train = 'ase/data/cfg/train/rlg/ase_humanoid.yaml'
            ase_args.motion_file = 'ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml'
            ase_args.checkpoint = 'ase/data/models/ase_llc_reallusion_sword_shield.pth'
            # args.headless = True
            ase_args.seed = 42

            cfg, cfg_train, logdir = load_cfg(ase_args)
            cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))
            cfg['env']['motion_file'] = ase_args.motion_file
            cfg['env']['envSpacing'] = 2
            cfg['env']['cameraFollow'] = False
            cfg['env']['stateInit'] = "Start"
            # cfg['env']['episodeLength'] = 1000

            config = copy.deepcopy(cfg_train['params']['config'])
            network = ASEBuilder()
            network.load(cfg_train['params']['network'])
            config['network'] = ModelASEContinuous(network)

            ase_player = ASEPlayerNoEnv(config, player.env)
            ase_player.restore(ase_args.checkpoint)
            
            def ase_get_action(self, obs_dict, is_determenistic=False):
                # Calculate Amp Lookahead
                task = self.env.task
                full_lookahead = task._tar_flat_full_state_obs[:].view(-1, task._full_state_obs_dim)
                amp_lookahead = full_lookahead[:, task._full_state_amp_obs_indxs].view(task.num_envs, -1)
                return ase_player.get_action(obs_dict = obs_dict, 
                                            amp_lookahead = amp_lookahead, 
                                            is_determenistic = is_determenistic)
                                             

            player.get_action = MethodType(ase_get_action, player)
        else:
            assert False, "Invalid Benchmark Player"

    if args.visualize_policy:
        render_demo=True
        render_target_markers = True
        
        if args.visualize_video:
            base_dir = 'ase/experimental/pan/visualize_video_motion/motions'
            keypoint_dict = {}
            motion_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".npy")]
            for motion_file in motion_files:
                    keypoint_dict[motion_file], mask = extract_keypoints_from_video(motion_file_path, source_fps = 60)
                    print(motion_file, keypoint_dict[motion_file].shape, type(keypoint_dict[motion_file]))
            visualize_keypoint_dict(keypoint_dict, mask, player)

        if args.visualize_text2motion:
            base_dir = 'ase/experimental/pan/t2m_gpt'
            keypoint_dict = {}
            motion_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".npy")]
            for motion_file in motion_files:
                keypoint_dict[motion_file], mask = extract_keypoints_from_text(motion_file_path)
                print(motion_file, keypoint_dict[motion_file].shape, type(keypoint_dict[motion_file]))
            visualize_keypoint_dict(keypoint_dict, mask, player)
        
        if args.visualize_vr_motion:
            vr_motion_file_name = "ase/experimental/pan/vr/data1.txt"
            keypoint_dict = {}
            keypoint_dict[vr_motion_file_name], mask, extra = extract_data_from_vr(vr_motion_file_name, root_shift = 0.6, global_shift_down = 0.05)
            visualize_keypoint_dict(keypoint_dict, mask, player)

        if args.interactive_set:
            interactive_set_library = {

                "compose_easy":{ "set1": {
                                                        "Motion Id(s)": [70, 73],
                                                        "Exploration Noise Std": 0,
                                                        "Random Start States (0 or 1)": False,
                                                        "Number of rounds": 1,
                                                        "Episode Length (0 = motion length)": 350,
                                                    },
                                                    "set2": {
                                                        "Motion Id(s)": [70, 49],
                                                        "Exploration Noise Std": 0,
                                                        "Random Start States (0 or 1)": False,
                                                        "Number of rounds": 1,
                                                        "Episode Length (0 = motion length)": 0,
                                                    },
                                                    "set3": {
                                                        "Motion Id(s)": [68, 49],
                                                        "Exploration Noise Std": 0,
                                                        "Random Start States (0 or 1)": False,
                                                        "Number of rounds": 1,
                                                        "Episode Length (0 = motion length)": 0,
                                                    },
                                                    "set4": {
                                                        "Motion Id(s)": [17, 73],
                                                        "Exploration Noise Std": 0,
                                                        "Random Start States (0 or 1)": False,
                                                        "Number of rounds": 1,
                                                        "Episode Length (0 = motion length)": 250,
                                                    },
                                                    "set5": {
                                                        "Motion Id(s)": [17, 38],
                                                        "Exploration Noise Std": 0,
                                                        "Random Start States (0 or 1)": False,
                                                        "Number of rounds": 1,
                                                        "Episode Length (0 = motion length)": 0,
                                                    },
                                                    "set6": {
                                                        "Motion Id(s)": [28, 56],
                                                        "Exploration Noise Std": 0,
                                                        "Random Start States (0 or 1)": False,
                                                        "Number of rounds": 1,
                                                        "Episode Length (0 = motion length)": 300,
                                                    },
                                                    "set7": {
                                                        "Motion Id(s)": [69, 54],
                                                        "Exploration Noise Std": 0,
                                                        "Random Start States (0 or 1)": False,
                                                        "Number of rounds": 1,
                                                        "Episode Length (0 = motion length)": 300,
                                                    },
                                                    "set8": {
                                                            "Motion Id(s)": [68, 54],
                                                            "Exploration Noise Std": 0,
                                                            "Random Start States (0 or 1)": False,
                                                            "Number of rounds": 1,
                                                            "Episode Length (0 = motion length)": 300,
                                                        },
                                                },
                "compose_hard": {"set8": {
                                        "Motion Id(s)": [68, 30],
                                        "Exploration Noise Std": 0,
                                        "Random Start States (0 or 1)": False,
                                        "Number of rounds": 1,
                                        "Episode Length (0 = motion length)": 300,
                                    },
                                },
                "catchup":{ "set1":{ "Motion Id(s)": [9999],
                                    "Exploration Noise Std": 0,
                                    "Random Start States (0 or 1)": True,
                                    "Number of rounds": 1,
                                    "Episode Length (0 = motion length)": 1000,
                }, 
                "set2":{ "Motion Id(s)": [49, 39, 54, 31, 16, 68, 2, 5, 14, 58, 70, 73, 33, 29, 41],
                                    "Exploration Noise Std": 0,
                                    "Random Start States (0 or 1)": True,
                                    "Number of rounds": 1,
                                    "Episode Length (0 = motion length)": 1000,
                }}
                                        }
            for _ in range(1000):
                # Take input a string
                interactive_set_name = input("Enter the name of the interactive set: ")

                if interactive_set_name == "catchup":
                    interactive_set = interactive_set_library["catchup"]
                    player._render_anchored_demo = False
                    player._render_shifted_demo = True

                    for interactive_set in interactive_set.values():
                        m_ids = interactive_set["Motion Id(s)"]
                        explr_prob = interactive_set["Exploration Noise Std"]
                        random_start = interactive_set["Random Start States (0 or 1)"]
                        num_rounds = interactive_set["Number of rounds"]
                        req_ep_length = interactive_set["Episode Length (0 = motion length)"]
                    
                        if len(m_ids) == 1 and m_ids[0] == 9999:
                            num_motions = len(yaml.safe_load(open(args.motion_file, 'r'))['motions'])
                            m_ids = list(range(num_motions))

                        player.env.task.max_episode_length = 1000 if req_ep_length == 0 else req_ep_length

                        for round_num in range(num_rounds):
                            player.visualize(demo_idx_pool = m_ids,
                                            random_start = bool(random_start),
                                            render_demo = True,
                                            render_target_markers = render_target_markers,
                                            mimic_random_root_poses = False,
                                            enable_lookahead_mask = False,
                                            compose_demo = False)

                
                if interactive_set_name == "compose":
                    interactive_sets = interactive_set_library["compose"]
                    
                    for interactive_set in interactive_sets.values():
                        m_ids = interactive_set["Motion Id(s)"]
                        explr_prob = interactive_set["Exploration Noise Std"]
                        random_start = interactive_set["Random Start States (0 or 1)"]
                        num_rounds = interactive_set["Number of rounds"]
                        req_ep_length = interactive_set["Episode Length (0 = motion length)"]
                    
                        for round_num in range(num_rounds):
                            # Visualize Composition 
                            # -----------------------------------------------------------------------------------------------
                            player._render_anchored_demo = False
                            mocap_ep_len = int(player.env.task._motion_lib._motion_lengths[m_ids[1]]/player.env.task.dt)
                            player.env.task.max_episode_length = min(3000,mocap_ep_len) if req_ep_length == 0 else req_ep_length
                            player.visualize(demo_idx_pool = m_ids, 
                                            random_start = bool(random_start),
                                            render_demo = True,
                                            render_target_markers = render_target_markers,
                                            mimic_random_root_poses= False,
                                            enable_lookahead_mask = False,
                                            compose_demo= True)
                            # -----------------------------------------------------------------------------------------------

        if args.xbox_interactive: 
            
            from utils.xbone import XboxController
            joy = XboxController()

            def _render_speed_color():
                for env_index in range(player.env.num_envs):
                    # Extract the x and y components of the speed from the lookahead data
                    speed_x = player.env.task._tar_flat_full_state_obs[env_index, 7]
                    speed_y = player.env.task._tar_flat_full_state_obs[env_index, 8]
                    # Calculate the magnitude of the speed vector
                    speed_magnitude = torch.sqrt(speed_x**2 + speed_y**2).item()

                    player._set_char_color(player.speed_to_color((speed_magnitude-1)/2), torch.LongTensor([env_index]).cuda())


            def _render_demo_traj(self):
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

            def _read_joystick_commands(self,):
                all_env_ids = torch.arange(player.num_envs).type(torch.LongTensor).cuda()

                # Speed, Target Heading and Direction
                # --------------------------------------------------------------------------
                task_tar_facing_dir = torch.FloatTensor([[joy.LeftJoystickX + 1e-6, joy.LeftJoystickY]]).cuda()
                task_tar_dir = torch.FloatTensor([[joy.RightJoystickX + 1e-6, joy.RightJoystickY]]).cuda()
                
                task_req_height = 0.85 - 0.5*joy.LeftTrigger
                task_tar_speed = (torch.norm(task_tar_dir)* 3) + 1e-4

                task_tar_facing_dir = task_tar_facing_dir/torch.norm(task_tar_facing_dir, dim = -1).unsqueeze(-1)
                task_tar_dir = task_tar_dir/torch.norm(task_tar_dir, dim = -1).unsqueeze(-1)

                lookahead = torch.zeros(self.num_envs, self._lookahead_obs_dim).cuda()
                lookahead[:,2] = task_req_height # Fixed Height
                task_tar_face_theta = torch.atan2(task_tar_facing_dir[:,1], task_tar_facing_dir[:,0])
                lookahead[:,3:7] = quat_from_angle_axis(angle = task_tar_face_theta.cuda(),
                                                            axis = torch.FloatTensor([[0,0,1]]).cuda()) 
                lookahead[:,7:9] = task_tar_speed * task_tar_dir / torch.norm(task_tar_dir, dim = -1).unsqueeze(-1)

                self._global_random_dir_speed_lookahead_bucket[:] = lookahead
                # -------------------------------------------------------------------------------------    


                # Upper Body 
                # -----------------------------------------------------------------------------------------------
                # Motion Sets
                leftBumperMotionSet = [2,5, 24, 26, 30]    # Motion Set 0 
                rightBumperMotionSet = [17,15,16, 18,34]  # Motion Set 1
                defaultMotionSet = [65, 69, 70, 73, 68]  # Motion Set 2 for default case

                # Calculate index based on the right trigger's value
                # Assume rightTrigger is a value between 0 and 1
                index = int(10 * joy.RightTrigger / 2)

                if not hasattr(self, "imit_joystick_motion_id"):
                    self.imit_joystick_motion_id = 0


                if joy.Y:
                    self.imit_joystick_motion_id += 1
                    self._global_demo_start_rotations[all_env_ids] = self._sample_random_rotation_quat_for_demo(len(all_env_ids))
                if joy.A:
                    self.imit_joystick_motion_id -= 1
                    self._global_demo_start_rotations[all_env_ids] = self._sample_random_rotation_quat_for_demo(len(all_env_ids))
                if joy.B:
                    self.progress_buf[:] = 9999999

                self.imit_joystick_motion_id = self.imit_joystick_motion_id % self._motion_lib.num_motions()

                # Normal Xbox Setup
                # --------------------------------------------------------------------------------------------------------------
                player._render_demo = False
                self.show_traj = False
                self.show_root_traj = False
                self._random_dir_speed_env_idxs = all_env_ids
                self._random_dir_speed_upper_body_env_idxs = all_env_ids
                self._global_demo_lookahead_mask[all_env_ids] = self.lookahead_mask_pool[self._upper_body_conditioning_compatible_mask_pool_indx]
                # --------------------------------------------------------------------------------------------------------------

                # Apply motion set based on bumper states
                if joy.RightBumper and not joy.LeftBumper:
                    self._global_demo_start_motion_ids[:] = rightBumperMotionSet[index]
                elif joy.LeftBumper and not joy.RightBumper:
                    self._global_demo_start_motion_ids[:] = leftBumperMotionSet[index]
                elif not joy.LeftBumper and not joy.RightBumper:
                    # Default to the third set if neither bumper is pressed
                    self._global_demo_start_motion_ids[:] = defaultMotionSet[index]
                elif joy.LeftBumper and joy.RightBumper:
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


                if joy.X:
                    # print("Joystick X pressed", self.progress_buf[0], self._last_toggle_timestep)
                    if abs(self.progress_buf[0] - self._last_toggle_timestep) > 50:
                        # print("Togling Perturbation", self._enable_perturbation)
                        self._enable_perturbation = not self._enable_perturbation
                        self._last_toggle_timestep = self.progress_buf[0].clone()

                _render_speed_color()
                _render_demo_traj(player)


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


                
            player.env.task.pre_render_hook_fxn = MethodType(_read_joystick_commands, player.env.task)

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

        if args.interactive:
            for _ in range(100):

                PARAMETERS = {"Motion Id(s)": {"type": list, "default": [1]},
                    "Exploration Noise Std": {"type": float, "default": 0},
                    "Random Start States (0 or 1)": {"type": bool, "default": False},
                    "Number of rounds": {"type": int, "default": 1},
                    "Episode Length (0 = motion length)": {"type": int, "default": 0},}

                for param, options in PARAMETERS.items():
                    try:
                        if options["type"] == list:
                            input_value = options["type"]([int(i) for i in input(param + f" [{options['default']}]: ").split(",")])
                        else:
                            input_value = options["type"](input(param + f" [{options['default']}]: ") or options["default"])
                        # if input is not specified, use default value
                    except ValueError:
                        print(f"Invalid input for {param}. Using default value: {options['default']}")
                        input_value = options["default"]
                    PARAMETERS[param]["value"] = input_value

                # access parameter values like this:
                m_ids = PARAMETERS["Motion Id(s)"]["value"]
                explr_prob = PARAMETERS["Exploration Noise Std"]["value"]
                random_start = PARAMETERS["Random Start States (0 or 1)"]["value"]
                num_rounds = PARAMETERS["Number of rounds"]["value"]
                req_ep_length = PARAMETERS["Episode Length (0 = motion length)"]["value"]
                    
                if len(m_ids) == 1 and m_ids[0] == 9999:
                    num_motions = len(yaml.safe_load(open(args.motion_file, 'r'))['motions'])
                    m_ids = list(range(num_motions))

                for round_num in range(num_rounds):

                    if m_ids[0] == -1:
                        player._render_anchored_demo = False
                        player._render_shifted_demo = False
                        if len(m_ids) == 5:                  
                            # Viualizing random Headings
                            player.env.task.max_episode_length = 500 if req_ep_length == 0 else req_ep_length
                            player.env.task._req_height = m_ids[1]/100
                            player.env.task._random_speed_min = m_ids[2]/100
                            player.env.task._random_speed_max = m_ids[3]/100 
                        
                            m_ids = [-1] # replace the m_ids before sending for visualziation
                            uniform_targets = bool(m_ids[4])
                        else:
                            # Viualizing random Headings
                            player.env.task.max_episode_length = 500 if req_ep_length == 0 else req_ep_length
                            player.env.task._req_height = 0.85
                            player.env.task._random_speed_min = 1.5
                            player.env.task._random_speed_max = 1.6 # You can manually change these settings. 
                            uniform_targets = False

                        player.visualize(demo_idx_pool = m_ids, 
                                        random_start = bool(random_start),
                                        uniform_targets = uniform_targets, 
                                        render_demo = False,
                                        render_target_markers = True,
                                        mimic_random_root_poses=True, 
                                        upper_body_conditioning=False)
                        
                    elif m_ids[0] == -2:
                        player._render_anchored_demo = True
                        player._render_shifted_demo = False
                        if len(m_ids) == 6:                  
                            # Viualizing random Headings
                            player.env.task.max_episode_length = 500 if req_ep_length == 0 else req_ep_length
                            player.env.task._req_height = m_ids[2]/100
                            player.env.task._random_speed_min = m_ids[3]/100
                            player.env.task._random_speed_max = m_ids[4]/100 
                        
                            m_ids = [-1] # replace the m_ids before sending for visualziation
                            uniform_targets = bool(m_ids[5])
                        else:
                            # Viualizing random Headings
                            player.env.task.max_episode_length = 500 if req_ep_length == 0 else req_ep_length
                            player.env.task._req_height = 0.85
                            player.env.task._random_speed_min = 1.5
                            player.env.task._random_speed_max = 1.6 # You can manually change these settings. 
                            uniform_targets = False

                        player.visualize(demo_idx_pool = [m_ids[1]], 
                                        random_start = bool(random_start),
                                        uniform_targets = uniform_targets, 
                                        render_demo = True,
                                        render_target_markers = True,
                                        mimic_random_root_poses=True, 
                                        upper_body_conditioning=True)
                        
                    elif len(m_ids) == 2 and m_ids[0] == -3:
                        player._render_anchored_demo = False
                        player._render_shifted_demo = True
                        if m_ids[1] == 9999:
                            for m_id in args.demo_motion_ids:
                                mocap_ep_len = int(player.env.task._motion_lib._motion_lengths[m_id]/player.env.task.dt)
                                player.env.task.max_episode_length = min(3000,mocap_ep_len) if req_ep_length == 0 else req_ep_length
                                player.visualize(demo_idx_pool = [m_id], 
                                                random_start = bool(random_start),
                                                render_demo = True,
                                                render_target_markers = True,
                                                mimic_random_root_poses= False,
                                                enable_lookahead_mask = True)
                        # Viualizing random Headings with Composed upper body
                        # Visualize Lookahead Masks
                        else:
                            mocap_ep_len = int(player.env.task._motion_lib._motion_lengths[m_ids[1]]/player.env.task.dt)
                            player.env.task.max_episode_length = min(3000,mocap_ep_len) if req_ep_length == 0 else req_ep_length
                            player.visualize(demo_idx_pool = [m_ids[1]], 
                                            random_start = bool(random_start),
                                            render_demo = True,
                                            render_target_markers = True,
                                            mimic_random_root_poses= False,
                                            enable_lookahead_mask = True,
                                            show_traj = True,
                                            show_heading = True)

                    elif len(m_ids) == 3 and m_ids[0] == -4:
                        player._render_anchored_demo = True
                        mocap_ep_len = int(player.env.task._motion_lib._motion_lengths[m_ids[1]]/player.env.task.dt)
                        player.env.task.max_episode_length = min(3000,mocap_ep_len) if req_ep_length == 0 else req_ep_length
                        player.visualize(demo_idx_pool = m_ids[1:], 
                                        random_start = bool(random_start),
                                        render_demo = True,
                                        render_target_markers = render_target_markers,
                                        mimic_random_root_poses= False,
                                        enable_lookahead_mask = True,
                                        compose_demo= True)
                    else:
                        player._render_anchored_demo = False
                        player._render_shifted_demo = True
                        # player._render_shifted_demo = bool(random_start)

                        if round_num > 1 and not bool(random_start):
                            m_id = m_ids[round_num%len(m_ids)]
                            mocap_ep_len = player.env.task._motion_max_steps[m_id]
                            viz_m_ids = [m_id]
                        else:
                            mocap_ep_len = max([player.env.task._motion_max_steps[m_id] for m_id in m_ids])
                            viz_m_ids = m_ids 

                        print("Demo Id(s)", viz_m_ids)
                        for m_id in viz_m_ids:
                            print(m_id, "Motion Name",player.env.task._motion_lib.demo_store["demo_names"][m_id],  "Motion Len", mocap_ep_len)
                        
                        if random_start :
                            player.env.task.max_episode_length = 1000 if req_ep_length == 0 else req_ep_length 
                        else:
                            player.env.task.max_episode_length = min(3000,mocap_ep_len) if req_ep_length == 0 else req_ep_length 

                        player.games_num = num_rounds
                        player.visualize(demo_idx_pool = viz_m_ids,
                                        random_start = bool(random_start),
                                        render_demo = render_demo,
                                        render_target_markers = render_target_markers,)
        
        else:
            for m_id in args.demo_motion_ids:
                mocap_ep_len = int(player.env.task._motion_lib._motion_lengths[m_id]/player.env.task.dt)
                print("Demo Id", m_id)
                print(m_id, "Motion Name",player.env.task._motion_lib.demo_store["demo_names"][m_id],  "Motion Len", mocap_ep_len)

                player.env.task.max_episode_length = min(3000,mocap_ep_len)
                player.games_num = 1
                player.visualize(demo_idx_pool = [m_id], 
                        random_start = False,
                        render_demo = render_demo,
                        render_target_markers = render_target_markers,)
    
    if args.benchmark_policy:
        player.benchmark_policy(benchmark_config = {"mode": args.benchmark_type, 
                                                "horizon": 360,
                                                "success_criteria": "shifted-mpjpe" if args.benchmark_player == "default" else "local-mpjpe",
                                                "honor_mask_for_rewards": True,},
                                write_to_file = True,
                                label = f"{args.wandb_path.split('/')[-2]}_Dataset-{args.benchmark_dataset}_Player-{args.benchmark_player}_Mode-{args.benchmark_type}", 
                                verbose = True)
