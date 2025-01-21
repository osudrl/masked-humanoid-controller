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
# ----- Config Setup --------------------------
# --------------------------------------------
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

def parse_mhc_args_and_config():
    # Get Arguments
    set_np_formatting()
    extra_parameters = get_extra_parameters()
    args = get_args(extra_parameters=extra_parameters)

    # Demo Motion IDs Setup
    args.demo_motion_ids = [int(x) for x in args.demo_motion_ids.split(',')]
    if len(args.demo_motion_ids) == 1 and args.demo_motion_ids[0] == 9999:
        num_motions = len(yaml.safe_load(open(args.motion_file, 'r'))['motions'])
        args.demo_motion_ids = list(range(num_motions))
    

    # Motion Set Description
    if len(args.demo_motion_ids) > 1:
        motion_set_desc = f"Train_MIdSetOf_{len(args.demo_motion_ids)}"
    else:
        motion_set_desc = f"Train_MId_{args.demo_motion_ids[0]}"
    args.motion_set_desc = motion_set_desc
    args.experiment_name = 'AMP_' + motion_set_desc + datetime.now().strftime("_%y%m%d-%H%M%S")

    # Task Setup
    if args.train_run:
        args.task = 'HumanoidAMPGetupRobust'
    else:
        args.task = 'HumanoidAMPGetupRobustTaskViz'

    args.checkpoint = ''

    cfg, cfg_train, logdir = setup_base_config(args)

    return args, cfg, cfg_train, logdir



# --------------------------------------------
# ------------------train---------------------
# --------------------------------------------
if __name__ == "__main__":
    # Parse Arguments
    args, cfg, cfg_train, logdir = parse_mhc_args_and_config()

    # Initialize Wandb
    wandb.init(
        entity = 'dacmdp',
        project = args.wandb_project if not args.debug else "catchup_amp_debug",
        name = args.experiment_name,
        config={ "info": "v1: disc sn",
                "motion": args.motion_set_desc,
                "arg_dump": args,
                "cfg_dump": cfg,
                "cfg_train_dump":cfg_train,
                },
        sync_tensorboard=True,
    )

    # Build Network
    network = AMPBuilderV1()
    network.load(cfg_train['params']['network'])

    # Build Agent
    config = copy.deepcopy(cfg_train['params']['config'])
    config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
    config['network'] = ModelAMPContinuousv1(network)
    config['features'] = {}
    config['features']['observer'] = RLGPUAlgoObserver()
    config['full_experiment_name'] = args.experiment_name
    agent = AMPAgent2(base_name='run', config=config)

    # Restore from wandb
    agent.restore_from_wandb(args.wandb_path)

    # Setup Dynamic Weighting 
    agent._dynamic_weighting = args.dynamic_weighting
    agent._dynamic_weighting_start_epoch = args.dynamic_weighting_start_epoch

    print("Device Motionlib", agent.vec_env.env.task._motion_lib._device)
    print("Device demo_store", agent.vec_env.env.task._motion_lib.demo_store["amp_obs"].device)
    
    # Train Agent
    agent.train()
