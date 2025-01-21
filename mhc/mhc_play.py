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

from mhc_train import create_rlgpu_env, RLGPUEnv
from mhc_train import setup_base_config, get_extra_parameters


# --------------------------------------------
# ----- Custom overrides/ Housekeeping--------
# --------------------------------------------
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(args, cfg, cfg_train, **kwargs),
    'vecenv_type': 'RLGPU'})


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


def parse_play_options():
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
    random_start = PARAMETERS["Random Start States (0 or 1)"]["value"]
    num_rounds = PARAMETERS["Number of rounds"]["value"]
    req_ep_length = PARAMETERS["Episode Length (0 = motion length)"]["value"] 

    # If 9999 is passed, visualize all motions
    if len(m_ids) == 1 and m_ids[0] == 9999:
        m_ids = list(range(len(yaml.safe_load(open(args.motion_file, 'r'))['motions'])))

    return m_ids, random_start, num_rounds, req_ep_length

from mhc_train import parse_mhc_args_and_config
if __name__ == "__main__":
    # Get Arguments
    args, cfg, cfg_train, logdir = parse_mhc_args_and_config()

   

    # rl_games expecting config later

    # Setup Player
    network = AMPBuilderV1()
    network.load(cfg_train['params']['network'])
    # cfg["env"]["CACHE_HORIZON"] = 3005
    config = copy.deepcopy(cfg_train['params']['config'])
    config['network'] = ModelAMPContinuous(network)
    player = AMPPlayer2(config)
    player.env.task._update_camera = MethodType(custom_update_camera, player.env.task)

    # Restore from wandb
    run = wandb.init(entity = "dacmdp", project="amp_debug", name = "VisualizeRun" + datetime.now().strftime("_%y%m%d-%H%M%S"))
    player.restore_from_wandb(args.wandb_path)

    # Seeding Run Variables
    player.games_num= 1
    player.env.task.max_episode_length = 10
    player.env.reset(torch.arange(player.env.task.num_envs).cuda())
    player.run()


    render_demo=True
    render_target_markers = True
    
    if args.interactive:
        while True:
            m_ids, random_start, num_rounds, req_ep_length = parse_play_options()
            for round_num in range(num_rounds):
                if m_ids[0] == -1: 
                    # Visualize Random Headings Mode
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
                    # Visualize random headings, while imitating upper body motion mode. 
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
                    # ToDo Figure out what this mode is , and is it necessary. 

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
                                        show_headings = True)

                elif len(m_ids) == 3 and m_ids[0] == -4:
                    # Visualize Composed Demo Mode
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
                    # Default Mode

                    player._render_anchored_demo = False
                    player._render_shifted_demo = True

                    mocap_ep_len = max([player.env.task._motion_max_steps[m_id] for m_id in m_ids])
                    viz_m_ids = m_ids 

                    print("Demo Id(s)", viz_m_ids)
                    for m_id in viz_m_ids:
                        print(m_id, "Motion Name",player.env.task._motion_lib.demo_store["demo_names"][m_id],  "Motion Len", mocap_ep_len)
                    
                    if random_start:
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