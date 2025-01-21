from collections import namedtuple
from typing import Tuple
import numpy as np
import torch
import time
import copy
from types import SimpleNamespace
import os



# Generic replay buffer for standard gym tasks
# most commonly used.
class StandardBuffer(object):
    """
    Initializes an array for elements of transitions as per the maximum buffer size. 
    Keeps track of the crt_size. 
    Saves the buffer element-wise as numpy array. Fast save and retreival compared to pickle dumps. 
    """
    def __init__(self, state_shape, action_shape,  history_shape, buffer_size, device, batch_size = 64):
        
        self.state_shape = state_shape
        self.action_shape = action_shape
        
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, *state_shape))
        self.action = np.zeros((self.max_size, *action_shape))
        self.tran_history = np.zeros((self.max_size, *history_shape))
        self.next_state = np.zeros_like(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        
        self.weights = np.zeros((self.max_size,)) + 1

        # Normalization parameters. 
        self.norm_params = SimpleNamespace(is_state_normalized = False,  is_action_normalized = False, 
                                        state_mean = None, state_std = None,
                                        action_mean = None, action_std = None)
    
    def __len__(self):
        return self.crt_size

    def __repr__(self):
        return f"Standard Buffer: \n \
                Total number of transitions: {len(self)}/{self.max_size} \n \
                State Store Shape: {self.state.shape} \n \
                Action Store Shape: {self.action.shape} \n"

    @property
    def all_states(self):
        return self.state[:self.crt_size]

    @property
    def all_next_states(self):
        return self.next_state[:self.crt_size]
    
    @property
    def all_actions(self):
        return self.action[:self.crt_size]

    @property
    def all_not_ep_ends(self):
        return self.not_done[:self.crt_size]

    @property
    def all_ep_ends(self):
        return 1- self.not_done[:self.crt_size]

    @property
    def all_rewards(self):
        return self.reward[:self.crt_size]

    @property
    def all_tran_history(self):
        return self.reward[:self.crt_size]


    def add(self, state, action, tran_history, next_state, reward, done, episode_done=None, episode_start=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.tran_history[self.ptr] = tran_history
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)


    def sample_indices(self, batch_size = None):
        indxs = np.random.randint(0, self.crt_size, size = batch_size or self.batch_size)
        return indxs

    def sample_using_indices(self, indxs, device = None):
        device = device or self.device
        indxs = np.array(indxs)
        return (
            torch.FloatTensor(self.state[indxs]).to(device),
            torch.FloatTensor(self.action[indxs]).to(device),
            torch.FloatTensor(self.tran_history[indxs]).to(device),
            torch.FloatTensor(self.next_state[indxs]).to(device),
            torch.FloatTensor(self.reward[indxs]).to(device),
            torch.FloatTensor(self.not_done[indxs]).to(device)
        )
    
    def sample(self, batch_size= None):
        indxs = self.sample_indices(batch_size or self.batch_size)
        return self.sample_using_indices(indxs)

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_tran_history.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
        self.tran_history[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")

        
        
import time
import torch
from ase.env.tasks.humanoid_amp import build_amp_observations, gymtorch, to_torch

def c_print_motion_stats(task, motion_id):
    print("Motion Id",motion_id)
    print("Motion File", task._motion_lib._motion_files[motion_id].split("/")[-1])
    print("Motion Length", task._motion_lib._motion_lengths[motion_id])
    print("Motion Time Steps", int(task._motion_lib._motion_lengths[motion_id].item() // task.dt))
    print("Motion Weight", task._motion_lib._motion_weights[motion_id].item())
    
    
def c_set_env_state(task, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
    task._set_env_state(env_ids=env_ids, 
                        root_pos=root_pos[env_ids], 
                        root_rot=root_rot[env_ids], 
                        dof_pos=dof_pos[env_ids], 
                        root_vel=root_vel[env_ids], 
                        root_ang_vel=root_ang_vel[env_ids], 
                        dof_vel=dof_vel[env_ids], )
    task._reset_env_tensors(env_ids) # needs to be called only once for each env step 
    task._refresh_sim_tensors()
    task._compute_observations(env_ids)


    

def c_build_amp_obs(task, env_ids, motion_id, motion_step, 
                    set_state = False, do_a_simulation_step = False):
    
    num_envs = task.num_envs
    motion_ids = task._motion_lib.sample_motions(num_envs)
    motion_ids[:] = motion_id
    motion_times = motion_step*task.dt * torch.ones(num_envs, device=task.device) - \
                        task.dt * torch.ones(num_envs, device=task.device)
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = task._motion_lib.get_motion_state(motion_ids, motion_times)
    
    internal_vars = {"root_pos":root_pos, 
                     "root_rot":root_rot,
                     "dof_pos":dof_pos, 
                     "root_vel":root_vel,
                     "root_ang_vel":root_ang_vel, 
                     "dof_vel":dof_vel, 
                     "key_pos":key_pos, 
                     "rigid_body_pos": None}
    
    amp_obs = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                  dof_pos, dof_vel, key_pos, 
                                  task._local_root_obs, task._root_height_obs, 
                                  task._dof_obs_size, task._dof_offsets).clone()

    if set_state:
        c_set_env_state(task, 
                        env_ids=env_ids, 
                        root_pos=root_pos[env_ids], 
                        root_rot=root_rot[env_ids], 
                        dof_pos=dof_pos[env_ids], 
                        root_vel=root_vel[env_ids], 
                        root_ang_vel=root_ang_vel[env_ids], 
                        dof_vel=dof_vel[env_ids])
        
        if do_a_simulation_step:
            gym, sim, viewer = task.gym,task.sim, task.viewer
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim);
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            task._refresh_sim_tensors()
            task._compute_observations(env_ids)
            internal_vars["rigid_body_pos"] = task._rigid_body_pos.clone().cpu()
            
    return amp_obs, internal_vars

def c_get_full_amp_motion(p, motion_id, play_motion = False, lag = 0.05):
    """
    Returns the full amp motion.
    Also has an option to play the motion in all environments.
    """
    
    c_print_motion_stats(p.env.task, motion_id)
    env_ids = torch.arange(p.env.task.num_envs).cuda()
    motion_len = int(p.env.task._motion_lib._motion_lengths[motion_id].item() // p.env.task.dt)
    demo_amp_obs = []
    demo_internal_vars = {"root_pos":[], 
                         "root_rot":[],
                         "root_vel":[],
                         "root_ang_vel":[], 
                         "dof_pos":[],
                         "dof_vel":[],
                         "key_pos":[],
                         "rigid_body_pos":[]
                         }
    
    time.sleep(lag)
    curr_amp_obs = c_build_amp_obs(p.env.task, env_ids, motion_id, motion_step=0, 
                                   set_state=play_motion, do_a_simulation_step = play_motion)
    p._change_char_color(env_ids)
    for step_i in range(1, motion_len + 1):
        curr_amp_obs, internal_vars = c_build_amp_obs(p.env.task, env_ids, motion_id, step_i, 
                                      set_state = play_motion, 
                                      do_a_simulation_step = play_motion)
        demo_amp_obs.append(curr_amp_obs.clone()[0]) # Just log the first env var
        for k,v in internal_vars.items():
            if v is not None:
                demo_internal_vars[k].append(v[0])
            
        time.sleep(lag)
        
    for k in demo_internal_vars.keys():
        demo_internal_vars[k] =torch.stack(demo_internal_vars[k]) if demo_internal_vars[k] else None
    return torch.stack(demo_amp_obs), demo_internal_vars




from collections import defaultdict

def get_rollout_as_demo(p, latent_repeat, rollout_len, latent_repeat_prob):    
    # Set Env Ids to particular
    num_envs = p.env.task.num_envs
    internal_states = []
    p._ase_latents = p.model.a2c_network.sample_latents(num_envs)

    rollout_amp_obs = torch.zeros(num_envs, rollout_len, p.env.task._amp_obs_buf.size(-1)).cuda()
    dofs_per_env = gymtorch.wrap_tensor(p.env.task.gym.acquire_dof_state_tensor(p.env.task.sim)).clone().cpu().shape[0] // num_envs
    rollout_internal_states = defaultdict( lambda : {"root_pos":torch.zeros(rollout_len,3).cuda(), 
                                "root_rot":torch.zeros(rollout_len,4).cuda(),
                                "root_vel":torch.zeros(rollout_len,3).cuda(),
                                "root_ang_vel":torch.zeros(rollout_len,3).cuda(), 
                                "dof_pos":torch.zeros(rollout_len, dofs_per_env).cuda(),
                                "dof_vel":torch.zeros(rollout_len, dofs_per_env).cuda(),
                                })

    obs_dict = p.env_reset()
    for i in range(rollout_len):
        if i%latent_repeat == 0  and torch.rand(1).item()>latent_repeat_prob:
            p._ase_latents = p.model.a2c_network.sample_latents(num_envs)
            p._change_char_color(range(num_envs))


        p._latent_step_count = 100 # To ensure that latent action does not change
        low_level_action = p.get_action(obs_dict, is_determenistic = True)
        obs, r, done, info =  p.env_step(p.env, low_level_action)
        obs_dict = {"obs": obs}  
        
        _dof_state = gymtorch.wrap_tensor(p.env.task.gym.acquire_dof_state_tensor(p.env.task.sim)).clone()
        _actor_root = gymtorch.wrap_tensor(p.env.task.gym.acquire_actor_root_state_tensor(p.env.task.sim)).clone()
        for eid in range(num_envs):
            rollout_amp_obs[eid,i,:] = p.env.task._curr_amp_obs_buf[eid]
            rollout_internal_states[eid]["root_pos"][i,:] = _actor_root[eid,0:3].clone()
            rollout_internal_states[eid]["root_rot"][i,:] = _actor_root[eid,3:7].clone()
            rollout_internal_states[eid]["root_vel"][i,:] = _actor_root[eid,7:10].clone()
            rollout_internal_states[eid]["root_ang_vel"][i,:] = _actor_root[eid,10:13].clone()
            rollout_internal_states[eid]["dof_pos"][i,:] = _dof_state.view(num_envs, dofs_per_env, 2)[..., :dofs_per_env, 0].clone()[eid,:]
            rollout_internal_states[eid]["dof_vel"][i,:] = _dof_state.view(num_envs, dofs_per_env, 2)[..., :dofs_per_env, 1].clone()[eid,:]
    
    return rollout_amp_obs, rollout_internal_states
       




import types
import copy
import time
import pickle as pk
import torch
import time
import numpy as np 



class InternalBufferLogger():
    def __init__(self, episode_len = 80):
        self.raw_internal_state_buffer = []
        self.raw_internal_state_history_buffer = []
        self.raw_internal_next_state_buffer = []
        self.raw_latent_action_buffer = []
        self.raw_reward_buffer =[]
        self.raw_done_buffer =[]
        
        
        self.internal_state_buffer_by_ep = torch.FloatTensor([])
        self.internal_next_state_buffer_by_ep = torch.FloatTensor([])
        self.latent_action_buffer_by_ep = torch.FloatTensor([])
        self.reward_buffer_by_ep = torch.FloatTensor([])
        
        self.episode_len = None

    #Sanitize buffer 
    def sanitize_raw_buffers(self):
        minlen = min([len(self.raw_internal_state_buffer ),
                      len(self.raw_internal_state_history_buffer ),
                      len(self.raw_internal_next_state_buffer ),
                      len(self.raw_latent_action_buffer ),
                      len(self.raw_reward_buffer ),
                      len(self.raw_done_buffer )
                     ])
        
        self.raw_internal_state_buffer = self.raw_internal_state_buffer[:minlen]
        self.raw_internal_state_history_buffer = self.raw_internal_state_history_buffer[:minlen]
        self.raw_internal_next_state_buffer = self.raw_internal_next_state_buffer[:minlen]
        self.raw_latent_action_buffer = self.raw_latent_action_buffer[:minlen]
        self.raw_reward_buffer = self.raw_reward_buffer[:minlen]
        self.raw_done_buffer = self.raw_done_buffer[:minlen]
        
        
        if "tar_pos" in self.raw_internal_next_state_buffer[0]:
            for i in range(len(self.raw_internal_state_buffer)): 
                l = len(self.raw_internal_next_state_buffer[i]["tar_pos"])
                self.raw_internal_next_state_buffer[i]["tar_pos"][:l] = self.raw_internal_state_buffer[i]["tar_pos"][:l]
                
            
        
    def reset_raw_buffers(self):
        self.raw_internal_state_buffer = []
        self.raw_latent_action_buffer = []
        self.raw_internal_state_history_buffer = []
        self.raw_internal_next_state_buffer = []
        self.raw_reward_buffer =[]

    def save(self, fn):
        pk.dump((self.internal_state_buffer, self.internal_next_state_buffer,  self.latent_action_buffer, self.reward_buffer), open(fn, "wb"))
        
    def load(self, fn):
        self.internal_state_buffer, self.internal_next_state_buffer,  self.latent_action_buffer, self.reward_buffer = pk.load(open(fn, "rb"))
        

        
        
# #     def get_action(self, obs_dict, is_determenistic=False, update_latent = True):
# #         if update_latent:
# #             self._update_latents()
# #         else:
# #             pass

# #         obs = obs_dict['obs']
# #         if len(obs.size()) == len(self.obs_shape):
# #             obs = obs.unsqueeze(0)
# #         obs = self._preproc_obs(obs)
# #         ase_latents = self._ase_latents

# #         input_dict = {
# #             'is_train': False,
# #             'prev_actions': None, 
# #             'obs' : obs,
# #             'rnn_states' : self.states,
# #             'ase_latents': ase_latents
# #         }
# #         with torch.no_grad():
# #             res_dict = self.model(input_dict)
# #         mu = res_dict['mus']
# #         action = res_dict['actions']
# #         self.states = res_dict['rnn_states']
# #         if is_determenistic:
# #             current_action = mu
# #         else:
# #             current_action = action
# #         current_action = torch.squeeze(current_action.detach())
# #         return  players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))


import random 

def append_to_raw_buffers(logger, p, n_games, 
                          latent_steps_phase_min = 1, 
                          latent_steps_phase_max = 10,
                          latent_steps_phase = 30, 
                          specialize_for_motion_ids = []):
    
    if len(specialize_for_motion_ids)>0:
        def_motion_weights = p.env.task._motion_lib._motion_weights.clone()
        new_motion_weights = p.env.task._motion_lib._motion_weights.clone()
        new_motion_weights[specialize_for_motion_ids] = 1
        p.env.task._motion_lib._motion_weights = new_motion_weights/torch.sum(new_motion_weights)
        print("Switched motion weights to favor selection motion_ids", specialize_for_motion_ids, p.env.task._motion_lib._motion_weights)
        

    def _c_pre_step(p, info):
        num_envs =  p.env.task.num_envs
        _dof_state = gymtorch.wrap_tensor(p.env.task.gym.acquire_dof_state_tensor(p.env.task.sim)).clone().cpu()
        _actor_root = gymtorch.wrap_tensor(p.env.task.gym.acquire_actor_root_state_tensor(p.env.task.sim)).clone().cpu()
        dofs_per_env = _dof_state.shape[0] // num_envs
        
        logger.raw_internal_state_buffer.append({
            "root_pos":_actor_root[:,0:3], 
            "root_rot":_actor_root[:,3:7],
            "root_vel":_actor_root[:,7:10],
            "root_ang_vel":_actor_root[:,10:13], 
            "dof_pos":_dof_state.view(num_envs, dofs_per_env, 2)[..., :dofs_per_env, 0],
            "dof_vel":_dof_state.view(num_envs, dofs_per_env, 2)[..., :dofs_per_env, 1],
            "Observation Buffer":info["Observation Buffer"],
            "AMP Observation Buffer":info["AMP Observation Buffer"],
                                                })
        logger.raw_latent_action_buffer.append(info["Latent Action"])
        return 

    def _c_post_step(p, info):
        num_envs = p.env.task.num_envs
        logger.raw_internal_state_history_buffer.append({"Observation Buffer History":info["Observation Buffer History"],
                                                      "AMP Observation Buffer History":info["AMP Observation Buffer History"],})
        logger.raw_internal_next_state_buffer.append({"Observation Buffer":info["Observation Buffer"],
                                                      "AMP Observation Buffer":info["AMP Observation Buffer"],})
        logger.raw_reward_buffer.append(info["Rewards"])
        logger.raw_done_buffer.append(info["Done"])

        return 
    
    
    p._latent_steps_phase = latent_steps_phase
    p._latent_steps_phase_min, p._latent_steps_phase_max = latent_steps_phase_min, latent_steps_phase_max
    
    p._c_pre_step = types.MethodType(_c_pre_step, p)
    p._c_post_step = types.MethodType(_c_post_step, p)
    
    num_envs = p.env.task.num_envs
    r_game = torch.zeros(num_envs, dtype=torch.float32, device=p.device)
    steps = torch.zeros(num_envs, dtype=torch.float32, device=p.device)
    done_indices = []
    games_played = 0
    sum_rewards, sum_steps = 0, 0
    st = time.time()
    
    for _ in range(n_games):
        #### Reset Environments ==============================
        obs_dict = p.env_reset()
        # ==================================================================================================
        
        #### Reset Latents ===================================
        p._c_curr_latent_step_phase = np.random.randint(p._latent_steps_phase_min, p._latent_steps_phase_max)
        p._ase_latents = p.model.a2c_network.sample_latents(num_envs)  # high_level_action
        cache_ase = p._ase_latents.clone()
        p._change_char_color(range(num_envs))
        # ==================================================================================================
        
        for n in range(p.max_steps):     
            #### Reset Environments =========================
            obs_dict = p.env_reset(done_indices)
            p._ase_latents[done_indices] = p.model.a2c_network.sample_latents(len(done_indices))  # high_level_action
            cache_ase = p._ase_latents.clone()
            # ===============================================================================================
            
            #### Reset Latents ===================================
            p._c_curr_latent_step_phase -=1
            if p._c_curr_latent_step_phase<=0:
                p._c_curr_latent_step_phase = np.random.randint(p._latent_steps_phase_min, p._latent_steps_phase_max)
                p._ase_latents = p.model.a2c_network.sample_latents(num_envs)  # high_level_action
                cache_ase = p._ase_latents.clone()
                p._change_char_color(range(num_envs))
            # ===============================================================================================
            
            #### Collect Data from Simulator =========================
            r_step = torch.zeros(num_envs, dtype=torch.float32, device=p.device)
            obs_history = torch.zeros((p._latent_steps_phase,*p.env.task.obs_buf.shape), dtype=torch.float32).cpu()
            amp_obs_history = torch.zeros((p._latent_steps_phase,*p.env.task._curr_amp_obs_buf.shape), dtype=torch.float32).cpu()
            
            p._c_pre_step({"Observation Buffer": p.env.task.obs_buf.clone().cpu(),
                            "AMP Observation Buffer": p.env.task._curr_amp_obs_buf.clone().cpu(),
                            "Latent Action": p._ase_latents.clone().cpu()})
            for ph_i in range(p._latent_steps_phase):
                p._latent_step_count = 100
                low_level_action = p.get_action(obs_dict, is_determenistic = True)
                obs, r, done, info =  p.env_step(p.env, low_level_action)
                obs_dict = {"obs": obs}              
                obs_history[ph_i] = p.env.task.obs_buf.clone().cpu()
                amp_obs_history[ph_i] = p.env.task._curr_amp_obs_buf.clone().cpu()
                r_step, r_game, steps = r_step+r , r_game+r, steps+1
            
            assert torch.sum(cache_ase - p._ase_latents) == 0, print(torch.sum(cache_ase, dim = -1), torch.sum(p._ase_latents, dim = -1))
            # print(torch.sum(cache_ase - p._ase_latents))
                # print(ph_i, torch.sum(p.env.task._curr_amp_obs_buf), torch.sum(p.env.task._amp_obs_buf[:,-2,:]), )
            # print(torch.sum(p.env.task._amp_obs_buf.clone().cpu() - amp_obs_history.permute(1,0,2)))
                
            p._c_post_step({"Observation Buffer": p.env.task.obs_buf.clone().cpu(),
                            "AMP Observation Buffer": p.env.task._curr_amp_obs_buf.clone().cpu(),
                            "Observation Buffer History": obs_history.clone().cpu(),
                            "AMP Observation Buffer History": amp_obs_history.clone().cpu(),
                            "Rewards": r_step, "Done": done})
            # ===============================================================================================
            
            
            #### Done Environments logic ====================================================================
            all_done_indices = done.nonzero(as_tuple=False)
            done_indices = all_done_indices[::p.num_agents].reshape(-1)
            done_count = len(done_indices)
            games_played += done_count

            if done_count > 0:
                # print("resetting envs", done_indices)
                # print(len(done_indices))
                
                cur_rewards = r_game[done_indices].sum().item()
                cur_steps = steps[done_indices].sum().item()

                r_game = r_game * (1.0 - done.float())
                steps = steps * (1.0 - done.float())
                sum_rewards += cur_rewards
                sum_steps += cur_steps
            # ===============================================================================================
            if games_played >=n_games:
                break
        if games_played >=n_games:
            break

            
    if len(specialize_for_motion_ids)>0:
        p.env.task._motion_lib._motion_weights = def_motion_weights.clone()
        print("Switched motion weights back to default", p.env.task._motion_lib._motion_weights)

    print(f"Total time for collection:{time.time()-st}, \n Length of Dataset {len(logger.raw_internal_state_buffer)}")
    return logger



def compute_dac_buffer_from_raw_buffer(self, max_buffer_size = int(1e10),
                                       obs_tag = "AMP Observation Buffer", 
                                       hist_tag = "AMP Observation Buffer",
                                       obs_filter_fxn = None):  
    self.sanitize_raw_buffers()
    obs_filter_fxn = obs_filter_fxn or (lambda s:s)
    new_states_by_thread = torch.stack([obs_filter_fxn(buffer_item[obs_tag])
                                    for buffer_item in self.raw_internal_state_buffer]).permute(1,0,2) # n_threads, data_len, obs_dim
    new_next_states_by_thread = torch.stack([obs_filter_fxn(buffer_item[obs_tag])
                                    for buffer_item in self.raw_internal_next_state_buffer]).permute(1,0,2) # n_threads, data_len, obs_dim
    tran_history_by_thread = torch.stack([obs_filter_fxn(buffer_item[hist_tag + " History"])
                            for buffer_item in self.raw_internal_state_history_buffer]).permute(2,0,1,3) # n_threads, data_len, history_size, obs_dim

    nt, dl, hs, od = tran_history_by_thread.shape
    observations_flat = new_states_by_thread.reshape(nt*dl, -1)
    next_observations_flat = new_next_states_by_thread.reshape(nt*dl, -1)
    tran_history_flat = tran_history_by_thread.reshape(nt*dl,hs*od)
    latent_actions_flat = torch.stack([o for o in self.raw_latent_action_buffer]).permute(1,0,2).reshape(nt*dl, -1)
    rewards_flat = torch.stack([o for o in self.raw_reward_buffer]).permute(1,0).reshape(nt*dl, -1)
    done_flat = torch.stack([o for o in self.raw_done_buffer]).permute(1,0).reshape(nt*dl, -1)
    print(observations_flat.shape, next_observations_flat.shape, tran_history_flat.shape, latent_actions_flat.shape, rewards_flat.shape, done_flat.shape)

    n_transitions = min(observations_flat.size(0), max_buffer_size)
    data_buffer_train = StandardBuffer(state_shape=(observations_flat.size(1),),
                              action_shape=(latent_actions_flat.size(1),),
                              history_shape = (hs*od,),
                              buffer_size= n_transitions, 
                              device = "cuda",
                              batch_size = 64)

    data_buffer_train.state[:n_transitions] = observations_flat.cpu().numpy()[:n_transitions]
    data_buffer_train.tran_history[:n_transitions] = tran_history_flat.cpu().numpy()[:n_transitions]
    data_buffer_train.next_state[:n_transitions] = next_observations_flat.cpu().numpy()[:n_transitions]
    data_buffer_train.reward[:n_transitions] = rewards_flat.cpu().numpy()[:n_transitions]
    data_buffer_train.action[:n_transitions] = latent_actions_flat.cpu().numpy()[:n_transitions]
    data_buffer_train.not_done[:n_transitions] = 1-done_flat.cpu().numpy()[:n_transitions]
    data_buffer_train.ptr = n_transitions
    data_buffer_train.crt_size = n_transitions
    return data_buffer_train

        
    
#########  
#     @staticmethod
#     def get_oracle_dynamics_with_sim(dac_simulation_helper):
#         class OracleDynamicsWithSim():
#             """_summary_
#             sa_repr_model wrapper defined using the provided repr_net. 
#             repr_net must have encode function that takes sates and actions as input. 
#             """

#             def __init__(self, simulate_simulator_step, simulator_env_count):
#                 self.simulate_simulator_step = simulate_simulator_step
#                 self.simulator_env_count = simulator_env_count
#                 # self.dynamics_net = dynamics_net
#                 # self.device = self.dynamics_net.device

#             @torch.no_grad()    
#             def encode_states(self, states, out_device="cpu"):
#                 assert False, "Not Implemented Error"

#             @torch.no_grad()    
#             def encode_state_action_pair(self, state: torch.Tensor, action: torch.Tensor) -> torch.tensor:
#                 assert False, "Not Implemented Error"

#             @torch.no_grad()    
#             def encode_states(self, states, out_device="cpu"):
#                 return states.to(out_device)

#             @torch.no_grad()    
#             def encode_state_action_pairs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.tensor:
#                 assert isinstance(states, torch.Tensor) and isinstance(actions, torch.Tensor)
                
                
#                 states, actions = states.to("cuda"), actions.to("cuda")
#                 pred_next_states =  dac_utils.v_map(fxn=lambda indxs: self.simulate_simulator_step(states[indxs], actions[indxs]),
#                                  iterable= range(len(states)),
#                                  batch_size=self.simulator_env_count,
#                                  reduce_fxn=lambda x: torch.cat(x, dim=0),
#                                  verbose = False).cuda()
#                 return pred_next_states.detach().cpu()
            
#         return OracleDynamicsWithSim(dac_simulation_helper.simulate_simulator_step, dac_simulation_helper.llc_player.env.task.num_envs)


# from dacmdp.core.utils_knn import THelper
# from collections import defaultdict 

# class DACSimluationHelper(object):
#     def __init__(self, llc_player, internal_buffer_logger, data_buffer, step_repeat = 1):
#         self.step_repeat = 1
#         self.llc_player = llc_player
#         self.internal_buffer_logger = internal_buffer_logger
#         obs_by_threads = torch.stack([o['AMP Observation Buffer'] for o in internal_buffer_logger.raw_internal_state_buffer])
#         nn,ee,s_dim = obs_by_threads.shape
#         self.obs_library = obs_by_threads.reshape(nn*ee, s_dim).cuda()
#         self.logger_num_envs = ee
#         self.ref_latent_action = self.llc_player.model.a2c_network.sample_latents(n = self.llc_player.env.task.num_envs).clone()
#         self.ref_internal_state = self._get_ref_internal_state(self.llc_player)
#         print("logger_num_envs", self.logger_num_envs)
        
#     def _get_ref_internal_state(self, p):
#         num_envs =  p.env.task.num_envs
#         _dof_state = gymtorch.wrap_tensor(p.env.task.gym.acquire_dof_state_tensor(p.env.task.sim)).clone().cpu()
#         _actor_root = gymtorch.wrap_tensor(p.env.task.gym.acquire_actor_root_state_tensor(p.env.task.sim)).clone().cpu()
#         dofs_per_env = _dof_state.shape[0] // num_envs
        
#         return {"root_pos":_actor_root[:,0:3], 
#             "root_rot":_actor_root[:,3:7],
#             "root_vel":_actor_root[:,7:10],
#             "root_ang_vel":_actor_root[:,10:13], 
#             "dof_pos":_dof_state.view(num_envs, dofs_per_env, 2)[..., :dofs_per_env, 0],
#             "dof_vel":_dof_state.view(num_envs, dofs_per_env, 2)[..., :dofs_per_env, 1]}
    
#     def find_buffer_indices_batch(self, observations):
#         nn_indexes, nn_distances = THelper.batch_calc_knn_pykeops(observations, self.obs_library, k = 1)
#         nn_indexes = nn_indexes.reshape(-1).cpu().numpy()
#         return nn_indexes//self.logger_num_envs, nn_indexes%self.logger_num_envs 
  
#     def simulate_simulator_step(self, q_observations, q_latent_actions, return_obs = True):
#         " q_observation must be the amp observation. "
        
#         self.llc_player.env.reset()
#         num_envs = self.llc_player.env.task.num_envs
#         n_query= len(q_observations)
#         # print("n_query", n_query)
        
#         latent_actions = self.ref_latent_action.cuda()
#         latent_actions[:n_query] = q_latent_actions
        
#         state_offsets, env_offsets = self.find_buffer_indices_batch(q_observations)    
#         internal_state = self.ref_internal_state
#         # print(state_offsets, env_offsets)
#         # important variables 
#         internal_state = self.ref_internal_state # seed internal state 
#         to_set_tags = ["root_pos", "root_rot","root_vel","root_ang_vel", "dof_pos","dof_vel"]
#         for i, (s_o, e_o) in enumerate(zip(state_offsets, env_offsets)):
#             for tag in to_set_tags:
#                 internal_state[tag][i] =  self.internal_buffer_logger.raw_internal_state_buffer[s_o][tag][e_o].clone()            

#         c_set_env_state(self.llc_player.env.task, 
#                         env_ids = range(num_envs), 
#                         root_pos = internal_state['root_pos'], 
#                         root_rot = internal_state['root_rot'], 
#                         root_val = internal_state['root_val'], 
#                         root_ang_vel = internal_state['root_ang_vel'],
#                         dof_pos = internal_state['dof_pos'],
#                         dof_vel = internal_state['dof_vel'],)
    
#         for _ in range(self.step_repeat):
#             self.llc_player._ase_latents = latent_actions
#             self.llc_player._latent_step_count = 100 # setting it to large number so that ase latent is not update in get action function
#             action = self.llc_player.get_action({'obs':internal_state["Observation Buffer"]}, 
#                                                 is_determenistic=self.llc_player.is_determenistic)
#             _ = self.llc_player.env_step(self.llc_player.env, action)

#         return  self._get_ref_internal_state(self.llc_player)
      