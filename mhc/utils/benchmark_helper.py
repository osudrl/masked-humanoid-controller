# Imports

# Env/Task Imports
from env.tasks.humanoid_im import HumanoidAMPRobust

# Python Imports
import torch
import numpy as np
import copy
from copy import deepcopy


# --------------------------------------------
# ----------Benchmark Helper -----------------
# --------------------------------------------

def generate_cache_for_env_vars(task):
    CACHE = {}
    # Variables that are set during benchmarking code
    CACHE["_state_init"] = deepcopy(task._state_init)
    CACHE["_demo_init"] = deepcopy(task._demo_init)
    CACHE["_state_init_rotate"] = deepcopy(task._state_init_rotate)
    CACHE["_state_demo_rotate"] = deepcopy(task._state_demo_rotate)
    CACHE["_demo_motion_weights"] = deepcopy(task._demo_motion_weights)
    CACHE["_start_state_motion_weights"] = deepcopy(task._start_state_motion_weights)
    CACHE["_fall_init_prob"] = deepcopy(task._fall_init_prob)
    CACHE["_start_demo_at_agent_state_prob"] = deepcopy(task._start_demo_at_agent_state_prob)
    CACHE["_uniform_targets"] = deepcopy(task._uniform_targets)
    CACHE["_switch_demos_within_episode"] = deepcopy(task._switch_demos_within_episode)
    CACHE["_demo_tar_reset_steps_max"] = deepcopy(task._demo_tar_reset_steps_max)
    CACHE["_demo_tar_reset_steps_min"] = deepcopy(task._demo_tar_reset_steps_min)
    CACHE["_enable_early_termination"] = deepcopy(task._enable_early_termination)
    CACHE["_random_dir_speed_env_idxs"] = deepcopy(task._random_dir_speed_env_idxs)
    CACHE["_random_dir_speed_upper_body_env_idxs"] = deepcopy(task._random_dir_speed_upper_body_env_idxs)
    CACHE["_compose_demo_targets_env_idxs"] = deepcopy(task._compose_demo_targets_env_idxs)
    CACHE["max_episode_length"] = deepcopy(task.max_episode_length)
    CACHE["_enable_lookahead_mask"] = deepcopy(task._enable_lookahead_mask)
    CACHE["_enable_lk_jl_mask"] =  deepcopy(task._enable_lk_jl_mask)
    CACHE["_enable_lk_channel_mask"] =  deepcopy(task._enable_lk_channel_mask)
    CACHE["_use_predefined_jl_mask"] = deepcopy(task._use_predefined_jl_mask)
    CACHE["_predefined_jl_mask_joint_prob"] = deepcopy(task._predefined_jl_mask_joint_prob)
    CACHE["_jl_mask_prob"] = deepcopy(task._jl_mask_prob)
    CACHE["lookahead_mask_pool_sample_weights"] = deepcopy(task.lookahead_mask_pool_sample_weights)

    # Add additional variables here if any are set in the visualize method
    CACHE["_use_predefined_targets"] = deepcopy(task._use_predefined_targets)
    CACHE["_random_dir_speed_lookahead_share"] = deepcopy(task._random_dir_speed_lookahead_share)
    CACHE["_random_dir_speed_mimic_upper_prob"] = deepcopy(task._random_dir_speed_mimic_upper_prob)
    CACHE["_compose_demo_targets_share"] = deepcopy(task._compose_demo_targets_share)
    try:
        CACHE["show_traj"] = deepcopy(task.show_traj)
        CACHE["show_headings"] = deepcopy(task.show_headings)
        CACHE["show_root_traj"] = deepcopy(task.show_root_traj)
    except:
        print("show_traj, show_headings, show_root_traj not set in task")
    return CACHE


def restore_env_vars_from_cache(task, CACHE):
    task._state_init = CACHE["_state_init"]
    task._demo_init = CACHE["_demo_init"]
    task._state_init_rotate = CACHE["_state_init_rotate"]
    task._state_demo_rotate = CACHE["_state_demo_rotate"]
    task._demo_motion_weights = CACHE["_demo_motion_weights"].clone()
    task._start_state_motion_weights = CACHE["_start_state_motion_weights"].clone()
    task._fall_init_prob = CACHE["_fall_init_prob"]
    task._start_demo_at_agent_state_prob = CACHE["_start_demo_at_agent_state_prob"]
    task._uniform_targets = CACHE["_uniform_targets"]
    task._switch_demos_within_episode = CACHE["_switch_demos_within_episode"]
    task._demo_tar_reset_steps_max = CACHE["_demo_tar_reset_steps_max"]
    task._demo_tar_reset_steps_min = CACHE["_demo_tar_reset_steps_min"]
    task._enable_early_termination = CACHE["_enable_early_termination"]
    task._random_dir_speed_env_idxs = CACHE["_random_dir_speed_env_idxs"].clone()
    task._random_dir_speed_upper_body_env_idxs = CACHE["_random_dir_speed_upper_body_env_idxs"].clone()
    task._compose_demo_targets_env_idxs = CACHE["_compose_demo_targets_env_idxs"].clone()
    task.max_episode_length = CACHE["max_episode_length"]
    task._enable_lookahead_mask = CACHE["_enable_lookahead_mask"]
    task._enable_lk_jl_mask =  CACHE["_enable_lk_jl_mask"]
    task._enable_lk_channel_mask =  CACHE["_enable_lk_channel_mask"]
    task._use_predefined_jl_mask = CACHE["_use_predefined_jl_mask"]
    task._predefined_jl_mask_joint_prob = CACHE["_predefined_jl_mask_joint_prob"]
    task._jl_mask_prob = CACHE["_jl_mask_prob"]
    task.lookahead_mask_pool_sample_weights = CACHE["lookahead_mask_pool_sample_weights"].clone()

    # Restore additional variables here if any are set in the visualize method
    task._use_predefined_targets = CACHE["_use_predefined_targets"]
    task._random_dir_speed_lookahead_share = CACHE["_random_dir_speed_lookahead_share"]
    task._random_dir_speed_mimic_upper_prob = CACHE["_random_dir_speed_mimic_upper_prob"]
    task._compose_demo_targets_share = CACHE["_compose_demo_targets_share"]

    try:
        task.show_traj = CACHE["show_traj"]
        task.show_headings = CACHE["show_headings"]
        task.show_root_traj = CACHE["show_root_traj"]
    except:
        print("show_traj, show_headings, show_root_traj not set in task")
    return task


def benchmark_policy(self, 
                    benchmark_config, 
                    write_to_file = False,
                    label = "_", 
                    verbose = False, 
                    ):


    horizon = benchmark_config.get("horizon", 350)
    mode = benchmark_config.get( "mode", "mimicry") # "mimicry" or "catchup" or "compose" or "complete" 
    honor_mask_for_rewards = benchmark_config.get("honor_mask_for_rewards", False)
    success_criteria = benchmark_config.get("success_criteria", "shifted-mpjpe")
    num_buckets = benchmark_config.get("num_buckets", 1)
    
    if hasattr(self, "vec_env"):
        self.env = self.vec_env.env

    def reset_play_step_vars():
        if hasattr(self, "current_rewards"):
            self.current_rewards[:] = 0
            self.current_penalties[:] = 0
            self.current_lengths[:] = 0
            self.dones[:] = 1
            self.env_reset(torch.arange(self.env.task.num_envs).cuda())

    def setup_env_for_benchmarking():
        self.env.task._state_init = HumanoidAMPRobust.StateInit.Hybrid if mode=="catchup" else HumanoidAMPRobust.StateInit.Start 
        self.env.task._demo_init = HumanoidAMPRobust.StateInit.Hybrid if mode=="catchup" else HumanoidAMPRobust.StateInit.Start 
        self.env.task._state_init_rotate = True if mode=="catchup" else False
        self.env.task._state_demo_rotate = True if mode=="catchup" else False

        self.env.task._demo_motion_weights[:] = 1/len(self.env.task._demo_motion_weights)
        self.env.task._start_state_motion_weights[:] = 1/len(self.env.task._start_state_motion_weights)

        self.env.task._fall_init_prob = 0.1 if mode=="catchup" else 0
        self.env.task._start_demo_at_agent_state_prob = 0

        self.env.task._uniform_targets = False
        self.env.task._switch_demos_within_episode = True if mode == "catchup" else False
        self.env.task._demo_tar_reset_steps_max = horizon//2 + 2
        self.env.task._demo_tar_reset_steps_min = horizon//2 + 1

        self.env.task._enable_early_termination = False
        self.env.task._random_dir_speed_env_idxs = torch.LongTensor([]).cuda()
        self.env.task._random_dir_speed_upper_body_env_idxs = torch.LongTensor([]).cuda()
        self.env.task._compose_demo_targets_env_idxs = torch.arange(self.env.task.num_envs-1).cuda() if mode == "compose" else torch.LongTensor([]).cuda() # -2 because last env cannot be composed with next
        self.env.task.max_episode_length = horizon*2 # to make sure the env is reset

        self.env.task._enable_lookahead_mask = True if mode == "complete" else False
        self.env.task._enable_lk_jl_mask =  True if mode == "complete" else False
        self.env.task._enable_lk_channel_mask = True if mode == "complete" else False
        self.env.task._jl_mask_prob = 1 if mode == "complete" else 0
        self.env.task._use_predefined_jl_mask = True
        self.env.task._predefined_jl_mask_joint_prob = 0 # will be set later

    def get_to_track_joint_ds_dict(mode):
        # Get To track joint ids
        if mode == "complete":
            to_track_joint_ids_dict = {
                                        "Root Obs + Joint Angle + Joint XYZ ": {"lookahead_mask_indx": 0, 
                                                                "jl_mask_joint_prob": 0},
                                        "Root Obs + Joint Angle":  {"lookahead_mask_indx": 2,  
                                                                "jl_mask_joint_prob": 0},
                                        "Joint GLOBAL XYZ (0)": {"lookahead_mask_indx": 3, 
                                                                "jl_mask_joint_prob": 0},
                                        "Joint GLOBAL XYZ (0.25)": {"lookahead_mask_indx": 3, 
                                                                "jl_mask_joint_prob": 0.25},
                                        "Joint GLOBAL XYZ (0.5)": {"lookahead_mask_indx": 3, 
                                                                "jl_mask_joint_prob": 0.5},
                                        "Joint GLOBAL XYZ (0.75)": {"lookahead_mask_indx": 3, 
                                                                "jl_mask_joint_prob": 0.75},
                                        "Global Joint Key XYZ": {"lookahead_mask_indx": 4,
                                                                "jl_mask_joint_prob": 0},
                                        "Joint LOCAL XYZ (0)": {"lookahead_mask_indx": 5, 
                                                                "jl_mask_joint_prob": 0},
                                        "Joint LOCAL XYZ (0.25)": {"lookahead_mask_indx": 5,
                                                                "jl_mask_joint_prob": 0.25},
                                        "Joint LOCAL XYZ (0.5)": {"lookahead_mask_indx": 5,
                                                                "jl_mask_joint_prob": 0.5},
                                        "Joint LOCAL XYZ (0.75)": {"lookahead_mask_indx": 5,
                                                                "jl_mask_joint_prob": 0.75},
                                        "Local Joint Key XYZ": {"lookahead_mask_indx": 6,
                                                                "jl_mask_joint_prob": 0},
                                        "Root Obs": {"lookahead_mask_indx": 1, 
                                                                "jl_mask_joint_prob": 0},
                                    }
        elif mode == "compose":
            to_track_joint_ids_dict = {"Root Obs + Joint XYZ ": {"lookahead_mask_indx": 5, # compose compatibel lookahead index
                                                                "jl_mask_joint_prob": 0},
                                    }
        else:
            to_track_joint_ids_dict = {"Root Obs + Joint Angle + Joint XYZ ": {"lookahead_mask_indx": 0, 
                                                                "jl_mask_joint_prob": 0},
                                    }
                                           
        return to_track_joint_ids_dict

    def initialize_benchmark_buffers(env, horizon, num_buckets = 1, shuffle = False):
        """
        Initializes benchmark buffers for storing various metrics during the benchmarking process.

        Parameters:
        env (YourEnvClass): The environment object that contains task and motion library information.
        horizon (int): The horizon length for the benchmarking process.

        Returns:
        dict: A dictionary containing initialized benchmark buffers.
        """
        num_envs = env.num_envs
        num_motions = env.task._motion_lib.num_motions()
        motion_lengths = env.task._motion_max_steps.cpu().numpy().tolist()
        episode_length = horizon

        all_to_set_m_ids = []
        all_to_set_start_ts = []
        for m_id in range(num_motions):
            motion_len = motion_lengths[m_id]
            start_ts = torch.arange(0, motion_len, episode_length).numpy().tolist()
            all_to_set_start_ts.extend(start_ts)
            all_to_set_m_ids.extend([m_id]*len(start_ts))

        # num_buckets = int(np.ceil(len(all_to_set_m_ids)/num_envs))
        assert num_buckets == 1 , "Only one bucket is supported for now"
        if shuffle: 
            # Shuffle with a fixed seed 
            np.random.seed(42) # 44 , 48
            # np.random.seed(100) # 44 , 48
            shuffled_idxs = np.arange(len(all_to_set_m_ids))
            np.random.shuffle(shuffled_idxs)
            all_to_set_m_ids = copy.deepcopy(np.array(all_to_set_m_ids)[shuffled_idxs].reshape(-1))
            all_to_set_start_ts = copy.deepcopy(np.array(all_to_set_start_ts)[shuffled_idxs].reshape(-1))
            all_to_set_m_ids = all_to_set_m_ids.tolist()
            all_to_set_start_ts = all_to_set_start_ts.tolist()
            print(all_to_set_m_ids)
        # print(f"num_buckets={num_buckets}")

        # Counter Buckets
        to_track_metrics = ["reward", "MPJPE-local", "MPJPE-global", "MPJPE-shifted", "root_pos_error", "root_vel_error", 
                            "failure@1000", "failure@500", "disc_prob",]
        benchmark_buffer = { m: torch.zeros((num_buckets, num_envs, episode_length)) 
                                        for m in to_track_metrics}
        benchmark_demo_id_buffer = torch.randint(0, num_motions, (num_buckets, num_envs))
        benchmark_demo_start_timestep_buffer = torch.randint(0, episode_length, (num_buckets, num_envs)) 

        # Combine all buffers into a single dictionary
        combined_buffers = {
            "benchmark_buffer": benchmark_buffer,
            "demo_id_buffer": benchmark_demo_id_buffer,
            "start_timestep_buffer": benchmark_demo_start_timestep_buffer
        }


        num_required_threads = len(all_to_set_m_ids)
        for bucket_id in range(num_buckets):
            for env_thread_id in range(num_envs):
                thread_id = (bucket_id * num_envs + env_thread_id)%num_required_threads
                m_id = all_to_set_m_ids[thread_id]
                start_ts = all_to_set_start_ts[thread_id]
                benchmark_demo_id_buffer[bucket_id, env_thread_id] = m_id
                benchmark_demo_start_timestep_buffer[bucket_id, env_thread_id] = start_ts

        return combined_buffers

    cached_vars = generate_cache_for_env_vars(self.env.task)
    setup_env_for_benchmarking()
    to_track_joint_ids_dict = get_to_track_joint_ds_dict(mode)

    main_table = [f"{label}", "-"*200, 
                    # "| {:^40} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} |{:^20} | "\
                    "| {:^40} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} |  {:^20} |"\
                    .format("Tag", "MPJPE-local","MPJPE-global", "MPJPE-shifted", "Mean Root-XYZ-Error", "Mean Root-Vel-Error", "Mean Success Rate", 
                            # "Mean Success Rate @ 1000", "Mean Success Rate @ 500", 
                            "Mean Prob Disc"), "-"*200]
    
    if verbose:
        print(" \n ".join(main_table))

    # Run Benchmarking
    modality_tables = []
    for tag, lk_dict in to_track_joint_ids_dict.items():
        
        lk_idx = lk_dict["lookahead_mask_indx"]
        if mode == "compose":
            self.env.task.lookahead_mask_pool_compose_compatible_sample_weights[:] = 0
            self.env.task.lookahead_mask_pool_compose_compatible_sample_weights[lk_idx] = 1
        self.env.task.lookahead_mask_pool_sample_weights[:] = 0
        self.env.task.lookahead_mask_pool_sample_weights[lk_idx] = 1
        self.env.task._predefined_jl_mask_joint_prob = lk_dict["jl_mask_joint_prob"]
        self.env.task._enable_lk_jl_mask = True
        self.env.task._enable_lk_channel_mask = True
        self.env.task._enable_lookahead_mask = True

        # total_len = np.sum(motion_lengths).item()
        # print(f"Total Motion Length : {total_len}")
        benchmark_buffer, benchmark_demo_id_buffer, benchmark_demo_start_timestep_buffer = initialize_benchmark_buffers(self.env, horizon,
                                                                                                                        shuffle=True if mode == "compose"
                                                                                                                        else False).values()


        # Populate the reward buffer
        num_buckets = 1
        for bucket_id in range(num_buckets):
            obs_dict = self.env_reset(torch.arange(self.env.task.num_envs).cuda())
            demo_tar_m_ids = benchmark_demo_id_buffer[bucket_id]
            demo_tar_m_ts = benchmark_demo_start_timestep_buffer[bucket_id]
            demo_tar_m_time = self.env.task._motion_lib._motion_dt[demo_tar_m_ids]*demo_tar_m_ts
            # Reset based on the target m id and time step
            self.env.task._reset_envs_with_manual_targets(self.env.task._all_env_ids, demo_tar_m_ids, demo_tar_m_time)

            done_indices = []
            # print("tag", torch.sum(self.env.task._global_demo_lookahead_mask, dim = -1))
            for n in range(horizon):
                obs_dict = self.env_reset(done_indices)

                if hasattr(self, "get_action_values"):
                    action = self.get_action_values(obs_dict, torch.zeros_like(self._rand_action_probs))['actions']
                    obs_dict_next, r, done, info =  self.env_step(action)
                else:
                    action = self.get_action(obs_dict, is_determenistic=True)
                    obs_dict_next, r, done, info =  self.env_step(self.env, action)

                reward_dict = compute_benchmark_reward_using_full_body_state(full_body_state=self.env.task._curr_full_state_obs_buf , 
                                                                        tar_body_state=self.env.task.extras["prev_tar_full_state_obs"][:,:self.env.task._full_state_obs_dim],
                                                                        lookahead_obs=self.env.task.extras["prev_tar_flat_lookahead_obs"][:, :self.env.task._lookahead_obs_dim],
                                                                        lookahead_obs_mask= self.env.task.extras["prev_tar_lookahead_obs_mask"][:, :self.env.task._lookahead_obs_dim],
                                                                        dof_obs_size=self.env.task._dof_obs_size,
                                                                        num_dofs=self.env.task._dof_offsets[-1],
                                                                        num_bodies=self.env.task.num_bodies, 
                                                                        key_body_ids=self.env.task._key_body_ids,
                                                                        ret_info = True,
                                                                        lookahead_obs_split_indxs_dict = self.env.task._lookahead_obs_split_indxs,
                                                                        honor_mask_for_rewards = honor_mask_for_rewards) # Added for Imitation

                disc_logits = self._eval_disc(amp_obs = info["amp_obs"], 
                                                amp_obs_mask = self._reshape_amp_obs_mask(info["amp_obs"], 
                                                                                torch.zeros(self.env.task._amp_obs_dim).type(torch.bool).cuda())
                                            )
                prob =  1 / (1 + torch.exp(-disc_logits))

                benchmark_buffer["reward"][bucket_id, :, n] = r.view(-1)
                benchmark_buffer["disc_prob"][bucket_id, :, n] = prob.view(-1)
                benchmark_buffer["MPJPE-shifted"][bucket_id, :, n]  = reward_dict["mean_joint_dist_shifted"].view(-1)
                benchmark_buffer["MPJPE-local"][bucket_id, :, n]  = reward_dict["mean_joint_dist_local"].view(-1)
                benchmark_buffer["MPJPE-global"][bucket_id, :, n]  = reward_dict["mean_joint_dist_global"].view(-1)
                benchmark_buffer["root_pos_error"][bucket_id, :, n] = reward_dict["root_dist_global"].view(-1)
                benchmark_buffer["root_vel_error"][bucket_id, :, n] = reward_dict["root_vel_error"].view(-1)

                if success_criteria == "shifted-mpjpe":
                    benchmark_buffer["failure@1000"][bucket_id, :, n] = reward_dict["max_joint_dist_shifted"].view(-1) > 1000
                    benchmark_buffer["failure@500"][bucket_id, :, n] = reward_dict["max_joint_dist_shifted"].view(-1) > 850
                else:
                    benchmark_buffer["failure@1000"][bucket_id, :, n] = reward_dict["max_joint_dist_local"].view(-1) > 1000
                    benchmark_buffer["failure@500"][bucket_id, :, n] = reward_dict["max_joint_dist_shifted"].view(-1) > 850


                # if not catchup:
                #     assert (torch.sum(benchmark_demo_id_buffer[bucket_id, :]).item() - torch.sum(self.env.task._global_demo_start_motion_ids).item())  == 0

                # self._post_step(info)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                done_indices = done_indices[:, 0]

                # if not catchup:
                #     assert len(done_indices) == 0 or len(done_indices) == self.env.task.num_envs


        # Calculate Mean Scores. 
        modality_table = []
        all_local_errors  = []
        all_global_errors = []
        all_shifted_errors = []
        all_local_root_shifts = []
        all_root_vel_errors = []
        all_total_failures_at_1000 = []
        all_total_failures_at_500 = []
        all_disc_prob =[]
        all_motion_success_at_1000 = {}
        all_motion_success_at_500 = {}

        benchmark_motion_list = benchmark_demo_id_buffer[bucket_id]
        for i in benchmark_motion_list:
            # benchmark_stop_step = self.env.task._motion_max_steps[i]-1
            # total_allowed_failures = max(10, int(self.env.task._motion_max_steps[i]*0.05))
            if mode == "catchup":
                to_benchmark_idxs = torch.arange(1, horizon-1)
                success_benchmark_idxs = torch.cat((torch.arange(1, horizon//2), torch.arange(horizon//2+60, horizon-1)))
                total_allowed_failures = max(10, int(len(success_benchmark_idxs)*0.1))
            else:
                to_benchmark_idxs = torch.arange(1, self.env.task._motion_max_steps[i]-1)
                success_benchmark_idxs = torch.arange(1, self.env.task._motion_max_steps[i]-1)
                total_allowed_failures = max(10, int(len(success_benchmark_idxs)*0.1))


            name = self.env.task._motion_lib.demo_store["demo_names"][i].replace('RL_Avatar_','')
            motion_mean_shifted_error = torch.mean(benchmark_buffer["MPJPE-shifted"][0,i, to_benchmark_idxs]).item()
            motion_mean_global_error = torch.mean(benchmark_buffer["MPJPE-global"][0,i, to_benchmark_idxs]).item()
            motion_mean_local_error = torch.mean(benchmark_buffer["MPJPE-local"][0,i,  to_benchmark_idxs]).item()
            motion_mean_root_pos_error = torch.mean(benchmark_buffer["root_pos_error"][0,i, to_benchmark_idxs]).item()
            motion_mean_root_vel_error = torch.mean(benchmark_buffer["root_vel_error"][0,i, to_benchmark_idxs]).item()
            motion_total_failures_at_1000 = torch.sum(benchmark_buffer["failure@1000"][0,i, success_benchmark_idxs]).item()
            motion_total_failures_at_500 = torch.sum(benchmark_buffer["failure@500"][0,i, success_benchmark_idxs]).item()
            motion_disc_prob = torch.mean(benchmark_buffer["disc_prob"][0,i, to_benchmark_idxs]).item()

            all_shifted_errors.append(motion_mean_shifted_error)
            all_global_errors.append(motion_mean_global_error)
            all_local_errors.append(motion_mean_local_error)
            all_local_root_shifts.append(motion_mean_root_pos_error)
            all_root_vel_errors.append(motion_mean_root_vel_error)
            all_disc_prob.append(motion_disc_prob)
            
            all_total_failures_at_1000.append(motion_total_failures_at_1000>total_allowed_failures)
            all_total_failures_at_500.append(motion_total_failures_at_500>total_allowed_failures)

            all_motion_success_at_1000[i] = 1 - (motion_total_failures_at_1000>total_allowed_failures)
            all_motion_success_at_500[i] = 1 - (motion_total_failures_at_500>total_allowed_failures)
            modality_table.append(f"{i=}, {name=},{motion_mean_local_error=:.2f}, {motion_mean_shifted_error=:.2f}," +\
                    f"{motion_mean_global_error=:.2f}, {motion_mean_root_pos_error=:.2f}, {motion_total_failures_at_1000=}, {motion_total_failures_at_500=}")

        main_table.append("| {:^40} | {:^20.2f} | {:^20.2f} | {:^20.2f} | {:^20.2f} | {:^20.2f} | {:^20.2f} | {:^20.2f} |".format(tag, 
                                                                                                                            np.mean(all_local_errors),
                                                                                                                            np.mean(all_global_errors),
                                                                                                                            np.mean(all_shifted_errors),
                                                                                                                            np.mean(all_local_root_shifts), 
                                                                                                                            np.mean(all_root_vel_errors),
                                                                                                                            1 - np.mean(all_total_failures_at_1000), 
                                                                                                                            # 1 - np.mean(all_total_failures_at_1000), 
                                                                                                                            # 1 - np.mean(all_total_failures_at_500), 
                                                                                                                            np.mean(all_disc_prob)))
        if verbose:
            print(main_table[-1])
        
        modality_tables.append(modality_table)
    # Close file after done  
    main_table.append("-"*200)

    if write_to_file:
        with open(f'result_logs/results_{label}.txt', 'w', newline='') as results_file:
            for modality_table in modality_tables:
                results_file.write("\n\n\n\n")
                for m in modality_table:
                    results_file.write(m + "\n")

            results_file.write("\n\n\n\n")
            for m in main_table:
                results_file.write(m + "\n")

    # Restore Env Variables
    restore_env_vars_from_cache(self.env.task, cached_vars)
    reset_play_step_vars()

    return all_motion_success_at_1000 #benchmark_buffer["reward"], benchmark_demo_id_buffer, benchmark_demo_start_timestep_buffer
