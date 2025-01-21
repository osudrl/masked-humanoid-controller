# Imports

# Python Imports
import torch
import torch.nn as nn
import numpy as np
import wandb
from types import MethodType

# AMP Imports
from learning.amp_agent import AMPAgent

# RL Games Imports
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

# Utils Imports
from utils.benchmark_helper import benchmark_policy


# --------------------------------------------
# ------------------train---------------------
# --------------------------------------------

class AMPAgent2(AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self._energy_penalty_activated = self.vec_env.env.task._energy_penalty_activated
        self.benchmark_policy = MethodType(benchmark_policy, self)
    
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

    def localize_lookahead_obs(self, lookahead_obs_flat_batch, lookahead_obs_mask):
        return self.vec_env.env.task.localize_lookahead_obs(lookahead_obs_flat_batch, 
                                                            num_stacks= self.vec_env.env.task._lookahead_timesteps, 
                                                            lookahead_obs_mask = lookahead_obs_mask)

    def init_tensors(self):
        super().init_tensors()
        self.game_penalties = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.current_penalties = torch.zeros_like(self.current_rewards)

    def _build_amp_buffers(self):
        super()._build_amp_buffers()

        batch_shape = self.experience_buffer.obs_base_shape
        lookahead_tar_obs_space = (self.vec_env.env.task.lookahead_tar_flat_dim,)
        lookahead_tar_mask_space = (self.vec_env.env.task._lookahead_obs_dim,)
        amp_obs_mask_space = (self.vec_env.env.task._amp_obs_dim,)
        self.experience_buffer.tensor_dict['lookahead_tar_obs'] = torch.zeros(batch_shape + lookahead_tar_obs_space, device=self.ppo_device)
        self.experience_buffer.tensor_dict['energy_penalty'] = torch.zeros_like(torch.zeros_like(self.experience_buffer.tensor_dict['rewards']))
        self.experience_buffer.tensor_dict['lookahead_tar_mask'] = torch.zeros(batch_shape + lookahead_tar_mask_space, device=self.ppo_device).type(torch.bool)
        self.experience_buffer.tensor_dict['amp_obs_mask'] = torch.zeros(batch_shape + amp_obs_mask_space, device=self.ppo_device).type(torch.bool)
        self.tensor_list += ['lookahead_tar_obs', 'lookahead_tar_mask', 'amp_obs_mask', 'energy_penalty']

    def obs_to_tensors(self, obs):
        obs_dict = super().obs_to_tensors(obs)
        obs_dict['lookahead_tar_obs'] = self.vec_env.env.task.lookahead_tar_flat_obs
        obs_dict['lookahead_tar_mask'] = self.vec_env.env.task._global_demo_lookahead_mask
        obs_dict['amp_obs_mask'] = self.vec_env.env.task._global_amp_obs_mask
        return obs_dict

    def play_steps(self):
        self.set_eval()

        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):

            self.obs = self.env_reset(done_indices)
            # self.obs['lookahead_tar_obs'] = self.vec_env.env.task.lookahead_tar_flat_obs  # AddtionalCodeLine
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('lookahead_tar_obs', n, self.obs['lookahead_tar_obs']) # AddtionalCodeLine
            self.experience_buffer.update_data('lookahead_tar_mask', n, self.obs['lookahead_tar_mask']) # AddtionalCodeLine
            self.experience_buffer.update_data('amp_obs_mask', n, self.obs['amp_obs_mask']) # AddtionalCodeLine

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            # self.obs['lookahead_tar_obs'] = self.vec_env.env.task.lookahead_tar_flat_obs  # AddtionalCodeLine
            
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('energy_penalty', n, infos['energy_penalty']) # AddtionalCodeLine
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_penalties += infos['energy_penalty'] # AddtionalCodeLine
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_penalties.update(self.current_penalties[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_penalties = self.current_penalties * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)
                
            done_indices = done_indices[:, 0]
            if len(done_indices) > 0:
                _ = self.env_reset(done_indices)

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        # mb_amp_obs_mask = self.experience_buffer.tensor_dict['amp_obs_mask']
        mb_energy_penalty = self.experience_buffer.tensor_dict['energy_penalty'] # AddtionalCodeLine
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, self.vec_env.env.task.amp_obs_mask_pool)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        mb_rewards_penalized = mb_rewards + int(self._energy_penalty_activated) * mb_energy_penalty # AddtionalCodeLine

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards_penalized, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict
    
    def get_action_values(self, obs_dict, rand_action_probs):
        batch_size = obs_dict['obs'].shape[0]
        processed_obs = self._preproc_obs(obs_dict['obs'])
        lookahead_tar_obs = self.localize_lookahead_obs(obs_dict['lookahead_tar_obs'], obs_dict['lookahead_tar_mask'])
        lookahead_tar_mask = obs_dict['lookahead_tar_mask']

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states,
            'lookahead_tar_obs': lookahead_tar_obs,
            'lookahead_tar_mask': lookahead_tar_mask
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        
        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask

        return res_dict
    
    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        
        lookahead_tar_obs = batch_dict['lookahead_tar_obs']
        self.dataset.values_dict['lookahead_tar_obs'] = lookahead_tar_obs
        lookahead_tar_mask = batch_dict['lookahead_tar_mask']
        self.dataset.values_dict['lookahead_tar_mask'] = lookahead_tar_mask
        amp_obs_mask = batch_dict['amp_obs_mask']
        self.dataset.values_dict['amp_obs_mask'] = amp_obs_mask
        
        return

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)
        
        lookahead_tar_obs = self.localize_lookahead_obs(input_dict['lookahead_tar_obs'], input_dict['lookahead_tar_mask'])
        lookahead_tar_mask = input_dict['lookahead_tar_mask']
        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)

        amp_obs_mask = input_dict['amp_obs_mask'][0:self._amp_minibatch_size]

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'amp_obs' : amp_obs,
            'amp_obs_replay' : amp_obs_replay,
            'amp_obs_demo' : amp_obs_demo,
            'lookahead_tar_obs': lookahead_tar_obs, #AddtionalCodeLine
            'lookahead_tar_mask': lookahead_tar_mask, #AddtionalCodeLine
            'amp_obs_mask': amp_obs_mask,  #AddtionalCodeLine
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)
            
            c_loss = torch.mean(c_loss)
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum

            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, res_dict["amp_obs_demo_input_tensor"])
            disc_loss = disc_info['disc_loss']

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)

        return

    def _load_config_params(self, config):
        super()._load_config_params(config)

    def _build_net_config(self):
        config = super()._build_net_config()
        config['lookahead_tar_flat_obs_shape'] = (self.vec_env.env.task.lookahead_tar_flat_dim,)
        config['lookahead_tar_mask_shape'] = (self.vec_env.env.task._lookahead_obs_dim,)
        config['amp_mask_shape'] = (self.vec_env.env.task._amp_obs_dim,)
        config['lookahead_timesteps'] = self.vec_env.env.task._lookahead_timesteps
        config['amp_input_shape'] = (self._amp_observation_space.shape[0] + self.vec_env.env.task._amp_obs_dim, )
        config['lookahead_factor_dims'] = self.vec_env.env.task._lookahead_factor_dims
        return config
    
    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict['obs']
        processed_obs = self._preproc_obs(obs)
        lookahead_tar_obs = self.localize_lookahead_obs(obs_dict['lookahead_tar_obs'], obs_dict['lookahead_tar_mask'])
        lookahead_tar_mask = obs_dict['lookahead_tar_mask']
        obs_w_embed = self.model.a2c_network.add_embedding_to_obs(processed_obs, lookahead_tar_obs, lookahead_tar_mask)
        value = self.model.a2c_network.eval_critic(obs_w_embed)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _preproc_obs(self, obs_batch):
        lk_embedding_dim = self.vec_env.env.task.lk_embedding_dim
        new_obs_batch = super()._preproc_obs(obs_batch)
        new_obs_batch[...,:lk_embedding_dim] = obs_batch[...,:lk_embedding_dim] # dont change the indexes        
        return new_obs_batch
    
    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
                
        if self.game_penalties.current_size > 0:
            mean_penalties = self.game_penalties.get_mean()
            for i in range(self.value_size):
                self.writer.add_scalar('penalties{0}/frame'.format(i), mean_penalties[i], frame)
        return

    def _eval_disc(self, amp_obs, amp_obs_mask):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs, amp_obs_mask)
    
    def _calc_amp_rewards(self, amp_obs, amp_obs_mask_pool):
        disc_r = self._calc_disc_rewards(amp_obs, amp_obs_mask_pool)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs, amp_obs_mask_pool):
        with torch.no_grad():
            disc_r_pool = []
            for amp_obs_mask in amp_obs_mask_pool:
                amp_obs_mask_reshaped = self._reshape_amp_obs_mask(amp_obs, amp_obs_mask)
                disc_logits = self._eval_disc(amp_obs, amp_obs_mask_reshaped)
                prob = 1 / (1 + torch.exp(-disc_logits)) 
                disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
                disc_r *= self._disc_reward_scale
                disc_r_pool.append(disc_r)
            
            disc_r_pool = torch.stack(disc_r_pool)
        return torch.mean(disc_r_pool, dim = 0)
    
    def _reshape_amp_obs_mask(self, amp_obs, amp_obs_mask):
        if len(amp_obs.shape) == 2:
            amp_obs_mask_reshaped = amp_obs_mask.unsqueeze(0).repeat(amp_obs.shape[0], 1)
        elif len(amp_obs.shape) == 3:
            amp_obs_mask_reshaped = amp_obs_mask.unsqueeze(0).unsqueeze(0).repeat(amp_obs.shape[0], amp_obs.shape[1], 1)
        else:
            assert False
        return amp_obs_mask_reshaped

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs_mask = info['amp_obs_mask']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs, amp_obs_mask)
            amp_rewards = self._calc_amp_rewards(amp_obs, self.vec_env.env.task.amp_obs_mask_pool)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return

    def save(self, fn):
        if hasattr(self, "save_counter"):
            self.save_counter += 1
        else:
            self.save_counter = 0

        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

        latest_checkpoint_path =  "/".join(fn.split("/")[:-1]) + "/latest_model"
        torch_ext.save_checkpoint(latest_checkpoint_path, state)
        wandb.save(latest_checkpoint_path + ".pth", base_path = "/".join(latest_checkpoint_path.split("/")[:-1]), policy = "now")

        # Evaluate 
        if self._dynamic_weighting:
            self.log_demo_weights_in_wandb()

        if self.save_counter%10 == 0 and self.benchmark_policy_flag:
            self.benchmark_policy_and_reweight_motions(reweight_motions = self._dynamic_weighting and self.epoch_num > self._dynamic_weighting_start_epoch)
        return
        
    @torch.no_grad()
    def benchmark_policy_and_reweight_motions(self, horizon = 350, reweight_motions = False):
        self.env = self.vec_env.env

        motion_success = self.benchmark_policy(benchmark_config = {"mode": "mimicry", 
                                                                   "horizon": horizon,
                                                                    "success_criteria": "shifted-mpjpe",
                                                                    "honor_mask_for_rewards": False,}, 
                                                write_to_file = False,
                                                label = "_",
                                                verbose = False, )

        if reweight_motions:
            self.reweight_demo_weights_based_on_success(motion_success)
        self.log_demo_weights_in_wandb()
        self.log_success_in_wandb(motion_success)

    def log_demo_weights_in_wandb(self):
        # Make a grayscale wandb image using demo weights
        demo_weights = self.vec_env.env.task._demo_motion_weights
        arr = demo_weights/torch.sum(demo_weights)
        arr = (200/arr.max().item()) * arr
        arr = arr.view(1,-1).repeat(20,1).cpu().numpy().astype(np.uint8)

        # Log as image
        wandb.log({"Motion Demo Sampling Weights": wandb.Image(arr)})

    def log_success_in_wandb(self, motion_success):
        # Normalize to 0-255 range 
        arr = 200 * torch.tensor(list(motion_success.values()))
        arr = arr.view(1,-1).repeat(20,1).cpu().numpy().astype(np.uint8)

        # Log as image
        wandb.log({"Motion Imitation Success": wandb.Image(arr)})
        wandb.log({"Motion Imitation Success Rate": np.mean(list(motion_success.values()))})

    def reweight_demo_weights_based_on_success(self, motion_success):
        # assign half the weights to unsuccessfull trajectories
        # set weights 
        total_success = sum(list(motion_success.values()))
        total_unsuccess = len(motion_success) - total_success

        success_weight = 0.5/total_success
        unsuccess_weight = 0.5/total_unsuccess
        new_demo_motion_weights = self.env.task._demo_motion_weights.clone()
        for m_id, success in motion_success.items():
            new_weight = success_weight if success else unsuccess_weight
            old_weight = self.env.task._demo_motion_weights[m_id]
            new_demo_motion_weights[m_id] = 0.9*old_weight + 0.1*new_weight
            print(f"New Weights for motion id {m_id}:{new_demo_motion_weights[m_id]}")
        # set new weights for only the Normalize Weights
        # Disclaimer , this will override the demo_motion_ids. so all motions will be mimiced from this point onwards.
        self.env.task._demo_motion_weights = new_demo_motion_weights/torch.sum(new_demo_motion_weights)
        return
