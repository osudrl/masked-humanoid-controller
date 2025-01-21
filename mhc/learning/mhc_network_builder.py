# Imports

# Python Imports
import torch
import torch.nn as nn

# AMP Imports
from learning import amp_network_builder


# --------------------------------------------
# -------------Networks Models----------------
# --------------------------------------------

class AMPBuilderV1(amp_network_builder.AMPBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    
    class Network(amp_network_builder.AMPBuilder.Network):
        def __init__(self, params, **kwargs):

            lookahead_tar_flat_obs_shape = kwargs.get("lookahead_tar_flat_obs_shape")
            lookahead_tar_mask_shape = kwargs.get("lookahead_tar_mask_shape")
            amp_mask_shape = kwargs.get('amp_mask_shape')
            self.lookahead_timesteps = kwargs.get('lookahead_timesteps')
            lookahead_factor_dims = kwargs.get('lookahead_factor_dims')
            
            super().__init__(params, **kwargs)

            self.disable_lookahead_mask_in_obs = params.get("disable_lookahead_mask_in_obs")
            self.disable_multi_mask_amp = params.get("disable_multi_mask_amp")
            self.lk_encoder_args = params.get("lk_encoder")

            embedding_dim = self.lk_encoder_args["embedding_dim"] # todo remove this hardcoding
            
            if self.disable_lookahead_mask_in_obs:
                lk_input_dim = lookahead_tar_flat_obs_shape[0]
            else:
                lk_input_dim = lookahead_tar_flat_obs_shape[0] + lookahead_tar_mask_shape[0]
            mlp_dims = (lk_input_dim, *self.lk_encoder_args["units"], embedding_dim)
            mlp_layers = [l for i in range(len(mlp_dims)-1) for l in [nn.Linear(mlp_dims[i], mlp_dims[i+1]), nn.SiLU()]]
            self.lookahead_tar_enc =  nn.Sequential(*mlp_layers)
            self.total_embedding_dim = embedding_dim
            return
        
        def _build_mlp(self, *args, **kwargs):
            self.activations_factory.register_builder('silu', lambda **kwargs : nn.SiLU(**kwargs))
            return super()._build_mlp(*args, **kwargs)
        
        def _build_disc(self, *args, **kwargs):
            self.activations_factory.register_builder('silu', lambda **kwargs : nn.SiLU(**kwargs))
            return super()._build_disc(*args, **kwargs)        

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            lookahead_tar_obs = obs_dict['lookahead_tar_obs']
            lookahead_tar_mask = obs_dict['lookahead_tar_mask']
            states = obs_dict.get('rnn_states', None)

            obs = self.add_embedding_to_obs(obs, lookahead_tar_obs, lookahead_tar_mask)
            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output
        
        def add_embedding_to_obs(self, obs, lookahead_tar_obs, lookahead_tar_mask):
            if self.disable_lookahead_mask_in_obs:
                input_tensor = lookahead_tar_obs
            else:
                input_tensor = torch.cat([lookahead_tar_obs, lookahead_tar_mask], dim=-1)

            lookahead_embedding = self.lookahead_tar_enc(input_tensor)
            obs = torch.cat((lookahead_embedding, obs[..., self.total_embedding_dim:]), dim=-1)

            return obs

        def eval_actor(self, obs):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out)
                     
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return
        
        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value
        
        def eval_disc(self, amp_obs, amp_obs_mask, return_input_tensor = False):
            ##################################################################
            amp_obs = amp_obs if self.disable_multi_mask_amp else self.apply_amp_obs_mask(amp_obs, amp_obs_mask)
            amp_input = torch.cat([amp_obs, amp_obs_mask], dim=-1)
            ##################################################################

            disc_mlp_out = self._disc_mlp(amp_input)
            disc_logits = self._disc_logits(disc_mlp_out)
            if return_input_tensor:
                return disc_logits, amp_input
            else:
                return disc_logits

        def apply_amp_obs_mask(self, amp_obs, amp_obs_mask):
            amp_obs = amp_obs.detach()
            assert amp_obs_mask.dtype == torch.bool

            if len(amp_obs.shape) == 2:
                assert len(amp_obs.shape) == 2
                bs = amp_obs.size(0)
                amp_obs.view(bs, self.lookahead_timesteps, -1)[amp_obs_mask.unsqueeze(-2).repeat(1,self.lookahead_timesteps,1)] = 0
                # ToRemove Assert Check
                assert torch.sum(amp_obs.view(bs, self.lookahead_timesteps, -1)[amp_obs_mask.unsqueeze(-2).repeat(1,self.lookahead_timesteps,1)]) == 0
            elif len(amp_obs.shape) == 3:
                assert len(amp_obs.shape) == 3
                bs = amp_obs.shape[:2]
                amp_obs.view(*bs, self.lookahead_timesteps, -1)[amp_obs_mask.unsqueeze(-2).repeat(1,1,self.lookahead_timesteps,1)] = 0
                assert torch.sum(amp_obs.view(*bs, self.lookahead_timesteps, -1)[amp_obs_mask.unsqueeze(-2).repeat(1,1,self.lookahead_timesteps,1)]) == 0
            else:
                assert False, "amp_obs shape is not supported"
            
            amp_obs.requires_grad = True
            return amp_obs
        
        def _build_disc(self, input_shape):
            self._disc_mlp = torch.nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_sequential_mlp_sn(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            # self._disc_logits = torch.nn.Linear(mlp_out_size, 1)
            self._disc_logits = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(mlp_out_size, 1))

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, torch.nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -amp_network_builder.DISC_LOGIT_INIT_SCALE, amp_network_builder.DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias)
            return

        
        def _build_sequential_mlp_sn(self, input_size, units, activation, dense_func, norm_only_first_layer=False, norm_func_name = None):
            print('build mlp:', input_size)
            self.activations_factory.register_builder('silu', lambda  **kwargs : nn.SiLU(**kwargs))

            layers = []
            in_size = input_size
            for unit in units:
                layers.append(torch.nn.utils.parametrizations.spectral_norm(dense_func(in_size, unit)))
                # layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))
                in_size = unit

            return torch.nn.Sequential(*layers)
    
    def build(self, name, **kwargs):
        net = AMPBuilderV1.Network(self.params, **kwargs)
        return net
