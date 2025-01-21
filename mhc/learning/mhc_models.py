# Imports

# RL Games Imports
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


# --------------------------------------------
# -------------Networks Models----------------
# --------------------------------------------

class ModelAMPContinuousv1(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('amp', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelAMPContinuousv1.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)

            if (is_train):
                amp_obs = input_dict['amp_obs']
                amp_obs_mask = input_dict['amp_obs_mask']

                disc_agent_logit = self.a2c_network.eval_disc(amp_obs, amp_obs_mask)
                result["disc_agent_logit"] = disc_agent_logit

                amp_obs_replay = input_dict['amp_obs_replay']
                disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay, amp_obs_mask)
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                amp_demo_obs = input_dict['amp_obs_demo']
                disc_demo_logit, input_demo_obs_tensor = self.a2c_network.eval_disc(amp_demo_obs, amp_obs_mask, return_input_tensor = True)
                result["disc_demo_logit"] = disc_demo_logit
                result["amp_obs_demo_input_tensor"] = input_demo_obs_tensor

            return result


