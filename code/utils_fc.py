import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import globals


# 复用Z-score标准化类
class ZScoreNormalizer:
    def __init__(self, state_dim, device):
        self.device = device
        self.count = 0
        self.mean = torch.zeros(state_dim, device=device)
        self.var = torch.ones(state_dim, device=device)
        self.epsilon = 1e-6

    def update(self, data):
        if len(data.shape) == 3:
            data = data.view(-1, data.shape[-1])
        elif len(data.shape) == 2:
            pass
        else:
            data = data.unsqueeze(0)

        batch_count = data.shape[0]
        batch_mean = torch.mean(data, dim=0)
        batch_var = torch.var(data, dim=0, unbiased=False)

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            total_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / total_count
            self.var = (self.count * self.var + batch_count * batch_var +
                        delta ** 2 * self.count * batch_count / total_count) / total_count

        self.count += batch_count

    def normalize(self, data):
        return (data - self.mean) / (torch.sqrt(self.var) + self.epsilon)

    def denormalize(self, data):
        return data * (torch.sqrt(self.var) + self.epsilon) + self.mean

    def state_dict(self):
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean'].to(self.device)
        self.var = state_dict['var'].to(self.device)
        self.count = state_dict['count']


def build_net(layer_shape, hidden_activation, output_activation, dropout_prob=0.1):
    '''构建包含Dropout正则化的网络'''
    layers = []
    for j in range(len(layer_shape) - 1):
        layers.append(nn.Linear(layer_shape[j], layer_shape[j + 1]))
        if j < len(layer_shape) - 2:
            layers.append(hidden_activation())
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """纯全连接层Actor网络，仅使用当前状态进行决策。"""

    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU, dropout_prob=0.05):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 全连接网络：state -> hidden layers -> mu / log_std
        layers = [state_dim] + list(hid_shape)
        self.fc = build_net(layers, hidden_activation, output_activation, dropout_prob)

        self.mu_layer = nn.Linear(hid_shape[-1], action_dim)
        self.log_std_layer = nn.Linear(hid_shape[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -10

    def forward(self, current_state=None, deterministic=False, with_logprob=False):
        '''
        纯全连接前向传播，仅使用current_state。
        '''
        x = self.fc(current_state)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        a = torch.tanh(u)
        if with_logprob:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (
                2 * (np.log(2) - u - F.softplus(-2 * u))
            ).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a


class Double_Q_Critic(nn.Module):
    """纯全连接层Double Q Critic网络，使用当前状态和动作进行Q值评估。"""

    def __init__(self, state_dim, action_dim, hid_shape_q1, hid_shape_q2, dropout_prob=0.05):
        super(Double_Q_Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q1 网络: (state + action) -> hidden layers -> Q value
        layers_q1 = [state_dim + action_dim] + list(hid_shape_q1)
        self.fc_q1 = build_net(layers_q1, nn.ReLU, nn.ReLU, dropout_prob)
        self.Q_1 = nn.Linear(hid_shape_q1[-1], 1)

        # Q2 网络: (state + action) -> hidden layers -> Q value
        layers_q2 = [state_dim + action_dim] + list(hid_shape_q2)
        self.fc_q2 = build_net(layers_q2, nn.ReLU, nn.ReLU, dropout_prob)
        self.Q_2 = nn.Linear(hid_shape_q2[-1], 1)

    def forward(self, current_state=None, current_action=None):
        '''
        纯全连接前向传播，仅使用current_state和current_action。
        '''
        sa = torch.cat([current_state, current_action], dim=1)

        q1 = self.Q_1(self.fc_q1(sa))
        q2 = self.Q_2(self.fc_q2(sa))

        return q1, q2


def Reward_adapter(r, EnvIdex):
    if EnvIdex == 0:
        r = (r + 8) / 8
    elif EnvIdex == 1:
        if r <= -100: r = -10
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100: r = -1
    return r


def Action_adapter(a, max_action):
    return a * max_action


def Action_adapter_reverse(act, max_action):
    return act / max_action


def evaluate_policy(env, agent, max_action, turns=3, noise_manager=None, eval_with_noise=False):
    total_scores = 0

    for j in range(turns):
        s_raw, info = env.reset()

        if eval_with_noise and noise_manager is not None:
            s = noise_manager.add_observation_noise(s_raw)
        else:
            s = s_raw.copy()

        done = False
        episode_score = 0
        while not done:
            a = agent.select_action(s, deterministic=True)

            if eval_with_noise and noise_manager is not None:
                a_noisy = noise_manager.add_action_noise(a, max_action)
            else:
                a_noisy = a

            act = Action_adapter(a_noisy, max_action)

            s_next_raw, r, dw, tr, info = env.step(act)
            done = (dw or tr)

            if eval_with_noise and noise_manager is not None:
                s_next = noise_manager.add_observation_noise(s_next_raw)
            else:
                s_next = s_next_raw.copy()

            episode_score += r
            s = s_next

        total_scores += episode_score

    return int(total_scores / turns)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
