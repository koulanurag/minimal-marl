import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append((np.ones(len(done)) - done).tolist())

        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(MuNet, self).__init__()
        self.num_agents = len(observation_space)
        self.action_space = action_space
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            num_action = action_space[agent_i].n
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, num_action)))

    def forward(self, obs):
        action_logits = [torch.empty(1, _.n) for _ in self.action_space]
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)
            action_logits[agent_i] = x

        return torch.cat(action_logits, dim=1)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        total_obs = sum([_.shape[0] for _ in observation_space])
        for agent_i in range(self.num_agents):
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(total_obs + self.num_agents, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, 1)))

    def forward(self, obs, action):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        x = torch.cat((obs.view(obs.shape[0], obs.shape[1] * obs.shape[2]), action), dim=1)
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(x)

        return torch.cat(q_values, dim=1)


def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, gamma, batch_size):
    state, action, reward, next_state, done_mask = memory.sample(batch_size)

    next_state_action_logits = mu_target(next_state)
    _, n_agents, action_size = next_state_action_logits.shape
    next_state_action_logits = next_state_action_logits.view(batch_size * n_agents, action_size)
    next_state_action = F.gumbel_softmax(logits=next_state_action_logits, tau=0.1, hard=False).max(dim=1)[0]
    next_state_action = next_state_action.view(batch_size, n_agents)

    target = reward + gamma * q_target(next_state, next_state_action) * done_mask
    q_loss = F.smooth_l1_loss(q(state, action), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    state_action_logits = mu(state)
    state_action_logits = state_action_logits.view(batch_size * n_agents, action_size)
    state_action = F.gumbel_softmax(logits=state_action_logits, tau=0.1, hard=False).max(dim=1)[0]
    state_action = state_action.view(batch_size, n_agents)

    mu_loss = -q(state, state_action).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def test(env, num_episodes, mu):
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]

        while not all(done):
            env.render()
            action_logits = mu(torch.Tensor(state).unsqueeze(0))
            action = action_logits.argmax(dim=2).squeeze(0).data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            score += np.array(reward)
            state = next_state

    return sum(score / num_episodes)


def main(env_name, lr_mu, lr_q, tau, gamma, batch_size):
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    memory = ReplayBuffer(buffer_limit=50000)

    q, q_target = QNet(env.observation_space, env.action_space), QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(env.observation_space, env.action_space), MuNet(env.observation_space, env.action_space)
    mu_target.load_state_dict(mu.state_dict())

    score = np.zeros(env.n_agents)
    log_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)

    for episode_i in range(10000):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]

        while not all(done):
            env.render()
            action_logits = mu(torch.Tensor(state).unsqueeze(0))
            action_probs = F.gumbel_softmax(logits=action_logits.squeeze(0), tau=1, hard=False)
            action = Categorical(probs=action_probs).sample().data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            memory.put((state, action, (np.array(reward)).tolist(), next_state,
                        np.array(done, dtype=int).tolist()))
            score += np.array(reward)
            state = next_state

        if memory.size() > 2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, gamma, batch_size)
                soft_update(mu, mu_target, tau)
                soft_update(q, q_target, tau)

        if episode_i % log_interval == 0 and episode_i != 0:
            test_score = test(test_env, 5, mu)
            print("# of episode :{}, avg train score : {:.1f}, "
                  "test score: {:.1f} ".format(episode_i, sum(score / log_interval), test_score))
            score = np.zeros(env.n_agents)

    env.close()
    test_env.close()


if __name__ == '__main__':
    main(env_name='ma_gym:Checkers-v0',
         lr_mu=0.0005,
         lr_q=0.001,
         tau=0.005,
         batch_size=32,
         gamma=0.99)
