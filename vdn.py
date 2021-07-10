import collections
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor

USE_WANDB = False  # if enabled, logs data on wandb server


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

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_mask_lst.append((np.ones(len(done)) - done).tolist())

        n_agents, obs_size, action_size = 2, 3, 5
        return torch.tensor(s_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(a_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, action_size), \
               torch.tensor(r_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(s_prime_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(done_mask_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                            nn.ReLU(),
                                                                            nn.Linear(128, 64),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(64, 64))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(64, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)

        return torch.cat(q_values, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        out = self.forward(obs, hidden)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, 64))


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10):
    for _ in range(update_iter):
        s, a, r, s_prime, done_mask = memory.sample_chunk(batch_size, chunk_size)

        q_out = q(s)
        q_a = q_out.gather(2, a.unsqueeze(-1).long()).squeeze(-1)
        sum_q = q_a.sum(dim=1, keepdims=True)
        max_q_prime = q_target(s_prime).max(dim=2)[0].squeeze(-1)
        target = r.sum(dim=1, keepdims=True) + gamma * (max_q_prime * done_mask).sum(dim=1, keepdims=True)
        loss = F.smooth_l1_loss(sum_q, target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(env, num_episodes, q):
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        with torch.no_grad():
            hidden = q.init_hidden()
            while not all(done):
                action = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)[0]
                next_state, reward, done, info = env.step(action.data.cpu().numpy().tolist())
                score += np.array(reward)
                state = next_state

    return sum(score / num_episodes)


def main(env_name, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, update_iter, recurrent, monitor):
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    if monitor:
        test_env = Monitor(test_env, directory='recordings/vdn/{}'.format(env_name),
                           video_callable=lambda episode_id: episode_id % 50 == 0)
    memory = ReplayBuffer(buffer_limit)

    q = QNet(env.observation_space, env.action_space, recurrent)
    q_target = QNet(env.observation_space, env.action_space, recurrent)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    score = np.zeros(env.n_agents)
    for episode_i in range(max_episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.4 * max_episodes)))
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        step_i = 0
        hidden = q.init_hidden()
        while not all(done):
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon)[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            step_i += 1
            if step_i >= env._max_steps or (step_i < env._max_steps and not all(done)):
                _done = [False for _ in done]
            else:
                _done = done
            memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(_done, dtype=int).tolist()))
            score += np.array(reward)

            state = next_state

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)

        if episode_i % log_interval == 0 and episode_i != 0:
            q_target.load_state_dict(q.state_dict())
            test_score = test(test_env, test_episodes, q)
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, sum(score / log_interval), test_score, memory.size(), epsilon))
            if USE_WANDB:
                wandb.log({'episode': episode_i, 'test-score': test_score,
                           'buffer-size': memory.size(), 'epsilon': epsilon, 'train-score': sum(score / log_interval)})
            score = np.zeros(env.n_agents)

    env.close()
    test_env.close()


if __name__ == '__main__':
    kwargs = {'env_name': 'ma_gym:Switch2-v2',
              'lr': 0.0005,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'log_interval': 20,
              'max_episodes': 30000,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'recurrent': True,
              'monitor': False}
    if USE_WANDB:
        import wandb

        wandb.init(project='minimal-marl', config={'algo': 'vdn', **kwargs}, monitor_gym=False)

    main(**kwargs)
