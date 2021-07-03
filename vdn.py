import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


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


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            n_action = action_space[agent_i].n
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, n_action)))

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)

        return torch.cat(q_values, dim=1)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action


def train(q, q_target, memory, optimizer, gamma, batch_size):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(2, a.unsqueeze(-1).long()).squeeze(-1)
        sum_q = q_a.sum(dim=1, keepdims=True)
        max_q_prime = q_target(s_prime).max(dim=2)[0].squeeze(-1)
        target = r.sum(dim=1, keepdims=True) + gamma * (max_q_prime * done_mask).sum(dim=1, keepdims=True)
        loss = F.smooth_l1_loss(sum_q, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(env, num_episodes, q):
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]

        while not all(done):
            env.render()
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon=0)
            action = action[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            score += np.array(reward)
            state = next_state

    return sum(score / num_episodes)


def main(env_name, lr, gamma, batch_size):
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    memory = ReplayBuffer(buffer_limit=50000)

    q = QNet(env.observation_space, env.action_space)
    q_target = QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())

    score = np.zeros(env.n_agents)
    log_interval = 20

    optimizer = optim.Adam(q.parameters(), lr=lr)

    for episode_i in range(1000):
        epsilon = max(0.1, 0.9 - 0.1 * (episode_i / 600))  # Linear annealing
        state = env.reset()
        done = [False for _ in range(env.n_agents)]

        env.render()
        while not all(done):
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)
            action = action[0].data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            memory.put((state, action, (np.array(reward)).tolist(), next_state,
                        np.array(done, dtype=int).tolist()))
            score += np.array(reward)
            state = next_state
            env.render()

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer, gamma, batch_size)

        if episode_i % log_interval == 0 and episode_i != 0:
            q_target.load_state_dict(q.state_dict())
            test_score = test(test_env, 5, q)
            print("# of episode :{}, avg train score : {:.1f}, "
                  "test score: {:.1f} n_buffer : {}, eps : {:.1f}%".format(episode_i,
                                                                           sum(score / log_interval), test_score,
                                                                           memory.size(), epsilon * 100))
            score = np.zeros(env.n_agents)

    env.close()
    test_env.close()


if __name__ == '__main__':
    main(env_name='ma_gym:Switch2-v1',
         lr=0.0005,
         batch_size=32,
         gamma=0.99)
