# minimal-marl

Minimal implementation of multi-agent reinforcement learning algorithms(marl). This repo
complements [`ma-gym`](https://github.com/koulanurag/ma-gym) and is is inspired
by [`minimalRl`](https://github.com/seungeunrho/minimalRL) which provides minimal implementation for RL algorithms for
the ease of understanding.

[![](https://img.shields.io/badge/-Training%20Results-informational?style=for-the-badge)](https://wandb.ai/koulanurag/minimal-marl/reports/Minimal-Marl--Vmlldzo4MzM2MDc?accessToken=vy6dydemfdvekct02pevp3girjvb0tnt1ou2acb2h0fl478hdjqqu8ydbco6uz38)

## Installation

```bash 
  pip install ma-gym>=0.0.6 torch>=1.8 wandb
```

## Usage

```bash
python <algo_name>.py # such as `vdn.py`
```

## Algorithms

- [ ] IQL ( Independent Q Learning) 
  - Decentralized training and Decentralized execution
  - It does not address the non-stationarity introduced due to the changing policies of the learning agents, and thus, unlike Q-learning, has no convergence guarantees even in the limit of infinite exploration.
- [x] VDN (Value Decomposition Network) _[centralized training and centralized execution]_
- [ ] MADDPG (Multi Agent Deep Deterministic Policy Gradient) _(centralized training and decentralized execution)_
- [ ] QMIX
  - Lies between the extremes of IQL and centralised Q-learning, but can represent a much richer class of action-value functions

## Contributing

Contributions are always welcome!

Feel free to send a `pull-request` if you would like to add a new algorithm or further optimize an existing algorithm.
Also, If adding a new algorithm, please start by raising an `issue`.

