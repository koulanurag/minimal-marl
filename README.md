# minimal-marl

Minimal implementation of multi-agent reinforcement learning algorithms(marl). This repo
complements [`ma-gym`](https://github.com/koulanurag/ma-gym) and is is inspired
by [`minimalRl`](https://github.com/seungeunrho/minimalRL) which provides minimal implementation for RL algorithms for
the ease of understanding.

[![](https://img.shields.io/badge/-Training%20Results-informational?style=for-the-badge)](https://wandb.ai/koulanurag/minimal-marl/reports/Minimal-Marl--Vmlldzo4MzM2MDc?accessToken=vy6dydemfdvekct02pevp3girjvb0tnt1ou2acb2h0fl478hdjqqu8ydbco6uz38)
[![](https://img.shields.io/badge/-Work%20in%20Progress-orange?style=for-the-badge)]()

## Installation

```bash 
  pip install ma-gym>=0.0.7 torch>=1.8 wandb
```

## Usage

```bash
python <algo_name>.py # such as `vdn.py`
```

## Algorithms

- [ ] IDQN ( Independent Deep-Q
  Network) [DQN version of [IQL](https://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf)]
- [ ] [VDN](https://arxiv.org/abs/1706.05296) (Value Decomposition Network)
- [ ] [QMIX](https://arxiv.org/pdf/1803.11485.pdf)
- [ ] [MADDPG](https://arxiv.org/abs/1706.02275) (Multi Agent Deep Deterministic Policy Gradient)
    - `Status: Not converging at the moment`

## Contributing

Contributions are always welcome!

Feel free to send a `pull-request` if you would like to add a new algorithm or further optimize an existing algorithm.
Also, If adding a new algorithm, please start by raising an `issue`.

