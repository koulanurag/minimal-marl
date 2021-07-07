# minimal-marl

Minimal implementation of multi-agent reinforcement learning algorithms(marl). This repo
complements [`ma-gym`](https://github.com/koulanurag/ma-gym) and is is inspired
by [`minimalRl`](https://github.com/seungeunrho/minimalRL) which provides minimal implementation for RL algorithms for
the sake of understanding. I couldn't find the same for  `multi-agent` environments and thereby created it.

[Report](https://wandb.ai/koulanurag/minimal-marl/reports/Minimal-Marl--Vmlldzo4MzM2MDc?accessToken=vy6dydemfdvekct02pevp3girjvb0tnt1ou2acb2h0fl478hdjqqu8ydbco6uz38)

## Installation

```bash 
  pip install ma-gym>=0.0.6 torch>=1.8 wandb
```

## Usage

```bash
python <algo_name>.py # such as `vdn.py`
```

## Algorithms

- [x] VDN (Value Decomposition Network)
- [x] MADDPG (Multi Agent Deep Deterministic Policy Gradient)

## Contributing

Contributions are always welcome!

Feel free to send a `pull-request` if you would like to add a new algorithm or further optimize an existing algorithm.
Also, If adding a new algorithm, please start by raising an `issue` and
use [`ma-gym`](https://github.com/koulanurag/ma-gym) or
[`PettingZoo`](https://github.com/PettingZoo-Team/PettingZoo) for testing.


