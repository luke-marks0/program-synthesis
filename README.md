NOTE: A lot of code has been taken from [this repo](https://github.com/Felix-Petersen/difflogic), which is the official implementation of [Peterson et al.](https://arxiv.org/abs/2210.08277). This repo just strips back everything other than an extremely basic implementation of deep differentiable logic gate networks.

The most significant difference is that this repo supports only a program synthesis task in which the network is supposed to learn a Boolean vector function. The inputs and outputs are therefore binary vectors.

The repo structure is:
```
program_synthesis/
├── config/              # Configuration files
│   └── default.yaml     # Default configuration
├── core/                # Core model implementation
│   ├── __init__.py
│   ├── logic_layer.py   # LogicLayer implementation
│   └── model.py         # Complete network model
├── problems/            # Program synthesis tasks
│   ├── __init__.py
│   ├── base.py          # Base problem class
│   ├── problems.py      # Various program synthesis tasks
├── utils/               
│   ├── __init__.py
│   ├── config.py        # Configuration handling
│   └── training.py      # Training utilities
├── main.py
```
