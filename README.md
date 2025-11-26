# Behavior-Policy-Design-for-Policy-Evaluation

This repository contains the code used in experiments on behavior policy design for policy evaluation. The goal is to study how a carefully chosen behavior policy can reduce the variance of return estimates and improve the sample efficiency of policy evaluation in reinforcement learning.

The code supports:
- ground-truth policy evaluation through direct environment rollouts,
- off-policy evaluation using collected trajectories,
- behavior policy search (BPS) to find a good behavior policy,
- comparisons with simple baselines such as on-policy Monte Carlo or random behavior policies.

## Repository Structure

```
.
├── agents/            # Policy implementations (target policy, behavior policy, etc.)
├── environments/      # Environment wrappers (MuJoCo or custom toy envs)
├── utilities/         # Logging, buffers, evaluation functions
├── requirements/      # Python dependencies
├── 01_ground_truth.py # Computes ground-truth returns for the target policy
├── 02_train_OPE.py    # Off-policy evaluation experiments
├── 03_BPS.py          # Behavior policy search
├── 04_ROS.py          # Random off-policy sampling baseline
└── README.md
```

## Installation

Requires **Python 3.8+**.

Install dependencies:

```bash
pip install -r requirements/requirements.txt
```

If using MuJoCo environments, ensure MuJoCo is installed and configured.

## Usage

### 1. Ground-truth evaluation

```bash
python 01_ground_truth.py
```

### 2. Off-policy evaluation (OPE)

```bash
python 02_train_OPE.py
```

### 3. Behavior Policy Search (BPS)

```bash
python 03_BPS.py
```

### 4. Robust On-policy Sampling (ROS) Baseline

```bash
python 04_ROS.py
```

## Extending

You can:
- add new environments under `environments/`,
- modify or add behavior policies in `agents/`,
- add new OPE estimators,
- integrate logging or plotting tools.

## Citation

```
@InProceedings{efficient2024liu,
  title = {Efficient Policy Evaluation with Offline Data Informed Behavior Policy Design},
  author = {Liu, Shuze and Zhang, Shangtong},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024}
}
```

