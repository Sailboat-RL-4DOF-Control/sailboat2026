# 4-DOF Unmanned Sailboat RL Control

This repository provides a control workflow for a four-degree-of-freedom
sailboat, including SAC and SAC-LSTM training, wind and no-wind simulation environments,
automatic model evaluation, reward ablations, and a full 20-step-horizon MPC baseline.
The code keeps the direct-script workflow used during development: enter `code/`, run
the desired `.py` file, or open it in VS Code/PyCharm and click Run.

## Directory

```text
sailboat2026/
├─ README.md
├─ requirements.txt
└─ code/
│  ├─ train_sac_nowind.py       # 14-D feed-forward SAC
│  ├─ train_sac_wind.py         # 16-D feed-forward SAC
│  ├─ train_sac_lstm.py         # SAC-LSTM and reward ablations
│  ├─ batch_eval.py             # automatic model/environment matching
│  ├─ run_mpc.py                # full 20-step-horizon MPC
│  ├─ sailboat_s14_nowind.py
│  ├─ sailboat_s14_wind.py
│  ├─ noise_utils_nowind.py
│  ├─ noise_utils_wind.py
│  ├─ SAC_fc.py / SAC_lstm.py
│  ├─ utils_fc.py / utils_lstm.py
   ├─ fast_mpc.py
   └─ model/                    # main and reward-ablation seed-400 models
```

All paths are relative. The source files outside this folder were not changed.

## Installation

Python 3.11 is recommended:

| Library/runtime | Fixed version |
|---|---:|
| Python | 3.11 |
| PyTorch | 2.2.2 |
| Recommended CUDA build | CUDA 12.1 (`cu121`) |
| NumPy | 1.23.5 |
| SciPy | 1.11.4 |
| Gymnasium | 0.29.1 |
| Stable-Baselines3 | 2.2.1 |
| cloudpickle | 3.0.0 |
| pygame | 2.5.2 |
| Matplotlib | 3.8.4 |

```bash
python -m venv .venv

# Linux/macOS
.venv/bin/python -m pip install -r requirements.txt

# Windows
.venv\Scripts\python -m pip install -r requirements.txt
```

CUDA training is recommended. On a CUDA-capable Linux or Windows machine,
install the pinned CUDA 12.1 PyTorch wheel first, then install the remaining
requirements:

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU-only use:

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

No installation of the local code is required.

## Training

From `code/`:

```bash
python train_sac_nowind.py --seed 400
python train_sac_wind.py --seed 400
python train_sac_lstm.py --seed 400
```

The training scripts select `cuda` automatically when PyTorch reports that a
CUDA device is available. The recommended experiment command retains the
original 15 parallel environment workers:

```bash
python train_sac_nowind.py --seed 0 --device cuda --num-parallel-envs 15
python train_sac_wind.py --seed 0 --device cuda --num-parallel-envs 15
python train_sac_lstm.py --seed 0 --device cuda --num-parallel-envs 15
```

Set `--num-parallel-envs 1` for debugging or low-resource machines. Environment
simulation remains on CPU processes; policy and gradient computation use the
selected PyTorch device. Total training steps count transitions across all
parallel environments.

The same files can be opened and run directly in an IDE. Their default
hyperparameters are visible in `build_parser()` at the bottom of each file.
Command-line arguments override those defaults.

Training outputs are written under `code/trained_models/` and do not overwrite
the released models under `code/model/`.

### Current training defaults

The values below are the defaults currently fixed in the three training
scripts. Edit them in `build_parser()` or override them on the command line.

| Parameter | SAC-nowind / SAC-wind | SAC-LSTM |
|---|---:|---:|
| Discount factor | 0.98 | 0.98 |
| Actor learning rate | 3×10^-4 | 3×10^-4 |
| Critic learning rate | 3×10^-4 | 3×10^-4 |
| Temperature coefficient | 0.2 | 0.2 |
| Temperature learning rate | 3×10^-5 | 3×10^-5 |
| Replay buffer capacity | 1×10^6 | 1×10^6 |
| Batch size | 256 | 256 |
| Soft update coefficient | 0.002 | 0.002 |
| Maximum steps per episode | 1,000 | 1,000 |
| Parallel environments | 15 | 15 |
| Recurrent training sequence length (K) | — | 10 |
| Total training steps | 7.5×10^6 | 7.5×10^6 |
| Independent training seeds | 0, 100, 200, 300, 400 | 0, 100, 200, 300, 400 |
| Save interval | 500,000 | 500,000 |
| Warm-up steps | 20,000 | 20,000 |

Feed-forward SAC actors and critics use hidden sizes `(256, 128, 64)` for new
training. SAC-LSTM uses a 256-unit recurrent layer, a 256-unit output layer,
and sequence length 10. The released checkpoints retain their original network
dimensions; `batch_eval.py` reads those dimensions from each weight file.

Common examples of numerical adjustment:

```bash
python train_sac_nowind.py --seed 100 --actor-learning-rate 0.0001 --batch-size 512
python train_sac_wind.py --total-training-steps 3000000 --replay-buffer-capacity 500000
python train_sac_lstm.py --recurrent-training-sequence-length 20 --recurrent-hidden-size 512
```

### Reward ablation

Reward ablations use exactly the same SAC-LSTM network. Only the enabled reward
terms change:

```bash
python train_sac_lstm.py --reward-ablation no_progress
python train_sac_lstm.py --reward-ablation no_action
python train_sac_lstm.py --reward-ablation no_checkpoint
python train_sac_lstm.py --reward-ablation no_arrival
```

For direct IDE execution, change the default value of `--reward-ablation` near
the bottom of `train_sac_lstm.py`.

Representative seed-400 weights are supplied for the four archived ablations:
no progress reward, no action penalty, no checkpoint reward, and no arrival
reward.

The reward terms are:

- progress: `(2 × distance progress + x progress) + 0.5 × u/(1+|v|)`;
- action: sail/rudder adjustment penalty;
- checkpoints: `+3`, `+5`, and `+12` at 375, 250, and 125 m;
- arrival: `+40` within 10 m;
- time: `-0.2` per control step, always enabled in all considered variants;
- truncation: `-100`.

## Fixed-scale noise

`noise_utils_nowind.py` applies value-independent sensor noise rather than pure
proportional noise. `noise_utils_wind.py` extends the same model to the 16-D
observation: index 14 is wind speed and index 15 is wind direction. The original
noise files were not modified.

Fixed sensor standard deviations are 1 m for position, 1 degree for attitude
and heading, 0.1 m/s for velocity, 0.1 degree/s for angular velocity, and
0.5 m for distance-derived variables. Wind observation noise is 0.1 m/s and
1 degree. Actuator execution noise uses a 3% base ratio plus dead zones,
quantization, response variation, and bounded error.

## Simulation environment parameters

| Parameter | Value |
|---|---:|
| Control interval | 1.0 s |
| Physical integrator | adaptive Dormand--Prince RK45 |
| Initial physical integration step | 0.05 s |
| Integration tolerance | 1e-6 |
| Episode limit | 1,000 control steps |
| Target position | (500 m, 0 m) |
| Success radius | 10 m |
| Workspace truncation radius | 750 m from origin |
| Sail/rudder command increment bound | ±10 degrees per control step |
| Absolute sail-angle bound | ±90 degrees |
| Absolute rudder-angle bound | ±30 degrees |
| Mean wind speed | 8 m/s |
| Dryden longitudinal scale | 300 m |
| Dryden lateral scale | 150 m |
| Dryden longitudinal intensity | 1.2 m/s |
| Dryden lateral intensity | 1.0 m/s |

The 14-D observation contains position, attitude/heading, linear and angular
velocities, sail/rudder angles, two progress variables, and target position.
The 16-D wind observation appends wind speed and wind direction.

## Batch evaluation

```bash
python batch_eval.py --episodes 10 --wind-directions random
python batch_eval.py --episodes 10 --wind-directions 0,90,150,180
```

`batch_eval.py` reads the actor weights directly:

- FC or LSTM is determined from state-dict keys;
- 14-D models use the nowind environment;
- 16-D models use the wind environment.

The four reward-ablation folders are discovered and evaluated automatically
along with the three principal policies.

Results are saved to `code/test_results/`.

Adjust `--episodes`, `--wind-directions`, `--max-steps`, `--base-seed`,
`--device`, and `--noise/--no-noise` as needed.

## MPC

```bash
python run_mpc.py --wind-directions 0,90,150,180 --episodes 1
```

The defaults reproduce the full archived MPC configuration:

- prediction horizon: 20 control steps;
- SLSQP with 8 starts;
- RK4 prediction step: 0.1 s;
- maximum optimizer iterations: 100;
- maximum episode length: 1000 control steps.

This configuration is computationally expensive. `run_mpc.py` prints progress
every 25 control steps so a terminal or IDE run does not appear frozen.

The physical environment uses adaptive Dormand--Prince RK45 integration with
an initial step of 0.05 s and tolerance `1e-6`. This is separate from the fixed
0.1 s RK4 step inside the MPC prediction model.

MPC numerical values can be changed with `--prediction-horizon`, `--rk4-step`,
`--slsqp-starts`, `--max-iter`, `--max-steps`, and `--wind-directions`.
