# StackForce SimReady Isaac Gym Export

This project was generated from StackForce SimReady and is ready to train in NVIDIA Isaac Gym / legged_gym.

The exported task id is:

```text
stackforce_train081801
```

### Copy-Paste Training Commands

```bash
conda activate <your_isaac_gym_env_name>
cd <exported_project>
python -m pip install -e source/stackforce_simready_train081801_gym
python scripts/list_envs.py
python scripts/train.py --task stackforce_train081801 --headless --num_envs 64 --max_iterations 100
```

If you want the viewer window to open, remove `--headless`.

### Recommended Isaac Gym / legged_gym Environment

This export is recommended with the validated stack below:

```text
Python 3.8
Isaac Gym Preview 4 / 1.0rc4
Torch 2.4.1+cu121
Torchvision 0.19.1+cu121
NumPy 1.24.1
LeggedGym-Ex 0.3.0
```

Download and extract Isaac Gym Preview 4 from NVIDIA first, because NVIDIA gates it behind its login and license flow. The bundled one-click script installs the remaining dependencies:

```bash
chmod +x scripts/setup_stackforce_isaac_gym_env.sh
ISAAC_GYM_ROOT=/path/to/isaacgym ./scripts/setup_stackforce_isaac_gym_env.sh
```

The default environment name is `env_isaacgym`. Override it with:

```bash
ENV_NAME=my_isaacgym ISAAC_GYM_ROOT=/path/to/isaacgym ./scripts/setup_stackforce_isaac_gym_env.sh
```


You can keep the project in any folder before installation. If you move it after `pip install -e`, rerun:

```bash
python -m pip install -e source/stackforce_simready_train081801_gym
```

### Training Outputs And Checkpoints

Training outputs are written under:

```text
logs/<experiment_name>/<timestamp>/
```

Useful commands:

```bash
find logs -name "*.pt"
find logs -name "*.pt" | sort | tail -n 1
find logs -name "policy.onnx"
```

Training saves `model_final.pt` and also attempts to export `exported/policies/policy.onnx`.

### Resume Training

```bash
python scripts/train.py --task stackforce_train081801 --resume --load_run <run_dir_name> --checkpoint <checkpoint_number>
```

### Play A Trained Policy

```bash
python scripts/play.py --task stackforce_train081801
```

To load a specific checkpoint:

```bash
python scripts/play.py --task stackforce_train081801 --checkpoint_path logs/train081801/<timestamp>/model_final.pt
```

Regenerate ONNX from a checkpoint:

```bash
python scripts/play.py --task stackforce_train081801 --checkpoint_path logs/train081801/<timestamp>/model_final.pt --export_onnx --num_steps 1
```

### Add A Custom Reward

Edit:

```text
source/stackforce_simready_train081801_gym/stackforce_simready_train081801_gym/tasks/train081801/custom_rewards.py
```

Implement your reward in `compute_custom_reward(env)`.
Then edit:

```text
source/stackforce_simready_train081801_gym/stackforce_simready_train081801_gym/tasks/train081801/train081801_config.py
```

Change:

```python
custom_reward = 0.0
```

to a non-zero scale such as:

```python
custom_reward = 1.0
```
