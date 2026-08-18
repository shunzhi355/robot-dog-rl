# StackForce SimReady Isaac Gym 导出工程

这个工程由 StackForce SimReady 导出，可直接用于 NVIDIA Isaac Gym / legged_gym 训练。

导出的 task id 是：

```text
stackforce_train081801
```

### 直接可复制的训练命令

```bash
conda activate <你自己的IsaacGym环境名称>
cd <exported_project>
python -m pip install -e source/stackforce_simready_train081801_gym
python scripts/list_envs.py
python scripts/train.py --task stackforce_train081801 --headless --num_envs 64 --max_iterations 100
```

如果你想看窗口，把训练命令里的 `--headless` 去掉即可。

### 推荐 Isaac Gym / legged_gym 环境

本导出工程推荐使用下面这套已验证配置：

```text
Python 3.8
Isaac Gym Preview 4 / 1.0rc4
Torch 2.4.1+cu121
Torchvision 0.19.1+cu121
NumPy 1.24.1
LeggedGym-Ex 0.3.0
```

Isaac Gym 需要你先从 NVIDIA 下载 Preview 4 并解压，因为 NVIDIA 需要登录和许可确认。导出包内的一键脚本会自动安装其它依赖：

```bash
chmod +x scripts/setup_stackforce_isaac_gym_env.sh
ISAAC_GYM_ROOT=/path/to/isaacgym ./scripts/setup_stackforce_isaac_gym_env.sh
```

脚本默认创建 `env_isaacgym`。如果你想改环境名：

```bash
ENV_NAME=my_isaacgym ISAAC_GYM_ROOT=/path/to/isaacgym ./scripts/setup_stackforce_isaac_gym_env.sh
```


工程可以先放到任意目录再安装；如果安装后又移动了目录，请在新目录里重新执行一次：

```bash
python -m pip install -e source/stackforce_simready_train081801_gym
```

### 训练输出和 checkpoint

训练输出在：

```text
logs/<experiment_name>/<timestamp>/
```

可以直接用：

```bash
find logs -name "*.pt"
find logs -name "*.pt" | sort | tail -n 1
find logs -name "policy.onnx"
```

Training saves `model_final.pt` and also attempts to export `exported/policies/policy.onnx`.

### 继续训练

```bash
python scripts/train.py --task stackforce_train081801 --resume --load_run <run_dir_name> --checkpoint <checkpoint_number>
```

### 播放训练后的策略

```bash
python scripts/play.py --task stackforce_train081801
```

如需加载指定 checkpoint：

```bash
python scripts/play.py --task stackforce_train081801 --checkpoint_path logs/train081801/<timestamp>/model_final.pt
```

Regenerate ONNX from a checkpoint:

```bash
python scripts/play.py --task stackforce_train081801 --checkpoint_path logs/train081801/<timestamp>/model_final.pt --export_onnx --num_steps 1
```

### 增加自定义 Reward

编辑：

```text
source/stackforce_simready_train081801_gym/stackforce_simready_train081801_gym/tasks/train081801/custom_rewards.py
```

在 `compute_custom_reward(env)` 中实现自己的 reward。
然后再编辑：

```text
source/stackforce_simready_train081801_gym/stackforce_simready_train081801_gym/tasks/train081801/train081801_config.py
```

把：

```python
custom_reward = 0.0
```

改成非零，比如：

```python
custom_reward = 1.0
```
