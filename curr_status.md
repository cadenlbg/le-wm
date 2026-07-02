# LeWM 远程环境当前状态

本文档记录当前可用状态、已经完成的修复/实验，以及下一步计划。不记录账号密码。

## 目录

- [第一部分 当前状态](#part-1)
  - [远程机器](#remote-machine)
  - [当前存储位置](#storage)
  - [Python 与 Conda 环境](#python-conda)
  - [PyTorch 与 CUDA](#torch-cuda)
  - [环境变量](#env-vars)
  - [当前数据状态](#data-state)
  - [Checkpoint 状态](#checkpoint-state)
  - [实验记录与归档](#archive)
  - [已经完成的事项](#done-items)
  - [关键修复](#fixes)
- [第二部分 未来计划](#part-2)
  - [PushT 参数对比实验](#pusht-sweep)
  - [TwoRoom 复现实验](#tworoom)
  - [训练冒烟测试](#train-smoke)

<a id="part-1"></a>
## 第一部分 当前状态

<a id="remote-machine"></a>
### 1. 远程机器

当前远程主机：

```text
ubuntu-zlin-cc18
```

当前用户：

```text
zflin
```

GPU 状态：

```text
NVIDIA GeForce RTX 3090 x 8
Each GPU memory: 24576 MiB
Driver Version: 570.169
CUDA Version reported by nvidia-smi: 12.8
```

<a id="storage"></a>
### 2. 当前存储位置

LeWM 代码仓库目录：

```text
/data/zflin/lewm_re/le-wm
```

LeWM 数据、checkpoint 和缓存目录：

```text
/data/zflin/lewm_re/stablewm_data
```

实验归档目录已经移到仓库外：

```text
/data/zflin/lewm_re/experiments
```

当前推荐工作目录：

```bash
cd /data/zflin/lewm_re/le-wm
```

说明：

- `/home` 不再用于存放 LeWM 代码、数据或 conda 环境。
- 当前所有 LeWM 相关内容统一放在 `/data/zflin` 下。
- PushT 和 TwoRoom 已下载并解压完成。
- Cube/Reacher 暂不下载。
- `experiments/` 不再放在 `le-wm` 仓库内部，避免同步 GitHub 时混入实验产物。

<a id="python-conda"></a>
### 3. Python 与 Conda 环境

当前使用 `/data` 下的新 conda：

```text
/data/zflin/software/miniconda3
```

当前 LeWM 环境：

```bash
conda activate lewm
```

期望检查结果：

```bash
which conda
which python
echo $CONDA_PREFIX
conda info --base
```

应指向：

```text
/data/zflin/software/miniconda3/bin/conda
/data/zflin/software/miniconda3/envs/lewm/bin/python
/data/zflin/software/miniconda3/envs/lewm
/data/zflin/software/miniconda3
```

核心依赖已可导入：

```python
import torch
import stable_worldmodel as swm
import stable_pretraining as spt
```

<a id="torch-cuda"></a>
### 4. PyTorch 与 CUDA

当前 PyTorch/CUDA 状态：

```text
torch: 2.11.0+cu128
cuda: True
cuda version: 12.8
device count: 8
```

CUDA 设备：

```text
0 NVIDIA GeForce RTX 3090
1 NVIDIA GeForce RTX 3090
2 NVIDIA GeForce RTX 3090
3 NVIDIA GeForce RTX 3090
4 NVIDIA GeForce RTX 3090
5 NVIDIA GeForce RTX 3090
6 NVIDIA GeForce RTX 3090
7 NVIDIA GeForce RTX 3090
```

单卡运行方式：

```bash
CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

<a id="env-vars"></a>
### 5. 环境变量

当前 LeWM 数据根目录：

```bash
export STABLEWM_HOME=/data/zflin/lewm_re/stablewm_data
```

当前 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

新开终端推荐执行：

```bash
conda activate lewm
export STABLEWM_HOME=/data/zflin/lewm_re/stablewm_data
export HF_ENDPOINT=https://hf-mirror.com
cd /data/zflin/lewm_re/le-wm
```

如果需要强制离线检查，可临时设置：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

<a id="data-state"></a>
### 6. 当前数据状态

已解压完成的数据：

```text
$STABLEWM_HOME/datasets/pusht_expert_train.h5
$STABLEWM_HOME/datasets/tworoom.h5
```

保留的下载压缩包：

```text
$STABLEWM_HOME/downloads/lewm-pusht/pusht_expert_train.h5.zst
$STABLEWM_HOME/downloads/lewm-tworooms/tworoom.tar.zst
```

暂不下载：

```text
lewm-cube
lewm-reacher
```

查看当前数据：

```bash
ls -lh "$STABLEWM_HOME/datasets/pusht_expert_train.h5" "$STABLEWM_HOME/datasets/tworoom.h5"
du -sh "$STABLEWM_HOME"/downloads/* 2>/dev/null
df -h /data
```

<a id="checkpoint-state"></a>
### 7. Checkpoint 状态

PushT 作者 checkpoint 已下载并转换成功。当前可被 `policy=pusht/lewm` 加载的本地缓存为：

```text
$STABLEWM_HOME/checkpoints/models--pusht--lewm/weights.pt
$STABLEWM_HOME/checkpoints/models--pusht--lewm/config.json
```

原始 HF 下载目录：

```text
$STABLEWM_HOME/hf_pusht
```

转换命令：

```bash
python scripts/convert_hf_checkpoint.py --repo hf_pusht --run-name pusht/lewm
```

说明：

- 当前 `load_pretrained` / `AutoCostModel('pusht/lewm')` 走的是 `$STABLEWM_HOME/checkpoints/models--pusht--lewm/*.pt` 这类缓存布局。
- 之前的 `$STABLEWM_HOME/pusht/lewm_object.ckpt` 格式不再作为主要加载目标。
- 转换脚本现在输出纯 `state_dict`，避免 PyTorch `weights_only` 反序列化报错。

<a id="archive"></a>
### 8. 实验记录与归档

实验统一放在仓库外：

```text
/data/zflin/lewm_re/experiments/<experiment_name>/
```

推荐目录内容：

```text
README.md
command.txt
config.yaml
results.txt
notes.md
artifacts/
hydra_outputs/
```

自动归档通式：

```bash
bash scripts/archive_experiment.sh <experiment_name> "<command>"
```

当前归档脚本会自动处理：

- 创建 `/data/zflin/lewm_re/experiments/<experiment_name>/`
- 写入 `README.md` 和 `command.txt`
- 移动 `pusht_results.txt` 或 `tworoom_results.txt`
- 移动仓库根目录下的 `*.mp4`
- 移动 `$STABLEWM_HOME/pusht/` 或 `$STABLEWM_HOME/tworoom/` 下的 `*.mp4`
- 复制对应 `config/eval/*.yaml` 到 `config.yaml`
- 移动 Hydra 的 `outputs/` 到 `hydra_outputs/`

Hydra 输出目录说明：

```text
outputs/YYYY-MM-DD/HH-MM-SS/
```

这些目录一般记录 `.hydra/config.yaml`、`.hydra/hydra.yaml`、`.hydra/overrides.yaml`，对复现实验有用，归档时应保留。

<a id="done-items"></a>
### 9. 已经完成的事项

- LeWM GitHub 仓库已 clone 到远程服务器。
- 代码、数据目录和 conda 环境已迁移到 `/data/zflin`。
- `lewm` conda 环境已可用。
- PyTorch 已匹配服务器 CUDA 12.8，CUDA 可用。
- `stable_worldmodel` 和 `stable_pretraining` 已可导入。
- PushT 数据已放入 `$STABLEWM_HOME/datasets/pusht_expert_train.h5`。
- TwoRoom 数据已放入 `$STABLEWM_HOME/datasets/tworoom.h5`。
- `hdf5plugin` 已安装，用于读取压缩 HDF5 数据。
- `eval.py` 已改为使用新版 `stable_worldmodel.data.load_dataset()`。
- PushT HF checkpoint 已转换为 loader 期望的缓存格式。
- PushT checkpoint 小规模评估已跑通。
- PushT 默认正式评估已跑完，并准备归档。
- `experiments/` 已迁移到 `le-wm` 仓库外层。
- `archive_experiment.sh` 已更新，会归档结果、视频和 Hydra outputs。

已确认 PushT 小规模评估结果：

```text
command: CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=2 solver.num_samples=50 solver.n_steps=5
success_rate: 50.0
episode_successes: [False, True]
```

<a id="fixes"></a>
### 10. 关键修复

#### 10.1 HDF5 数据读取

当前 `stable_worldmodel` 版本不再暴露旧的 `HDF5Dataset` 类，评估脚本需要使用新版：

```python
swm.data.load_dataset(...)
```

同时读取压缩 HDF5 时需要：

```bash
pip install hdf5plugin
```

#### 10.2 HF checkpoint 转换

`config.json` 是 Hydra 风格配置，包含 `_target_`，不能直接写：

```python
ARPredictor(**cfg["predictor"])
```

现在用：

```python
from hydra.utils import instantiate
model = instantiate(cfg)
```

#### 10.3 ViT 权重命名重写

转换脚本会处理 HF 权重和当前 LeWM 模型之间的命名差异，例如：

```text
encoder.encoder.layer. -> encoder.layers.
attention.attention.query -> attention.q_proj
attention.attention.key -> attention.k_proj
attention.attention.value -> attention.v_proj
attention.output.dense -> attention.o_proj
intermediate.dense -> mlp.fc1
output.dense -> mlp.fc2
```

#### 10.4 PyTorch weights_only 报错

之前保存整个模型对象会触发 `pickle.UnpicklingError` / `weights_only` 问题。当前脚本保存纯 `state_dict`：

```python
torch.save(sd, out)
```

<a id="part-2"></a>
## 第二部分 未来计划

<a id="pusht-sweep"></a>
### 11.1 PushT 参数对比实验

以默认评估作为 baseline：

```bash
CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

默认参数等价于：

```text
eval.num_eval=50
solver.num_samples=300
solver.n_steps=30
solver.topk=30
```

接下来主要固定其他参数，只改变 `solver.n_steps` 做对比：

```bash
CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=pusht/lewm solver.n_steps=5
CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=pusht/lewm solver.n_steps=10
CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=pusht/lewm solver.n_steps=20
CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=pusht/lewm solver.n_steps=40
```

每次跑完立刻归档，例如：

```bash
bash scripts/archive_experiment.sh 2026-07-02_pusht_lewm_nsteps10 "CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=pusht/lewm solver.n_steps=10"
```

<a id="tworoom"></a>
### 11.2 TwoRoom 复现实验

TwoRoom 数据已经准备好。下一步需要下载并转换 TwoRoom 作者 checkpoint，然后跑：

```bash
CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=tworoom.yaml policy=tworoom/lewm eval.num_eval=2 solver.num_samples=50 solver.n_steps=5
```

小规模通过后再跑默认评估。

<a id="train-smoke"></a>
### 11.3 训练冒烟测试

评估链路稳定后，再测训练链路：

```bash
CUDA_VISIBLE_DEVICES=3 python train.py data=pusht trainer.devices=1 trainer.max_epochs=1 loader.batch_size=16
```

如果成功，再考虑默认训练或 frozen backbone 下游实验。
