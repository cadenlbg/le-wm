# LeWM 远程环境当前状态

本文档只记录当前可用状态和下一步实验入口，不记录账号密码。

## 目录

- [第一部分 当前状态](#part-1)
  - [远程机器](#remote-machine)
  - [当前存储位置](#storage)
  - [Python 与 Conda 环境](#python-conda)
  - [PyTorch 与 CUDA](#torch-cuda)
  - [环境变量](#env-vars)
  - [当前数据状态](#data-state)
  - [当前已经完成的事项](#done-items)
  - [实验记录与归档](#archive)
  - [本次复现实验的关键改进](#improvements)
- [第二部分 未来计划](#part-2)
  - [先做数据与环境冒烟测试](#smoke-test)
  - [准备作者 checkpoint](#checkpoint)
  - [跑 checkpoint 小规模评估](#small-eval)
  - [跑 1 epoch 训练冒烟测试](#train-smoke)

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

当前推荐工作目录：

```bash
cd /data/zflin/lewm_re/le-wm
```

说明：

- `/home` 不再用于存放 LeWM 代码、数据或 conda 环境。
- 当前所有 LeWM 相关内容统一放在 `/data/zflin` 下。
- PushT 和 TwoRoom 已下载并解压完成。
- Cube/Reacher 暂不下载。

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

CUDA 检查命令：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
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

<a id="data-state"></a>
### 6. 当前数据状态

数据根目录：

```text
$STABLEWM_HOME
```

已解压完成的数据：

```text
$STABLEWM_HOME/datasets/pusht_expert_train.h5
$STABLEWM_HOME/datasets/tworoom.h5
```

保留的下载压缩包目录：

```text
$STABLEWM_HOME/downloads/lewm-pusht
$STABLEWM_HOME/downloads/lewm-tworooms
```

PushT 下载源文件：

```text
$STABLEWM_HOME/downloads/lewm-pusht/pusht_expert_train.h5.zst
```

TwoRoom 下载源文件：

```text
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
<a id="archive"></a>
### 7. 实验记录与归档

建议每次实验单独放进：

```text
experiments/YYYY-MM-DD_task_stage/
```

例如：

```text
experiments/2026-07-01_pusht_random/
```

目录里可以放：

- `README.md`
- `command.txt`
- `config.yaml`
- `results.txt`
- `notes.md`
- `artifacts/`

自动归档通式：

```bash
bash scripts/archive_experiment.sh <experiment_name> "<command>"
```

当前脚本会自动做这些事：

- 创建 `experiments/<experiment_name>/`
- 写入 `README.md`
- 写入 `command.txt`
- 把根目录下的 `pusht_results.txt` 或 `tworoom_results.txt` 移进去
- 把根目录下的 `*.mp4` 移到 `artifacts/`
- 复制对应的 `config/eval/*.yaml` 到 `config.yaml`

归档后，根目录不应再保留这次实验的：

- `pusht_results.txt`
- `tworoom_results.txt`
- `env_*.mp4`


<a id="done-items"></a>
### 8. 当前已经完成的事项

- LeWM GitHub 仓库已 clone。
- 代码、数据目录和 conda 环境已迁移到 `/data/zflin`。
- `lewm` conda 环境已可用。
- PyTorch 已匹配服务器 CUDA 12.8，CUDA 可用。
- `stable_worldmodel` 和 `stable_pretraining` 已可导入。
- PushT 数据已解压并放入 `$STABLEWM_HOME/datasets/pusht_expert_train.h5`。
- TwoRoom 数据已解压并放入 `$STABLEWM_HOME/datasets/tworoom.h5`。
- PushT 和 TwoRoom 的压缩包暂时保留，未删除。
- `hdf5plugin` 已纳入当前可用环境，用于读取压缩 HDF5 数据。
- HDF5 数据集的可用路径约定为 `$STABLEWM_HOME/datasets/*.h5`。
- `eval.py` 已改为使用新版 `stable_worldmodel.data.load_dataset()`。
- 已确认可用的单卡/多卡选择方式是 `CUDA_VISIBLE_DEVICES=...`。
- 仓库内新增了 `experiments/` 目录，用来统一存放每次实验的记录和产物。
- 仓库内新增了 `scripts/archive_experiment.sh`，可以一键把实验命令、结果和视频移动归档到 `experiments/<name>/`。


<a id="improvements"></a>
### 9. 本次复现实验的关键改进

#### 9.1 HDF5 数据读取

当前 `stable_worldmodel` 版本不再暴露旧的 `HDF5Dataset` 类，评估脚本需要使用新版
`swm.data.load_dataset()`。

同时，读取压缩 HDF5 时需要确保环境里有：

```bash
pip install hdf5plugin
```

#### 9.2 数据放置约定

当前可用约定是把 HDF5 数据放到：

```text
$STABLEWM_HOME/datasets/
```

例如：

```text
$STABLEWM_HOME/datasets/pusht_expert_train.h5
$STABLEWM_HOME/datasets/tworoom.h5
```

#### 9.3 选卡方式

单卡运行：

```bash
CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=random eval.num_eval=2
```

训练时配合单卡参数更稳：

```bash
CUDA_VISIBLE_DEVICES=3 python train.py data=pusht trainer.devices=1
```

#### 9.4 当前推荐的验证顺序

1. 检查 `hdf5plugin` 是否安装。
2. 确认 `.h5` 在 `$STABLEWM_HOME/datasets/`。
3. 用 `CUDA_VISIBLE_DEVICES` 固定一张空闲卡。
4. 先跑 `policy=random` 的小规模评估。
5. 再跑 checkpoint 的小规模评估。
6. 最后再做 1 epoch 训练冒烟测试。

<a id="part-2"></a>
## 第二部分 未来计划

<a id="smoke-test"></a>
### 10.1 先做数据与环境冒烟测试

先跑 PushT random policy 小评估：

```bash
python eval.py --config-name=pusht.yaml policy=random eval.num_eval=2
```

这一步不需要 LeWM checkpoint，主要检查：

- `eval.py` 能否启动。
- `stable_worldmodel` 环境能否创建。
- `$STABLEWM_HOME/datasets/pusht_expert_train.h5` 能否被读取。
- evaluation pipeline 能否完整跑完并写结果。

如果 PushT random 通过，再跑 TwoRoom random：

```bash
python eval.py --config-name=tworoom.yaml policy=random eval.num_eval=2
```

<a id="checkpoint"></a>
### 10.2 准备作者 checkpoint

如果要复现 LeWM planning，需要对应 checkpoint。

PushT 评估需要：

```text
$STABLEWM_HOME/pusht/lewm_object.ckpt
```

TwoRoom 评估需要：

```text
$STABLEWM_HOME/tworoom/lewm_object.ckpt
```

评估命令中的 `policy` 不写 `_object.ckpt` 后缀。

正确：

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

错误：

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

<a id="small-eval"></a>
### 10.3 跑 checkpoint 小规模评估

PushT 小规模调试：

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=2 solver.num_samples=50 solver.n_steps=5
```

TwoRoom 小规模调试：

```bash
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm eval.num_eval=2 solver.num_samples=50 solver.n_steps=5
```

确认无误后再跑默认评估：

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm
```

<a id="train-smoke"></a>
### 10.4 跑 1 epoch 训练冒烟测试

评估链路通了以后，再测训练链路：

```bash
python train.py data=pusht trainer.max_epochs=1 loader.batch_size=16
```

TwoRoom 训练冒烟测试：

```bash
python train.py data=tworoom trainer.max_epochs=1 loader.batch_size=16
```

如果成功，再考虑默认训练：

```bash
python train.py data=pusht
python train.py data=tworoom
```
