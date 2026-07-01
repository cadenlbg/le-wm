# Scripts

## archive_experiment.sh

把一次实验的命令、结果、配置和 mp4 视频归档到 `experiments/<name>/`。

### 用法

```bash
bash scripts/archive_experiment.sh 2026-07-01_pusht_random "CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=random eval.num_eval=2"