# Scripts

## archive_experiment.sh

把一次 CEM 或原始评估实验的命令、结果、配置和 mp4 视频归档到 `experiments/<name>/`。

### 用法

```bash
bash scripts/archive_experiment.sh 2026-07-01_pusht_random "CUDA_VISIBLE_DEVICES=3 python eval.py --config-name=pusht.yaml policy=random eval.num_eval=2"
```

## archive_latent_bc_experiment.sh

归档 latent BC 实验目录。默认实验根目录是 `/data/zflin/lewm_re/experiments`，也可以用 `LEWM_EXPERIMENTS_DIR` 覆盖。

归档内容包括：

- `README.md`
- `command.txt`
- `manifest.txt`
- `policy.pt` / `metrics.jsonl` / `pusht_results.txt` 的存在状态
- 根目录下的 `*.mp4`，移动到 `artifacts/`
- `config/eval/pusht.yaml`
- BC 相关 notes
- 最新的 BC Hydra runs，复制到 `hydra_outputs/`

### 用法

```bash
bash scripts/archive_latent_bc_experiment.sh 2026-07-03_pusht_latent_bc "python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt eval.num_eval=50"
```

如果想显式指定要归档的 Hydra run：

```bash
HYDRA_RUNS="/data/zflin/lewm_re/experiments/hydra/train_latent_bc/2026-07-03/19-10-00:/data/zflin/lewm_re/experiments/hydra/eval_latent_bc/2026-07-03/20-00-00" \
bash scripts/archive_latent_bc_experiment.sh 2026-07-03_pusht_latent_bc "python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt eval.num_eval=50"
```
