# LeWM IDM Auxiliary

This folder contains an isolated experiment branch for adding an inverse
dynamics auxiliary decoder to LeWM without modifying the original LeWM files.

## Idea

The auxiliary decoder predicts embedding-before raw action blocks from adjacent
LeWM latents:

\[
\hat{a}_t = D_\psi(z_t, z_{t+1})
\]

For PushT, the original LeWM config uses:

```text
frameskip = 5
action_dim = 2
raw action block dim = 10
```

The decoder input is only:

```text
concat(z_t, z_next)
```

It does not use \(z_{t+1} - z_t\).

## Files

```text
lewm_idm_aux/
  module_idm.py        # InverseDynamicsDecoder
  jepa_idm.py          # JEPAWithIDM wrapper
  train_idm_aux.py     # isolated training entrypoint
  configs/             # self-contained Hydra configs
```

The original files are imported but not edited:

```text
module.py
jepa.py
train.py
utils.py
```

## Loss

\[
\mathcal{L}
=
\mathcal{L}_{pred}
+
\lambda_{sig}\mathcal{L}_{SIGReg}
+
\lambda_{idm}\mathcal{L}_{IDM}
\]

\[
\mathcal{L}_{IDM}
=
\|D_\psi(z_t, z_{t+1}) - a_t\|_2^2
\]

## Run

From the repo root:

```bash
python -m lewm_idm_aux.train_idm_aux
```

Useful overrides:

```bash
python -m lewm_idm_aux.train_idm_aux \
  output_model_name=lewm_pred_sigreg_idm \
  loss.idm.weight=0.1 \
  wandb.enabled=true
```

Ablations:

```bash
python -m lewm_idm_aux.train_idm_aux --config-name ablation_pred_only
python -m lewm_idm_aux.train_idm_aux --config-name ablation_pred_idm
python -m lewm_idm_aux.train_idm_aux --config-name ablation_pred_sigreg_idm
```

