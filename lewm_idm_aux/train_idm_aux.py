from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def _safe_weight(cfg: DictConfig, path: str, default: float = 0.0) -> float:
    node = cfg
    for key in path.split("."):
        if key not in node:
            return default
        node = node[key]
    return float(node)


def _latent_stats(emb: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = emb.detach().reshape(-1, emb.size(-1)).float()
    var = flat.var(dim=0, unbiased=False)
    return {
        "latent/variance_mean": var.mean(),
        "latent/variance_min": var.min(),
        "latent/variance_max": var.max(),
        "latent/norm_mean": flat.norm(dim=-1).mean(),
    }


def lejepa_idm_forward(self, batch, stage: str, cfg: DictConfig):
    """Encode observations, predict next latents, and add optional IDM loss."""
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    sig_weight = _safe_weight(cfg, "loss.sigreg.weight", 0.0)
    idm_weight = _safe_weight(cfg, "loss.idm.weight", 0.0)
    idm_enabled = bool(cfg.loss.get("idm", {}).get("enabled", False))

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = self.sigreg(emb.transpose(0, 1)) if sig_weight > 0 else pred_loss.new_tensor(0.0)

    idm_loss = pred_loss.new_tensor(0.0)
    if idm_enabled and idm_weight > 0:
        z_t = emb[:, :-1]
        z_next = emb[:, 1:]
        target_action = batch["action"][:, :-1]
        pred_action = self.model.inverse_decode(z_t, z_next)
        idm_loss = F.mse_loss(pred_action, target_action)
        output["pred_action"] = pred_action

    total_loss = pred_loss + sig_weight * sigreg_loss + idm_weight * idm_loss
    output.update(
        {
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "idm_loss": idm_loss,
            "loss": total_loss,
        }
    )

    log_dict = {
        f"{stage}/loss": total_loss.detach(),
        f"{stage}/pred_loss": pred_loss.detach(),
        f"{stage}/sigreg_loss": sigreg_loss.detach(),
        f"{stage}/idm_loss": idm_loss.detach(),
    }
    log_dict.update({f"{stage}/{k}": v for k, v in _latent_stats(emb).items()})
    self.log_dict(log_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="configs", config_name="lewm_idm_aux")
def run(cfg: DictConfig):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        action_block_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.model.action_encoder.input_dim = action_block_dim
        if "inverse_decoder" in cfg.model and cfg.model.inverse_decoder is not None:
            cfg.model.inverse_decoder.action_dim = action_block_dim

        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    world_model = hydra.utils.instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_idm_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=cfg.checkpoint.epoch_interval,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == "__main__":
    run()

