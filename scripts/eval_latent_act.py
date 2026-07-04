import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from latent_act.eval import _allow_new_hydra_overrides, _set_default_hydra_dir, run


if __name__ == "__main__":
    _allow_new_hydra_overrides(("policy_ckpt", "lewm_policy", "device"))
    _set_default_hydra_dir("eval_latent_act")
    run()

