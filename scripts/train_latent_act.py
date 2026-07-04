import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from latent_act.train import _allow_new_hydra_overrides, _set_default_hydra_dir, run


if __name__ == "__main__":
    _set_default_hydra_dir("train_latent_act")
    _allow_new_hydra_overrides(("dataset", "output", "max_samples", "device"))
    run()

