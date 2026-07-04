import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from latent_act.train import run
from omegaconf import OmegaConf


if __name__ == "__main__":
    cfg = OmegaConf.from_cli(sys.argv[1:])
    run(cfg)
