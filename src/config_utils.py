from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_run_config(path):
    """Compose a repository model configuration and its Hydra defaults."""
    path = Path(path).expanduser().resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        cfg = compose(config_name=path.stem)
    OmegaConf.resolve(cfg)
    return cfg


def load_calibration(path=None) -> tuple[float, float, float]:
    """Read the shared SynLF disparity-to-depth coefficients."""
    path = Path(path) if path is not None else REPO_ROOT / "configs" / "data" / "synlf.yaml"
    cfg = OmegaConf.load(path)
    return tuple(float(cfg[key]) for key in ("a", "b", "c"))
