import argparse
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import hydra
import pyrootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers.logger import Logger

from modules.losses.losses import load_loss
from modules.metrics.metrics import load_metrics

log = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): Callbacks config.

    Returns:
        list[Callback]: list with all instantiated callbacks.
    """

    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_conf in callbacks_cfg.values():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg (DictConfig): Loggers config.

    Returns:
        list[LightningLoggerBase]: list with all instantiated loggers.
    """

    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for lg_conf in logger_cfg.values():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def get_args_parser() -> argparse.ArgumentParser:
    """Get parser for additional Hydra's command line flags."""
    parser = argparse.ArgumentParser(description="Additional Hydra's command line flags parser.")

    parser.add_argument(
        "--config-path",
        "-cp",
        nargs="?",
        default=None,
        help="""Overrides the config_path specified in hydra.main().
                    The config_path is absolute or relative to the Python file declaring @hydra.main()""",
    )

    parser.add_argument(
        "--config-name",
        "-cn",
        nargs="?",
        default=None,
        help="Overrides the config_name specified in hydra.main()",
    )

    parser.add_argument(
        "--config-dir",
        "-cd",
        nargs="?",
        default=None,
        help="Adds an additional config dir to the config search path",
    )
    return parser


def register_custom_resolvers(version_base: str, config_path: str, config_name: str) -> Callable:
    """Optional decorator to register custom OmegaConf resolvers. It is
    excepted to call before `hydra.main` decorator call.

    Replace resolver: To avoiding copying of loss and metric names in configs,
    there is custom resolver during hydra initialization which replaces
    `__loss__` to `loss.__class__.__name__` and `__metric__` to
    `main_metric.__class__.__name__` For example: ${replace:"__metric__/valid"}
    Use quotes for defining internal value in ${replace:"..."} to avoid grammar
    problems with hydra config parser.

    Args:
        version_base (str): Hydra version base.
        config_path (str): Hydra config path.
        config_name (str): Hydra config name.

    Returns:
        Callable: Decorator that registers custom resolvers before running
            main function.
    """

    # parse additional Hydra's command line flags
    parser = get_args_parser()
    args, _ = parser.parse_known_args()
    if args.config_path:
        config_path = args.config_path
    if args.config_dir:
        config_path = args.config_dir
    if args.config_name:
        config_name = args.config_name

    # register last_ckpt resolver: ${last_ckpt:logs/train/runs}
    # returns the most recently modified .ckpt file from the latest timestamped run dir
    if not OmegaConf.has_resolver("last_ckpt"):
        _root = pyrootutils.find_root(
            search_from=Path(__file__).parent,
            indicator=[".git", "pyproject.toml"],
        )

        def _find_last_checkpoint(runs_dir: str) -> str:
            runs_path = _root / runs_dir
            if not runs_path.exists():
                raise FileNotFoundError(f"Runs directory not found: {runs_path}")
            run_dirs = sorted([d for d in runs_path.iterdir() if d.is_dir()])
            if not run_dirs:
                raise FileNotFoundError(f"No run directories found in: {runs_path}")
            for run_dir in reversed(run_dirs):
                for ckpt_dirname in ("checkpoints", "ckpts"):
                    ckpt_dir = run_dir / ckpt_dirname
                    if not ckpt_dir.exists():
                        continue
                    last_ckpt = ckpt_dir / "last.ckpt"
                    if last_ckpt.exists():
                        return str(last_ckpt)
                    ckpts = [f for f in ckpt_dir.iterdir() if f.suffix == ".ckpt"]
                    if ckpts:
                        return str(max(ckpts, key=lambda p: p.stat().st_mtime))
            raise FileNotFoundError(f"No checkpoints found under: {runs_path}")

        OmegaConf.register_new_resolver("last_ckpt", _find_last_checkpoint)

    # register of replace resolver
    if not OmegaConf.has_resolver("replace"):
        with initialize_config_dir(version_base=version_base, config_dir=config_path):
            cfg = compose(config_name=config_name, return_hydra_config=True, overrides=[])
        cfg_tmp = cfg.copy()
        loss = load_loss(cfg_tmp.module.loss)
        metric, _ = load_metrics(cfg_tmp.module.metrics)
        GlobalHydra.instance().clear()

        OmegaConf.register_new_resolver(
            "replace",
            lambda item: item.replace("__loss__", loss.__class__.__name__).replace(
                "__metric__", metric.__class__.__name__
            ),
        )

    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)

        return wrapper

    return decorator
