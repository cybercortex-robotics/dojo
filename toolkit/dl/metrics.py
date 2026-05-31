import hydra
from omegaconf import DictConfig
from torchmetrics import Metric


def load_metrics(metrics_cfg: DictConfig) -> tuple[Metric, Metric]:
    main_metric = hydra.utils.instantiate(metrics_cfg.main)
    if not metrics_cfg.get("valid_best"):
        raise RuntimeError(
            "Requires valid_best metric that would track best state of "
            "Main Metric. Usually it can be MaxMetric or MinMetric."
        )
    valid_metric_best = hydra.utils.instantiate(metrics_cfg.valid_best)

    return main_metric, valid_metric_best
