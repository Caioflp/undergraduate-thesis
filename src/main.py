"""Entry point to run stuff.

Author: @Caioflp

"""
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    model.fit(dataset)
    model.make_plots(dataset)


if __name__ == "__main__":
    main()
