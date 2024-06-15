import fire

from engine import Trainer


def run(
    cfg: str,
    is_train: bool = True,
    start_wandb: bool = True,
):
    if is_train:
        runner = Trainer(cfg, start_wandb)
    else:
        raise NotImplementedError("Not implemented tester object yet.")
    runner()


if __name__ == "__main__":
    fire.Fire(run)