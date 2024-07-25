import fire

from engine import Trainer, Tester


def run(
    cfg: str,
    is_train: bool = True,
    start_wandb: bool = True,
):
    if is_train:
        runner = Trainer(cfg, start_wandb)
    else:
        runner = Tester(
            ckpt="ckpts/latest_ckpt.pth",
            patch_mode=False,
            k=32,
            sz=384,
            data_path="./challenge/validation",
            denim_type="dncm",
        )
    runner()


if __name__ == "__main__":
    fire.Fire(run)
