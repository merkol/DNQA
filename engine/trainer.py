import os
import yaml
import torch
import glog as log
import wandb
from pathlib import Path
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

from dncm import DNCM, Encoder, DeNIM_StyleSwap_to_Canon, RelativeRankingLoss
from dataset import UHD_IQA


class Trainer:
    def __init__(self, cfg, start_wandb: bool) -> None:
        self._setup(cfg)
        self._init_parameters()
        self.start_wandb = start_wandb
        if self.start_wandb:
            self.wandb = wandb
            self.wandb.init(
                project=self.PROJECT_NAME,
                resume=self.INIT_FROM is not None,
                notes=str(self.LOG_DIR),
                config=self.cfg,
                entity=self.ENTITY,
            )
            self.wandb.define_metric("epoch")
            self.wandb.define_metric("epoch_loss", step_metric="epoch")
            self.wandb.define_metric("lr", step_metric="epoch")
            self.wandb.define_metric("val_loss", step_metric="epoch")

        self.transform = transforms.Compose(
            [
                transforms.RandomCrop((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.dataset = UHD_IQA(
            root=self.DATASET_ROOT, transform=self.transform, subset="training"
        )
        split_ratio = 0.9
        train_size = int(split_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = data.random_split(
            self.dataset, [train_size, test_size]
        )
        # self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE)
        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=self.SHUFFLE,
            num_workers=self.NUM_WORKERS,
        )
        self.val_loader = data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
        )
        self.to_pil = transforms.ToPILImage()

        print("Train dataset size: ", len(self.train_dataset))
        print("Val dataset size: ", len(self.val_dataset))

        if self.ARCH_TYPE == "dncm":
            self.DNCM = DNCM(self.k)
        elif self.ARCH_TYPE == "style_swap":
            self.DNCM = DeNIM_StyleSwap_to_Canon(self.k)

        self.encoder = Encoder((self.sz, self.sz), self.k)

        self.optimizer = torch.optim.Adam(
            list(self.DNCM.parameters()) + list(self.encoder.parameters()),
            lr=self.LR,
            betas=self.BETAS,
        )
        if self.SCHEDULER_TYPE == "step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.SCHEDULER_STEP, gamma=0.1
            )
        elif self.SCHEDULER_TYPE == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.SCHEDULER_T_MAX
            )
        elif self.SCHEDULER_TYPE == "cosine_restart":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.SCHEDULER_T_0
            )

        self.current_epoch = 0
        if self.INIT_FROM is not None and self.INIT_FROM != "":
            log.info("Checkpoints loading from ckpt file...")
            self.load_checkpoints(self.INIT_FROM)
        self.check_and_use_multi_gpu()
        self.loss = RelativeRankingLoss().cuda()

    def _setup(self, cfg):
        with open(cfg, "r") as stream:
            self.cfg = yaml.safe_load(stream)

    def _init_parameters(self):
        self.k = int(self.cfg["k"])
        self.sz = int(self.cfg["sz"])
        self.LR = float(self.cfg["LR"])
        self.BETAS = (self.cfg["BETA1"], self.cfg["BETA2"])
        self.NUM_GPU = int(self.cfg["NUM_GPU"])
        self.DATASET_ROOT = Path(self.cfg["DATASET_ROOT"])
        self.BATCH_SIZE = int(self.cfg["BATCH_SIZE"])
        self.EPOCHS = int(self.cfg["EPOCHS"])
        self.IMAGE_SIZE = int(self.cfg["IMAGE_SIZE"])
        self.ARCH_TYPE = self.cfg["ARCH_TYPE"]
        self.BACKBONE_ARCH = self.cfg["BACKBONE_ARCH"]
        self.SCHEDULER_STEP = list(self.cfg["SCHEDULER_STEP"])
        self.SCHEDULER_GAMMA = float(self.cfg["SCHEDULER_GAMMA"])
        self.SCHEDULER_T_0 = int(self.cfg["SCHEDULER_T_0"])
        self.SCHEDULER_T_MAX = int(self.cfg["SCHEDULER_T_MAX"])
        self.SCHEDULER_TYPE = self.cfg["SCHEDULER_TYPE"]
        self.SHUFFLE = bool(self.cfg["SHUFFLE"])
        self.NUM_WORKERS = int(self.cfg["NUM_WORKERS"])
        self.CKPT_DIR = Path(self.cfg["CKPT_DIR"])
        self.INIT_FROM = self.cfg["INIT_FROM"]
        self.PROJECT_NAME = self.cfg["PROJECT_NAME"]
        self.LOG_DIR = Path(self.cfg["LOG_DIR"])
        self.ENTITY = self.cfg["ENTITY"]

    def __call__(self):
        self.run()

    def run(self):
        for e in range(self.EPOCHS):
            log.info(f"Epoch {e+1}/{self.EPOCHS}")
            epoch_loss = 0
            val_loss = 0
            for step, (I, quality) in enumerate(
                tqdm(self.train_loader, total=len(self.train_loader))
            ):
                self.optimizer.zero_grad()

                I = I.float().cuda()
                quality = quality.float().cuda()

                d = self.encoder(I)
                Z = self.DNCM(I, d)
                Z = Z.view(-1)
                loss = self.loss(Z, quality)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if self.start_wandb:
                    self.wandb.log(
                        {
                            "loss": loss.item(),
                        },
                        commit=True,
                    )
                if self.SCHEDULER_TYPE == "step" or self.SCHEDULER_TYPE == "cosine":
                    self.scheduler.step()
            if self.SCHEDULER_TYPE == "cosine_restart":
                self.scheduler.step(e + step / len(self.train_loader))

            self.do_checkpoint()
            if self.start_wandb:
                self.wandb.log(
                    {
                        "epoch": e,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "epoch_loss": epoch_loss / len(self.train_loader),
                    },
                    commit=True,
                )

            for step, (I, quality) in enumerate(
                tqdm(self.val_loader, total=len(self.val_loader))
            ):
                I = I.float().cuda()
                quality = quality.float().cuda()

                d = self.encoder(I)
                Z = self.DNCM(I, d)
                Z = Z.view(-1)
                loss = self.loss(Z, quality)
                val_loss += loss.item()

            if self.start_wandb:
                self.wandb.log(
                    {
                        "val_loss": val_loss / len(self.val_loader),
                    },
                    commit=True,
                )

        if self.start_wandb:
            self.upload_checkpoint()

    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.NUM_GPU > 1:
            log.info(f"Using {torch.cuda.device_count()} GPUs...")
            self.DNCM = torch.nn.DataParallel(self.DNCM).cuda()
            self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        else:
            log.info(f"GPU ID: {torch.cuda.current_device()}")
            self.DNCM = self.DNCM.cuda()
            self.encoder = self.encoder.cuda()

    def do_checkpoint(self):
        os.makedirs(str(self.CKPT_DIR), exist_ok=True)
        checkpoint = {
            "epoch": self.current_epoch,
            "DCNM": self.DNCM.module.state_dict()
            if isinstance(self.DNCM, torch.nn.DataParallel)
            else self.DNCM.state_dict(),
            "encoder": self.encoder.module.state_dict()
            if isinstance(self.encoder, torch.nn.DataParallel)
            else self.encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, str(self.CKPT_DIR / "latest_ckpt.pth"))

    def upload_checkpoint(self):
        self.wandb.save(str(self.CKPT_DIR / "latest_ckpt.pth"))

    def load_checkpoints(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.DNCM.load_state_dict(checkpoints["DCNM"])
        self.encoder.load_state_dict(checkpoints["encoder"])
        self.optimizer.load_state_dict(checkpoints["optimizer"])
        self.init_epoch = checkpoints["epoch"]
        self.optimizers_to_cuda()

    def optimizers_to_cuda(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
