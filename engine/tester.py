import fire
import torch
from typing import Union
from pathlib import Path
from torchvision import transforms
from PIL import Image
import pandas as pd
from dncm import DNCM, Encoder, DeNIM_StyleSwap_to_Canon
from tqdm import tqdm


class Tester:
    def __init__(self, ckpt, patch_mode, k, sz, data_path, denim_type) -> None:
        self.ckpt = ckpt
        self.patch_mode = patch_mode
        self.k = k
        self.sz = sz
        self.data_path = data_path
        self.denim_type = denim_type
        self.inference_size = 2048

        self.transform = transforms.Compose(
            [
                transforms.Resize(1024),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self):
        self.run()

    def run(self):
        print(self.ckpt)
        dncm, encoder = self._build_model(
            ckpt=self.ckpt, k=self.k, sz=self.sz, denim_type=self.denim_type
        )
        images = list(Path(self.data_path).iterdir())
        data = {}
        for img in tqdm(images):
            ## get image name
            image_name = img.name
            img = Image.open(img).convert("RGB")
            img = self.transform(img).unsqueeze(0).cuda()
            # print(img.shape)
            if self.patch_mode:
                patches = self._divide_patches(img, 4)
                scores = []
                for patch in patches:
                    with torch.no_grad():
                        features = encoder(patch)
                        score = dncm(patch, features)
                        scores.append(score.item())
                data[image_name] = sum(scores) / len(scores)
            else:
                with torch.no_grad():
                    features = encoder(img)
                    score = dncm(img, features)
                    data[image_name] = score.item()

        self._export_csv(data)

    def _export_csv(self, data):
        df = pd.DataFrame(data.items(), columns=["image_name", "quality_mos"])
        df["image_int"] = df["image_name"].apply(lambda x: int(x.split(".")[0]))
        df = df.sort_values("image_int")
        df = df.drop("image_int", axis=1)
        df.to_csv("challenge/submission.csv", index=False)

    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _divide_patches(self, img, patch_count):
        bs, _, H, W = img.shape
        h = H // patch_count
        w = W // patch_count
        patches = []
        for i in range(patch_count):
            for j in range(patch_count):
                patch = img[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w]
                patches.append(patch)
        return patches

    def _build_model(
        self,
        ckpt: Union[str, Path],
        k: int = 16,
        sz: int = 512,
        denim_type: str = "dncm",
    ):
        if denim_type == "dncm":
            dncm = DNCM(k)
        elif denim_type == "style_swap":
            dncm = DeNIM_StyleSwap_to_Canon(k)
        encoder = Encoder((sz, sz), k)
        checkpoints = torch.load(ckpt)
        dncm.load_state_dict(checkpoints["DCNM"])
        encoder.load_state_dict(checkpoints["encoder"])
        dncm = dncm.cuda()
        encoder = encoder.cuda()
        return dncm, encoder
