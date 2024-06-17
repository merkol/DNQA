import fire
import torch
from typing import Union
from pathlib import Path
from torchvision.transforms import ToTensor
from PIL import Image
import pandas as pd
from thop import profile
from dncm import DNCM, Encoder
from tqdm import tqdm


to_tensor = ToTensor()

def build_model(
    ckpt: Union[str, Path],
    k: int = 16,
    sz: int = 252,
):
    dncm = DNCM(k)
    encoder = Encoder((sz, sz), k)
    checkpoints = torch.load(ckpt)
    dncm.load_state_dict(checkpoints["DCNM"])
    encoder.load_state_dict(checkpoints["encoder"])
    dncm = dncm.cuda()
    encoder = encoder.cuda()
    return dncm, encoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run(
    ckpt: Union[str, Path],
    k: int = 16,
    sz: int = 252,
    data_path: Union[str, Path] = "challenge/validation",
):
    dncm, encoder = build_model(ckpt=ckpt, k=k, sz=sz)
    images = list(Path(data_path).iterdir())
    data = {}
    for img in tqdm(images):
        ## get image name
        image_name = img.name
        img = Image.open(img).convert("RGB")
        img = to_tensor(img).unsqueeze(0).cuda()
        with torch.no_grad():
            features = encoder(img)
            score = dncm(img, features)
        data[image_name] = score.item()
    
    df = pd.DataFrame(data.items(), columns=["image_name", "quality_mos"])
    df["image_int"] = df["image_name"].apply(lambda x: int(x.split(".")[0]))
    df = df.sort_values("image_int")
    df = df.drop("image_int", axis=1)
    df.to_csv("challenge/submission.csv", index=False)
    
if __name__ == "__main__":
    fire.Fire(run)