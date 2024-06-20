import fire
import torch
from typing import Union
from pathlib import Path
from torchvision.transforms import ToTensor
from PIL import Image
import pandas as pd
from dncm import DNCM, Encoder, DeNIM_StyleSwap_to_Canon
from tqdm import tqdm


to_tensor = ToTensor()

def build_model(
    ckpt: Union[str, Path],
    k: int = 16,
    sz: int = 252,
    denim_type: str = "style_swap",
):
    if denim_type == "dncm":
        dncm = DNCM(k)
    elif denim_type == "style_swap":
        dncm = DeNIM_StyleSwap_to_Canon(k)
    encoder = Encoder((sz, sz), k, "vit_base_patch8_224")
    checkpoints = torch.load(ckpt)
    dncm.load_state_dict(checkpoints["DCNM"])
    encoder.load_state_dict(checkpoints["encoder"])
    dncm = dncm.cuda()
    encoder = encoder.cuda()
    return dncm, encoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def divide_patches(img, patch_count):
    bs, _, H, W = img.shape
    h = H // patch_count
    w = W // patch_count
    patches = []
    for i in range(patch_count):
        for j in range(patch_count):
            patch = img[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w]
            patches.append(patch)
    return patches

def run(
    ckpt: Union[str, Path],
    patch_mode: bool = False,
    k: int = 16,
    sz: int = 224,
    data_path: Union[str, Path] = "challenge/validation",
):
    dncm, encoder = build_model(ckpt=ckpt, k=k, sz=sz)
    images = list(Path(data_path).iterdir())
    data = {}
    for img in tqdm(images):
        ## get image name
        image_name = img.name
        img = Image.open(img).convert("RGB")
        img = img.resize((768, 768))
        img = to_tensor(img).unsqueeze(0).cuda()
        
        if patch_mode:
            patches = divide_patches(img, 4)
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
    
    df = pd.DataFrame(data.items(), columns=["image_name", "quality_mos"])
    df["image_int"] = df["image_name"].apply(lambda x: int(x.split(".")[0]))
    df = df.sort_values("image_int")
    df = df.drop("image_int", axis=1)
    df.to_csv("challenge/submission.csv", index=False)
    
if __name__ == "__main__":
    fire.Fire(run)