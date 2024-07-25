from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, CenterCrop, RandomCrop, Resize
from torch.utils import data
import pandas as pd


class UHD_IQA(Dataset):
    def __init__(self, root, transform=None, subset=None) -> None:
        super().__init__()
        self.imgs = list((Path(root) / subset).iterdir())
        self.metadata = pd.read_csv(Path(root) / "uhd-iqa-training-metadata.csv")
        self.transform = transform
        self.subset = subset

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.subset == "training":
            label = self.metadata.loc[
                self.metadata["image_name"] == self.imgs[index].name, "quality_mos"
            ].values[0]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    transforms = Compose(
        [
            Resize((968, 3840)),
            # CenterCrop(968),
            ToTensor(),
        ]
    )
    dataset = UHD_IQA(
        root="/home/vgl/Research/DNQA/challenge",
        transform=transforms,
        subset="training",
    )
    print("Dataset length: ", len(dataset))

    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)

    unique_widths = []
    for img, label in dataloader:
        width = img.shape[-1]
        print(img.shape, label)
        break

    # for i in range(len(dataset)):
    #     img, label = dataset.__getitem__(i)
    #     print(img.shape, label)
