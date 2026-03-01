import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class CSVDataset(Dataset):
    def __init__(self, df, root, transform=None, path_col="path", label_col="label"):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        p = self.df.loc[i, self.path_col]
        y = int(self.df.loc[i, self.label_col])
        full = p if os.path.isabs(p) else os.path.join(self.root, p)
        img = Image.open(full).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y

def try_imagefolder(dataset_dir):
    for name in ["train", "Train", "training", "Training"]:
        d = os.path.join(dataset_dir, name)
        if os.path.isdir(d):
            try:
                ds = ImageFolder(d)
                if len(ds.classes) >= 2:
                    return d
            except:
                pass
    return None

def find_csv(dataset_dir):
    for f in os.listdir(dataset_dir):
        if f.lower().endswith(".csv"):
            return os.path.join(dataset_dir, f)
    return None