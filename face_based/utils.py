# from pathlib import Path
# from sklearn.model_selection import train_test_split
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class RafDataset(Dataset):
    def __init__(self, phase, raf_pth, transform=None):
        self.raf_path = raf_pth
        self.phase = phase
        self.transform = transform

        df = pd.read_csv(
            os.path.join(self.raf_path, "EmoLabel/list_patition_label.txt"),
            sep=" ",
            header=None,
            names=["name", "label"],
        )

        if phase == "train":
            self.data = df[df["name"].str.startswith("train")]
        else:
            self.data = df[df["name"].str.startswith("test")]

        file_names = self.data.loc[:, "name"].values
        self.label = (
            self.data.loc[:, "label"].values - 1
        )  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, "Image/aligned", f)
            self.file_paths.append(path)

    def get_labels(self):
        return self.label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = np.array(Image.open(path).convert("RGB"))
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def make_dataloaders(params_data, num_workers=0, transforms=None):

    CLASSES = [
        "Surprise",
        "Fear",
        "Disgust",
        "Happiness",
        "Sadness",
        "Anger",
        "Neutral",
    ]

    if transforms != None:
        train_tf, val_tf = transforms
    else:
        train_tf, val_tf = (None, None)

    train_dataset = RafDataset(
        phase="train", raf_pth=params_data.dataset_path, transform=train_tf
    )
    test_dataset = RafDataset(
        phase="test", raf_pth=params_data.dataset_path, transform=val_tf
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params_data["batch_size"],
        # sampler=train_sampler,
        shuffle=True,
        num_workers=params_data["workers"],
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params_data["batch_size"],
        shuffle=False,
        num_workers=params_data["workers"],
        pin_memory=True,
    )
    test_loader = val_loader

    return [train_loader, val_loader, test_loader, CLASSES]
