import sys
from abc import abstractmethod
from pathlib import Path

import torch
from backbones.MobileFaceNets import MobileFaceNet
from paper_models.RUL import RUL
from torch import nn
from torchvision import models as torchmodels


class BackboneMaker:
    backbone_weights = {
        "MobileFaceNet": "/home/mangaboba/environ/fer/itmo-emotion-recognition/face-based/backbone_weights/MobileFaceNet_MSceleb.pt",
        "GhostNet": "",
        "ReXNetv1": "",
        "SwinT": "",
        "LiteCNN29": "",
        "ResNet18": "../backbone_weights/resnet18_msceleb.pth",
    }

    @staticmethod
    def get_backbone(model_name):
        # torch.load(Path(__file__).parent / \
        d = torch.load(BackboneMaker.backbone_weights[model_name])["state_dict"]
        d = {key: val for key, val in d.items() if key.startswith("backbone")}
        valid = {key.split("backbone.", 1)[1]: val for key, val in d.items()}

        bbone = None

        if model_name == "ResNet18":
            bbone = torchmodels.resnet18()
            bbone.load_state_dict(valid, strict=False)
            bbone = nn.Sequential(*list(bbone.children())[:-2])

        elif model_name == "MobileFaceNet":
            bbone = MobileFaceNet(512, 7, 7)
            bbone.load_state_dict(valid)
            bbone = nn.Sequential(*list(bbone.children())[:-4])

        return bbone


class Model:
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def getModel(self, cfg):
        pass


class MobileRUL(Model):
    def __init__(self, cfg):
        super().__init__("MobileRUL")
        self.__cfg = cfg
        # cfg['out_feat'] == 64, RUL(out_feat) default
        feature_extractor = RUL(bbone=BackboneMaker.get_backbone("MobileFaceNet"))
        self.__model = {
            "feature_extrator": feature_extractor,
            "classifier": torch.nn.Linear(64, 7),
        }

    def getModel(self):
        return self.__model
