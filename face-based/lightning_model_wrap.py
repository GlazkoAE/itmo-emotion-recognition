import albumentations as A
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, ConfusionMatrix, F1Score
from torchmetrics.functional import precision_recall
from torchvision import models, transforms, utils


class ModelPL(pl.LightningModule):
    """
    Lightning wrap base class

    """

    def __init__(self, classes, cfg):
        super().__init__()
        self.classes = classes
        self.cfg = cfg
        self.save_hyperparameters()
        self.num_classes = len(classes)
        # self.f1score_micro = F1Score(num_classes = len(classes), average='micro')
        self.f1score_macro = F1Score(num_classes=self.num_classes, average="macro")
        self.f1score_weighted = F1Score(
            num_classes=self.num_classes, average="weighted"
        )
        # self.confusion_matrix = ConfusionMatrix(num_classes=len(classes))
        self.acc = Accuracy()

    def training_step(self):
        assert "Not implemented!"

    def _log_train(self, loss, logits, labels):
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", self.acc(logits, labels), on_epoch=True, on_step=False)
        self.log(
            "train_f1_weighted",
            self.f1score_weighted(logits, labels),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_f1_macro",
            self.f1score_macro(logits, labels),
            on_epoch=True,
            on_step=False,
        )

        precol_macro = precision_recall(
            logits, labels, num_classes=self.num_classes, average="macro"
        )
        precol_weighted = precision_recall(
            logits, labels, num_classes=self.num_classes, average="weighted"
        )

        self.log(
            "train_precision_macro",
            precol_macro[0].item(),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_recall_macro", precol_macro[1].item(), on_epoch=True, on_step=False
        )

        self.log(
            "train_precision_weighted",
            precol_weighted[0].item(),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_recall_weighted",
            precol_weighted[1].item(),
            on_epoch=True,
            on_step=False,
        )

        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=True,
            on_step=False,
        )

    def validation_step(self):
        assert "Not implemented!"

    def _log_validation(self, loss, logits, labels):
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_acc", self.acc(logits, labels), on_epoch=True, on_step=False)
        self.log(
            "val_f1_weighted",
            self.f1score_weighted(logits, labels),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_f1_macro",
            self.f1score_macro(logits, labels),
            on_epoch=True,
            on_step=False,
        )

        precol_macro = precision_recall(
            logits, labels, num_classes=self.num_classes, average="macro"
        )
        precol_weighted = precision_recall(
            logits, labels, num_classes=self.num_classes, average="weighted"
        )

        self.log(
            "val_precision_macro", precol_macro[0].item(), on_epoch=True, on_step=False
        )
        self.log(
            "val_recall_macro", precol_macro[1].item(), on_epoch=True, on_step=False
        )

        self.log(
            "val_precision_weighted",
            precol_weighted[0].item(),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val_recall_weighted",
            precol_weighted[1].item(),
            on_epoch=True,
            on_step=False,
        )

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp["preds"] for tmp in outputs])
        targets = torch.cat([tmp["target"] for tmp in outputs])

        self.logger[0].experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=preds.cpu().numpy(),
                    y_true=targets.cpu().numpy(),
                    class_names=self.classes,
                )
            }
        )

    def test_step(self):
        assert "Not implemented!"

    def test_epoch_end(self, outputs):
        preds = torch.cat([tmp["preds"] for tmp in outputs])
        targets = torch.cat([tmp["target"] for tmp in outputs])

        self.logger[0].experiment.log(
            {
                "conf_test": wandb.plot.confusion_matrix(
                    probs=preds.cpu().numpy(),
                    y_true=targets.cpu().numpy(),
                    class_names=self.classes,
                )
            }
        )

    def configure_optimizers(self):
        assert "Not implemented"


class RULplWrap(ModelPL):
    def __init__(self, model, classes, cfg=None):
        super().__init__(classes, cfg)
        self.model = model["feature_extrator"]
        self.fc = model["classifier"]

    # add-up loss
    def _mixup_criterion(self, y_a, y_b):
        return lambda criterion, pred: 0.5 * criterion(pred, y_a) + 0.5 * criterion(
            pred, y_b
        )

    def training_step(self, batch, batch_idx):

        imgs, labels = batch

        mixed_x, y_a, y_b, att1, att2 = self.model.train_forward(imgs, labels)
        logits = self.fc(mixed_x)

        criterion = nn.CrossEntropyLoss()
        loss_func = self._mixup_criterion(y_a, y_b)
        loss = loss_func(criterion, logits)

        self._log_train(loss, logits, labels)
        # self.log('train_f1_macro', self.f1score_macro(logits, y), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        outputs = self.model(imgs)
        logits = self.fc(outputs)

        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        self._log_validation(loss, logits, labels)

        return {"loss": loss, "preds": logits, "target": labels}

    def test_step(self, batch, batch_idx):
        imgs, labels = batch

        outputs = self.model(imgs)
        logits = self.fc(outputs)

        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        self._log_validation(loss, logits, labels)

        return {"loss": loss, "preds": logits, "target": labels}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters()},
                {"params": self.fc.parameters(), "lr": 0.01},
            ],
            lr=0.0001,
            weight_decay=0.0001,
            amsgrad=True,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }  # , 'monitor':"val_loss",
