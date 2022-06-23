import argparse
import os
from datetime import datetime

import albumentations as A
import pytorch_lightning as pl
import torch
import wandb
from dataTransforms import get_transforms
from dvclive.lightning import DvcLiveLogger
from lightning_model_wrap import RULplWrap
from models_factory import create_factory
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import make_dataloaders


def jitBestModel(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


def load_model_from_ckpt(ckpt_path):
    try:
        model_pl = ModelPL.load_from_checkpoint(checkpoint_path=ckpt_path)
    except FileNotFoundError:
        raise FileNotFoundError("Invalid ckpt path.")

    return model_pl


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model train stage.")
    parser.add_argument("-t", required=False, action="store_true", help="Test mode.")
    parser.add_argument("-runname", required=False, help="Wandb run name prefix.")
    args = parser.parse_args()

    os.sys.path.insert(0, "config")
    try:
        from config import cfg_mdl, cfg_run, cfg_sys
    except LookupError as k:
        raise ImportError("Failed to import config.") from k
    cfg_logs = cfg_sys.logs_cgf

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    wandb.login(key=cfg_logs.wandb_api_key)  # os.environ['APIKEY'].strip())
    wandb.init()
    print(str(args.runname))
    run_name = args.n if not cfg_logs.run_name else cfg_logs.run_name
    wandb.run.name = "_".join(
        (
            run_name,
            f"({wandb.run.id})",
            datetime.now().strftime("%d.%m.%Y_%H:%M:%S"),
        )
    )

    wandb_logger = WandbLogger(
        project=cfg_logs["wandb_project"],
        entity=cfg_logs["wandb_entity"],
        log_model=False,
    )
    # dvclive_logger = DvcLiveLogger(
    #                     path = 'dvc_log_check')

    val_checkpoint = ModelCheckpoint(
        filename=cfg_run.model.model_version + "_valloss-{epoch}-{step}-{val_loss}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    f1w_checkpoint = ModelCheckpoint(
        filename=cfg_run.model.model_version
        + "_valf1w-{epoch}-{step}-{val_f1_weighted}",
        monitor="val_f1_weighted",
        mode="max",
        save_top_k=1,
    )
    last_checkpoint = ModelCheckpoint(
        filename=cfg_run.model.model_version + "_last-{epoch}-{step}-{val_loss}",
        monitor="val_f1_weighted",
        every_n_epochs=cfg_run.train.max_epochs,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )

    try:
        transform_from_yaml = A.load("config/transform.yaml", data_format="yaml")
    except KeyError:
        transform_from_yaml = None
        print("Transform.yaml configuration seems invalid. Transform set None.")
    except FileNotFoundError:
        transform_from_yaml = None
        print("Failed to find transform.yaml. Transform set None.")

    transforms = get_transforms(cfg_run.model.model_version)  # -> (train, val)
    print(transforms)
    (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        classes,
    ) = make_dataloaders(cfg_run.data, cfg_sys.system.num_workers, transforms)

    if (not cfg_run.model.ckpt_path) and (not args.t):
        modelFactory = create_factory(cfg_run.model.model_version)
        modelWrap = modelFactory.createModel(cfg_mdl)
        model = modelWrap.getModel()
        model_pl = RULplWrap(model, classes)
    elif cfg_run.model.ckpt_path:
        model_pl = load_model_from_ckpt(cfg_run.model.ckpt_path)

    trainer = pl.Trainer(
        default_root_dir=cfg_run.train["ckpt_dir"],
        gpus=cfg_run.train["gpus"],
        max_epochs=cfg_run.train["max_epochs"],
        max_steps=cfg_run.train["max_steps"],
        logger=[wandb_logger],
        callbacks=[val_checkpoint, f1w_checkpoint, last_checkpoint],
    )

    if not args.t:
        trainer.fit(model_pl, train_dataset_loader, val_dataset_loader)

    trainer.test(model_pl, test_dataset_loader)
