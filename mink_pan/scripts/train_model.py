import os
import subprocess
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from mink_pan.datasets.semantic_dataset import SemanticDatasetModule
from mink_pan.models.model import MinkPan
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@click.command()
@click.option("--w", type=str, default=None, required=False, help="weights to load")
@click.option(
    "--ckpt",
    type=str,
    default=None,
    required=False,
    help="checkpoint to resume training",
)
@click.option("--nuscenes", is_flag=True)
@click.option("--mini", is_flag=True, help="use mini split for nuscenes")
@click.option(
    "--seq",
    type=int,
    default=None,
    required=False,
    help="use a single sequence for train and val",
)
@click.option(
    "--id", type=str, default=None, required=False, help="set id of the experiment"
)
def main(w, ckpt, nuscenes, mini, seq, id):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg})
    cfg.git_commit_version = str(
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip()
    )

    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    if mini and nuscenes:
        cfg.NUSCENES.MINI = True
    if seq:
        cfg.TRAIN.ONLY_SEQ = seq
    if id:
        cfg.EXPERIMENT.ID = id

    data = SemanticDatasetModule(cfg)
    model = MinkPan(cfg)
    if w:
        w_weights = torch.load(w, map_location="cpu")
        # remove smlp head
        if "depthcontrast" in w or "pointcontrast" in w:
            del w_weights["model"]["head.clf.0.linear.weight"]
            del w_weights["model"]["head.clf.0.linear.bias"]
            del w_weights["model"]["head.clf.2.linear.weight"]
            del w_weights["model"]["head.clf.2.linear.bias"]

        model.backbone.load_state_dict(w_weights["model"], strict=True)

    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg.EXPERIMENT.ID, default_hp_metric=False
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    iou_ckpt = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_{epoch:02d}_{iou:.2f}",
        mode="max",
        save_last=True,
    )
    pq_ckpt = ModelCheckpoint(
        monitor="metrics/iou",
        filename=cfg.EXPERIMENT.ID + "_{epoch:02d}_{pq:.2f}",
        mode="max",
        save_last=True,
    )

    trainer = Trainer(
        # num_sanity_val_steps=0,
        gpus=cfg.TRAIN.N_GPUS,
        accelerator="ddp",
        logger=tb_logger,
        max_epochs=cfg.TRAIN.MAX_EPOCH,
        callbacks=[lr_monitor, iou_ckpt, pq_ckpt],
        # track_grad_norm=2,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=50,
        # overfit_batches=0.0001,
        accumulate_grad_batches=cfg.TRAIN.BATCH_ACC,
        resume_from_checkpoint=ckpt,
        limit_val_batches=0.01,
    )

    trainer.fit(model, data)


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


if __name__ == "__main__":
    main()
