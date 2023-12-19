import os
from os.path import join

import click
import torch
import yaml
from easydict import EasyDict as edict
from mink_pan.datasets.semantic_dataset import SemanticDatasetModule
from mink_pan.models.model import MinkPan
from pytorch_lightning import Trainer


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


@click.command()
@click.option("--w", type=str, default=None, required=True, help="weights to load")
@click.option("--save", is_flag=True, help="save ply predictions")
@click.option("--save_testset", is_flag=True, help="save predictions for test set")
@click.option("--nuscenes", is_flag=True)
@click.option("--mini", is_flag=True, help="use mini split for nuscenes")
@click.option(
    "--seq",
    type=int,
    default=None,
    required=False,
    help="use a single sequence for train and val",
)
def main(w, save, save_testset, nuscenes, mini, seq):
    model_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(join(getDir(__file__), "../config/backbone.yaml")))
    )
    cfg = edict({**model_cfg, **backbone_cfg})

    cfg.EVALUATE = True
    if save_testset:
        results_dir = create_dirs()
        print(f"Saving test set predictions in directory {results_dir}")
        cfg.RESULTS_DIR = results_dir
    if save:
        sem_dir = join(getDir(__file__), "..", "val_pred", "sem")
        ins_dir = join(getDir(__file__), "..", "val_pred", "ins")
        if not os.path.exists(sem_dir):
            os.makedirs(sem_dir, exist_ok=True)
        if not os.path.exists(ins_dir):
            os.makedirs(ins_dir, exist_ok=True)
        cfg.SAVE_VAL = True

    if nuscenes:
        cfg.MODEL.DATASET = "NUSCENES"
    if mini and nuscenes:
        cfg.NUSCENES.MINI = True
    if seq:
        cfg.TRAIN.ONLY_SEQ = seq

    data = SemanticDatasetModule(cfg)
    model = MinkPan(cfg)
    w = torch.load(w, map_location="cpu")
    model.load_state_dict(w["state_dict"])

    cfg.UPDATE_METRICS = "True"

    trainer = Trainer(gpus=cfg.TRAIN.N_GPUS, logger=False)

    print("Setup finished, start evaluation")

    if save_testset:
        trainer.test(model, data)
    else:
        trainer.validate(model, data)
    model.evaluator.print_results()


def create_dirs(nuscenes):
    if nuscenes:
        results_dir = join(getDir(__file__), "..", "output", "nuscenes_test")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = join(getDir(__file__), "..", "output", "test", "sequences")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        for i in range(11, 22):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), "predictions")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
    return results_dir


if __name__ == "__main__":
    main()
