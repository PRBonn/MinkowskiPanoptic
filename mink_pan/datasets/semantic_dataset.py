import json
import os

import numpy as np
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SemanticDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.things_ids = []
        self.color_map = []
        self.label_names = []
        self.dataset = cfg.MODEL.DATASET
        self.mini = cfg[cfg.MODEL.DATASET].MINI
        self.min_volume_space = cfg[cfg.MODEL.DATASET].MIN_VOLUME_SPACE
        self.max_volume_space = cfg[cfg.MODEL.DATASET].MAX_VOLUME_SPACE

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if "ONLY_SEQ" in self.cfg.TRAIN.keys():
            only_seq = self.cfg.TRAIN.ONLY_SEQ
        else:
            only_seq = None
        val_split = "valid"

        if self.mini:
            val_split = "mini_" + val_split
            train_split = "mini_train"
        else:
            train_split = "train"

        train_set = SemanticDataset(
            self.cfg[self.cfg.MODEL.DATASET].PATH + "/sequences/",
            self.cfg[self.cfg.MODEL.DATASET].CONFIG,
            split=train_split,
            seq=only_seq,
            dataset=self.dataset,
            percent=self.cfg.TRAIN.PERCENT,
        )
        self.train_pan_set = PanopticDataset(
            dataset=train_set,
            split="train",
            space=self.cfg[self.cfg.MODEL.DATASET].SPACE,
            num_pts=self.cfg[self.cfg.MODEL.DATASET].SUB_NUM_POINTS,
            subsample=self.cfg.TRAIN.SUBSAMPLE,
            aug=self.cfg.TRAIN.AUG,
        )

        val_set = SemanticDataset(
            self.cfg[self.cfg.MODEL.DATASET].PATH + "/sequences/",
            self.cfg[self.cfg.MODEL.DATASET].CONFIG,
            split=val_split,
            seq=only_seq,
            dataset=self.dataset,
        )
        self.val_pan_set = PanopticDataset(
            dataset=val_set, split="valid", space=self.cfg[self.cfg.MODEL.DATASET].SPACE
        )

        test_set = SemanticDataset(
            self.cfg[self.cfg.MODEL.DATASET].PATH + "/sequences/",
            self.cfg[self.cfg.MODEL.DATASET].CONFIG,
            split=val_split,
            seq=only_seq,
            dataset=self.dataset,
        )
        self.test_pan_set = PanopticDataset(
            dataset=test_set, split="test", space=self.cfg[self.cfg.MODEL.DATASET].SPACE
        )

        self.things_ids = train_set.things_ids
        self.color_map = train_set.color_map
        self.label_names = train_set.label_names

    def train_dataloader(self):
        dataset = self.train_pan_set
        collate_fn = BatchCollation()
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        dataset = self.val_pan_set
        collate_fn = BatchCollation()
        self.valid_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)
        return self.valid_loader

    def test_dataloader(self):
        dataset = self.test_pan_set
        collate_fn = BatchCollation()
        self.test_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)
        return self.test_loader


class SemanticDataset(Dataset):
    def __init__(
        self, data_path, cfg_path, split="train", seq=None, dataset="KITTI", percent=1.0
    ):
        yaml_path = cfg_path
        with open(yaml_path, "r") as stream:
            semyaml = yaml.safe_load(stream)

        self.things = get_things(dataset)
        self.stuff = get_stuff(dataset)

        self.label_names = {**self.things, **self.stuff}
        self.things_ids = get_things_ids(dataset)

        self.color_map = semyaml["color_map_learning"]
        self.labels = semyaml["labels"]
        self.learning_map = semyaml["learning_map"]
        self.inv_learning_map = semyaml["learning_map_inv"]
        self.split = split
        split = semyaml["split"][self.split]

        if seq:
            split = [seq]

        self.im_idx = []
        pose_files = []
        calib_files = []
        token_files = []
        fill = 2 if dataset == "KITTI" else 4
        if percent == 1.0:
            for i_folder in split:
                self.im_idx += absoluteFilePaths(
                    "/".join([data_path, str(i_folder).zfill(fill), "velodyne"])
                )
                pose_files.append(
                    absoluteDirPath(
                        "/".join([data_path, str(i_folder).zfill(fill), "poses.txt"])
                    )
                )
                calib_files.append(
                    absoluteDirPath(
                        "/".join([data_path, str(i_folder).zfill(fill), "calib.txt"])
                    )
                )
                if dataset == "NUSCENES":
                    token_files.append(
                        absoluteDirPath(
                            "/".join(
                                [
                                    data_path,
                                    str(i_folder).zfill(fill),
                                    "lidar_tokens.txt",
                                ]
                            )
                        )
                    )

            self.im_idx.sort()
            self.poses = load_poses(pose_files, calib_files)
            self.tokens = load_tokens(token_files)
        else:
            percentiles = json.load(open("config/percentiles_split.json", "r"))
            self.poses = []
            for i_folder in split:
                pose_file = absoluteDirPath(
                    "/".join([data_path, str(i_folder).zfill(fill), "poses.txt"])
                )

                calib_file = absoluteDirPath(
                    "/".join([data_path, str(i_folder).zfill(fill), "calib.txt"])
                )

                seq_poses = load_poses([pose_file], [calib_file])
                scan_files = percentiles[str(percent)][str(i_folder).zfill(fill)][
                    "points"
                ]
                scan_files.sort()
                self.poses += [
                    seq_poses[int(scan.split("/")[-1][:-4])] for scan in scan_files
                ]
                self.im_idx += scan_files

            self.tokens = load_tokens(token_files)

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        fname = self.im_idx[index]
        pose = self.poses[index]
        points = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        xyz = points[:, :3]
        intensity = points[:, 3]
        if len(intensity.shape) == 2:
            intensity = np.squeeze(intensity)
        token = "0"
        if len(self.tokens) > 0:
            token = self.tokens[index]
        if self.split == "test":
            annotated_data = np.expand_dims(
                np.zeros_like(points[:, 0], dtype=int), axis=1
            )
            sem_labels = annotated_data
            ins_labels = annotated_data
        else:
            annotated_data = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
                dtype=np.int32,
            ).reshape((-1, 1))
            sem_labels = annotated_data & 0xFFFF  # delete high 16 digits binary
            ins_labels = annotated_data >> 16
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)

        return (xyz, sem_labels, ins_labels, intensity, fname, pose, token)


class PanopticDataset(Dataset):
    def __init__(self, dataset, split, space, num_pts=0, subsample=False, aug=False):
        self.dataset = dataset
        self.num_points = num_pts
        self.split = split
        self.aug = aug
        self.subsample = subsample
        self.th_ids = dataset.things_ids
        self.xlim = space[0]
        self.ylim = space[1]
        self.zlim = space[2]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        xyz, sem_labels, ins_labels, intensity, fname, pose, token = data
        foreground = np.isin(sem_labels, self.th_ids).reshape(-1)
        keep = np.argwhere(
            (self.xlim[0] < xyz[:, 0])
            & (xyz[:, 0] < self.xlim[1])
            & (self.ylim[0] < xyz[:, 1])
            & (xyz[:, 1] < self.ylim[1])
            & (self.zlim[0] < xyz[:, 2])
            & (xyz[:, 2] < self.zlim[1])
        )[:, 0]
        xyz = xyz[keep]
        sem_labels = sem_labels[keep]
        ins_labels = ins_labels[keep]
        intensity = intensity[keep]
        foreground = foreground[keep]

        feats = np.concatenate((xyz, np.expand_dims(intensity, axis=1)), axis=1)

        # Subsample
        if self.split == "train" and self.subsample and len(xyz) > self.num_points:
            idx = np.random.choice(np.arange(len(xyz)), self.num_points, replace=False)
            xyz = xyz[idx]
            sem_labels = sem_labels[idx]
            ins_labels = ins_labels[idx]
            feats = feats[idx]
            intensity = intensity[idx]
            foreground = foreground[idx]

        if self.split == "train" and self.aug:
            xyz = pcd_augmentations(xyz)

        offset = get_offsets(xyz, ins_labels, sem_labels, self.th_ids)

        return (
            xyz,
            feats,
            sem_labels,
            ins_labels,
            offset,
            foreground,
            fname,
            pose,
            token,
        )


class BatchCollation:
    def __init__(self):
        self.keys = [
            "pt_coord",
            "feats",
            "sem_label",
            "ins_label",
            "offset",
            "foreground",
            "fname",
            "pose",
            "token",
        ]

    def __call__(self, data):
        return {self.keys[i]: list(x) for i, x in enumerate(zip(*data))}


def pcd_augmentations(xyz):
    # rotation
    rotate_rad = np.deg2rad(np.random.random() * 360)
    c, s = np.cos(rotate_rad), np.sin(rotate_rad)
    j = np.matrix([[c, s], [-s, c]])
    xyz[:, :2] = np.dot(xyz[:, :2], j)

    # flip
    flip_type = np.random.choice(4, 1)
    if flip_type == 1:
        xyz[:, 0] = -xyz[:, 0]
    elif flip_type == 2:
        xyz[:, 1] = -xyz[:, 1]
    elif flip_type == 3:
        xyz[:, 0] = -xyz[:, 0]
        xyz[:, 1] = -xyz[:, 1]

    # scale
    noise_scale = np.random.uniform(0.95, 1.05)
    xyz[:, 0] = noise_scale * xyz[:, 0]
    xyz[:, 1] = noise_scale * xyz[:, 1]

    # transform
    trans_std = [0.1, 0.1, 0.1]
    noise_translate = np.array(
        [
            np.random.normal(0, trans_std[0], 1),
            np.random.normal(0, trans_std[1], 1),
            np.random.normal(0, trans_std[2], 1),
        ]
    ).T
    xyz[:, 0:3] += noise_translate

    return xyz


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def absoluteDirPath(directory):
    return os.path.abspath(directory)


def parse_calibration(filename):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose
    calib_file.close()
    return calib


def parse_poses(filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses


def load_poses(pose_files, calib_files):
    poses = []
    # go through every file and get all poses
    # add them to match im_idx
    for i in range(len(pose_files)):
        calib = parse_calibration(calib_files[i])
        seq_poses_f64 = parse_poses(pose_files[i], calib)
        seq_poses = [pose.astype(np.float32) for pose in seq_poses_f64]
        poses += seq_poses
    return poses


def load_tokens(token_files):
    if len(token_files) == 0:
        return []
    token_files.sort()
    tokens = []
    # go through every file and get all tokens
    for f in token_files:
        token_file = open(f)
        for line in token_file:
            token = line.strip()
            tokens.append(token)
        token_file.close()
    return tokens


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


def calc_xyz_middle(xyz):
    return np.array(
        [
            (np.max(xyz[:, 0]) + np.min(xyz[:, 0])) / 2.0,
            (np.max(xyz[:, 1]) + np.min(xyz[:, 1])) / 2.0,
            (np.max(xyz[:, 2]) + np.min(xyz[:, 2])) / 2.0,
        ],
        dtype=np.float32,
    )


def get_offsets(xyz, ins_labels, sem_labels, th_ids):
    offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
    things_ids, th_idx = np.unique(ins_labels[:, 0], return_index=True)
    keep_th = np.array(
        [i for i, idx in enumerate(th_idx) if sem_labels[idx] in th_ids], dtype=int
    )
    # remove instances with wrong sem class
    things_ids = things_ids[keep_th]
    th_idx = th_idx[keep_th]
    for ti in things_ids:
        idx = ins_labels[:, 0] == ti
        xyz_i = xyz[idx]
        if xyz_i.shape[0] <= 0:
            continue
        center = calc_xyz_middle(xyz_i)
        offsets[idx] = center - xyz_i
    return offsets


def get_things(dataset):
    if dataset == "KITTI":
        things = {
            1: "car",
            2: "bicycle",
            3: "motorcycle",
            4: "truck",
            5: "other-vehicle",
            6: "person",
            7: "bicyclist",
            8: "motorcyclist",
        }
    elif dataset == "NUSCENES":
        things = {
            2: "bycicle",
            3: "bus",
            4: "car",
            5: "construction-vehicle",
            6: "motorcycle",
            7: "pedestrian",
            9: "trailer",
            10: "truck",
        }
    return things


def get_stuff(dataset):
    if dataset == "KITTI":
        stuff = {
            9: "road",
            10: "parking",
            11: "sidewalk",
            12: "other-ground",
            13: "building",
            14: "fence",
            15: "vegetation",
            16: "trunk",
            17: "terrain",
            18: "pole",
            19: "traffic-sign",
        }
    elif dataset == "NUSCENES":
        stuff = {
            1: "barrier",
            8: "traffic_cone",
            11: "driveable_surface",
            12: "other_flat",
            13: "sidewalk",
            14: "terrain",
            15: "manmade",
            16: "vegetation",
        }
    return stuff


def get_things_ids(dataset):
    if dataset == "KITTI":
        return [1, 2, 3, 4, 5, 6, 7, 8]
    elif dataset == "NUSCENES":
        return [2, 3, 4, 5, 6, 7, 9, 10]
