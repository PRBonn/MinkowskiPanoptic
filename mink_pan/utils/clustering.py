import hdbscan
import numpy as np
import torch.nn as nn
from sklearn.cluster import MeanShift


class Clustering(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bandwidth = cfg.BANDWIDTH
        self.min_cluster_size = cfg.MIN_CLUSTER
        clustering = cfg.ALG
        if clustering == "MEANSHIFT":
            self.clustering = self.meanshift_cluster
        elif clustering == "HDBSCAN":
            self.clustering = self.hdbscan_cluster

    def set_ids(self, ids):
        self.things_ids = ids

    def forward(self, sem_preds, offsets, coors):
        last_id = 0
        pt_offsets = offsets.detach().cpu().numpy().reshape(-1, 3)
        valid = np.isin(sem_preds, self.things_ids).reshape(-1)
        ids = []
        clustered_ids = self.clustering(coors + pt_offsets, valid)
        thing_ind = np.where(clustered_ids != 0)
        clustered_ids[thing_ind] += last_id + 1
        last_id = max(clustered_ids)
        ids = clustered_ids

        return ids

    def meanshift_cluster(self, shifted_pts, valid):
        shift_dim = shifted_pts.shape[1]
        clustered_ins_ids = np.zeros(shifted_pts.shape[0], dtype=np.int32)
        valid_shifts = (
            shifted_pts[valid, :].reshape(-1, shift_dim)
            if valid is not None
            else shifted_pts
        )
        if valid_shifts.shape[0] == 0:
            return clustered_ins_ids

        ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True)
        try:
            ms.fit(valid_shifts)
        except Exception as e:
            ms = MeanShift(bandwidth=self.bandwidth)
            ms.fit(valid_shifts)
            print("\nException: {}.".format(e))
            print("Disable bin_seeding.")
        labels = ms.labels_ + 1
        assert np.min(labels) > 0
        if valid is not None:
            clustered_ins_ids[valid] = labels
            return clustered_ins_ids
        else:
            return labels

    def hdbscan_cluster(self, shifted_pcd, valid):
        clustered_ins_ids = np.zeros(shifted_pcd.shape[0], dtype=np.int32)
        valid_shifts = shifted_pcd[valid, :].reshape(-1, 3)
        if valid_shifts.shape[0] <= self.min_cluster_size:
            return clustered_ins_ids
        cluster = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size, allow_single_cluster=True
        ).fit(valid_shifts)
        instance_labels = cluster.labels_
        instance_labels += -instance_labels.min() + 1
        clustered_ins_ids[valid] = instance_labels
        return clustered_ins_ids
