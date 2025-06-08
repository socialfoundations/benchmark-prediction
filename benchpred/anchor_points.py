"""
This code is adapted from the AnchorPoints repository:
https://github.com/rvivek3/AnchorPoints/blob/64b6087d11176cc707ebeabfd6a5b13f8a1cfaf2/optimal_valset_validation.py
"""

import kmedoids
import numpy as np
import joblib as jbl
from numpy.linalg import lstsq
from tqdm import tqdm

from .base import BenchPred, set_random_seed


class AnchorPointsWeightedPred(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.weights = None

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        # perform k-medoids over embeddings of our known points
        with np.errstate(divide="ignore", invalid="ignore"):
            corrs = np.corrcoef(source_full_scores, rowvar=False)
            corrs[np.isnan(corrs)] = 0.0

        selected_idxs = kmedoids.fasterpam(
            1 - corrs, coreset_size, init="random"
        ).medoids
        selected_idxs = list(set(selected_idxs))

        # get cluster sizes
        cluster_members = np.argmax(corrs[selected_idxs, :], axis=0)
        unique, cluster_sizes = np.unique(cluster_members, return_counts=True)
        cluster_sizes = list(cluster_sizes)

        # if clusters are empty, assign size of 0
        for i in range(coreset_size):
            if i not in unique:
                cluster_sizes.insert(i, 0)

        weights = cluster_sizes / np.sum(cluster_sizes)

        self.compressed_data_indices = selected_idxs
        self.weights = weights
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return (target_coreset_scores * self.weights).sum(1)

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.weights), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.weights = jbl.load(path_load)
        return self


class AnchorPointPredictorPred(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.M = None
        self.B = None
        self.nearest = None

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        with np.errstate(divide="ignore", invalid="ignore"):
            corrs = np.corrcoef(source_full_scores, rowvar=False)
            corrs[np.isnan(corrs)] = 0.0

        self.compressed_data_indices = kmedoids.fasterpam(
            1 - np.abs(corrs), coreset_size, init="random"
        ).medoids
        self.compressed_data_indices = list(set(self.compressed_data_indices))
        floaters = np.array(
            [i for i in range(num_data) if i not in self.compressed_data_indices]
        )

        coreset_size = len(self.compressed_data_indices)
        num_floaters = num_data - coreset_size
        # slopes (y = mx + b)
        self.M = np.zeros((num_floaters, coreset_size))
        # biases
        self.B = np.zeros((num_floaters, coreset_size))
        # indicator of which anchor point is nearest to any given point
        self.nearest = np.zeros((num_floaters, coreset_size))
        # residuals (error for each anchor on each point)
        resid = np.zeros((num_floaters, coreset_size))

        for j, anchor in enumerate(tqdm(self.compressed_data_indices)):
            apoints = source_full_scores[:, anchor]
            # add one for bias term
            A = np.vstack([apoints, np.ones(len(apoints))]).T
            for i, floater in enumerate(floaters):
                fpoints = source_full_scores[:, floater]

                # get y = mx + b terms and residual
                theta, residual = lstsq(A, fpoints, rcond=None)[:2]

                # set the params
                self.M[i, j] = theta[0]
                self.B[i, j] = theta[1]
                resid[i, j] = ((A @ theta - fpoints) ** 2).sum()

        mins = np.argmin(resid, axis=1)
        for i, mn in enumerate(mins):
            # weigh the nearest neighbor with smallest residuals 1, all others are zero
            self.nearest[i, mn] = 1

    def get_coreset(self):
        return self.compressed_data_indices

    def predict_single(self, anchors):
        anchor_preds = self.M * anchors[np.newaxis, :] + self.B
        weighted_anchor_preds = anchor_preds * self.nearest
        preds = np.sum(weighted_anchor_preds, axis=1)
        return preds

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)
        ret = []
        for anchor in target_coreset_scores:
            ret.append(self.predict_single(anchor).squeeze().mean())
        return np.array(ret)

    def save(self, path_save):
        jbl.dump(
            (self.compressed_data_indices, self.M, self.B, self.nearest), path_save
        )

    def load(self, path_load):
        self.compressed_data_indices, self.M, self.B, self.nearest = jbl.load(path_load)
        return self
