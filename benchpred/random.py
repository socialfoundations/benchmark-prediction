import numpy as np
import joblib as jbl
from sklearn.linear_model import RidgeCV
from .base import BenchPred, set_random_seed


class RandomSampling(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        self.compressed_data_indices = np.random.permutation(num_data)[
            :coreset_size
        ]
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return target_coreset_scores.mean(1)

    def save(self, path_save):
        jbl.dump(self.compressed_data_indices, path_save)

    def load(self, path_load):
        self.compressed_data_indices = jbl.load(path_load)
        return self


class RandomSamplingAndLearn(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        self.compressed_data_indices = np.random.permutation(num_data)[
            :coreset_size
        ]

        self.rgs = RidgeCV()
        self.rgs.fit(
            source_full_scores[:, self.compressed_data_indices],
            source_full_scores.mean(1),
        )
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return self.rgs.predict(target_coreset_scores)

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.rgs), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.rgs = jbl.load(path_load)
        return self


class RandomSearchAndLearn(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None

    def fit(
        self,
        source_full_scores,
        coreset_size,
        num_search=10000,
        seed=42,
        *args,
        **kwargs
    ):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        order = np.random.permutation(num_model)
        tr_models = order[: int(num_model * 0.75)]
        val_models = order[int(num_model * 0.75) :]

        best_idxs, best_gap = None, 1e9
        for _ in range(num_search):
            selected_idxs = np.random.permutation(num_data)[:coreset_size]
            rgs = RidgeCV()
            rgs.fit(
                source_full_scores[tr_models][:, selected_idxs],
                source_full_scores.mean(1)[tr_models],
            )
            estimated_scores = rgs.predict(source_full_scores[:, selected_idxs])
            gap = np.fabs(
                estimated_scores[val_models] - source_full_scores.mean(1)[val_models]
            ).mean()
            if gap < best_gap:
                best_gap = gap
                best_idxs = selected_idxs

        self.compressed_data_indices = best_idxs
        self.rgs = RidgeCV()
        self.rgs.fit(
            source_full_scores[:, self.compressed_data_indices],
            source_full_scores.mean(1),
        )
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return self.rgs.predict(target_coreset_scores)

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.rgs), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.rgs = jbl.load(path_load)
        return self
