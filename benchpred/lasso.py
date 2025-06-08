import joblib as jbl
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from .base import BenchPred, set_random_seed


class LassoPred(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None

    @staticmethod
    def train_lasso_with_specific_sparsity(X, y, num_nonzero_coefs, alphas=None):
        if alphas is None:
            alphas = np.logspace(-4, 1, 1000)  # Define a range of alpha values

        best_model = None
        best_coef_count_diff = float("inf")

        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X, y)
            nonzero_coefs = np.sum(lasso.coef_ != 0)

            coef_count_diff = abs(nonzero_coefs - num_nonzero_coefs)
            # print(f"Alpha: {alpha}, Non-zero Coefs: {nonzero_coefs}, Coef Count Diff: {coef_count_diff}")

            if (
                nonzero_coefs <= num_nonzero_coefs
                and coef_count_diff < best_coef_count_diff
            ):
                best_coef_count_diff = coef_count_diff
                best_model = lasso

            if coef_count_diff == 0:
                break

        return best_model

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

        self.rgs = self.train_lasso_with_specific_sparsity(
            source_full_scores, source_full_scores.mean(-1), coreset_size
        )
        self.compressed_data_indices = np.where(self.rgs.coef_ != 0)[0]

        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        x = np.zeros([len(target_coreset_scores), len(self.rgs.coef_)])
        x[:, self.compressed_data_indices] = target_coreset_scores

        return self.rgs.predict(x)

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.rgs), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.rgs = jbl.load(path_load)
