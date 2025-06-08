import numpy as np
import joblib as jbl
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from .base import BenchPred, set_random_seed


class PCAPred(BenchPred):
    def __init__(self):
        super().__init__()
        self.max_iter = None
        self.tol = None
        self.n_comp = None
        self.compressed_data_indices = None
        self.source_full_scores = None

    def fit(
        self,
        source_full_scores,
        coreset_size,
        n_comp=None,
        max_iter=100,
        tol=1e-4,
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

        self.source_full_scores = source_full_scores
        self.max_iter = max_iter
        self.tol = tol
        self.compressed_data_indices = np.random.permutation(num_data)[
            :coreset_size
        ]

        if n_comp is not None:
            self.n_comp = n_comp
        else:
            order = np.random.permutation(num_model)
            # tr_models = order[:int(self.num_model * 0.75)]
            val_models = order[int(num_model * 0.75) :]

            assert num_data > 20
            assert num_model > 20

            masked_matrix = source_full_scores.copy()
            not_selected_indices = np.array(
                [i for i in range(num_data) if i not in self.compressed_data_indices]
            )
            masked_matrix[val_models][:, not_selected_indices] = np.nan

            best_n_comp, best_gap = None, 1e9
            for n_comp in [2, 5, 10, 20]:
                filled_matrix = self.pca_impute(
                    masked_matrix, n_comp, self.max_iter, self.tol
                )
                gap = np.fabs(
                    filled_matrix[val_models].mean(1)
                    - source_full_scores[val_models].mean(1)
                ).mean()
                if gap < best_gap:
                    best_gap = gap
                    best_n_comp = n_comp
            self.n_comp = best_n_comp
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        num_data = self.source_full_scores.shape[1]
        num_target_models = target_coreset_scores.shape[0]

        target_matrix = np.zeros((num_target_models, num_data))
        target_matrix[:, self.compressed_data_indices] = target_coreset_scores
        not_selected_indices = np.array(
            [i for i in range(num_data) if i not in self.compressed_data_indices]
        )
        target_matrix[:, not_selected_indices] = np.nan

        masked_matrix = np.concatenate(
            [self.source_full_scores.copy(), target_matrix], axis=0
        )
        return self.pca_impute(
            masked_matrix, self.n_comp, self.max_iter, self.tol
        ).mean(1)[-num_target_models:]

    def save(self, path_save):
        jbl.dump(
            (
                self.source_full_scores,
                self.compressed_data_indices,
                self.n_comp,
                self.max_iter,
                self.tol,
            ),
            path_save,
        )

    def load(self, path_load):
        (
            self.source_full_scores,
            self.compressed_data_indices,
            self.n_comp,
            self.max_iter,
            self.tol,
        ) = jbl.load(path_load)
        return self

    @staticmethod
    def pca_impute(matrix, n_comp, max_iter, tol):
        mask = np.isnan(matrix)
        filled_matrix = SimpleImputer(strategy="mean").fit_transform(matrix).copy()

        for _ in range(max_iter):
            # Apply PCA
            pca = PCA(n_components=n_comp)
            filled_matrix_low_dim = pca.fit_transform(filled_matrix)
            filled_matrix_reconstructed = pca.inverse_transform(filled_matrix_low_dim)

            # Check for convergence: Calculate the norm of the difference for the missing values
            diff = filled_matrix_reconstructed[mask] - filled_matrix[mask]
            if np.linalg.norm(diff) < tol:
                break

            # Update the missing values with the reconstructed matrix values
            filled_matrix[mask] = filled_matrix_reconstructed[mask]

        # eigenvalues = np.sqrt(pca.explained_variance_)
        return filled_matrix
