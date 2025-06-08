"""
This code is adapted from the DeepCore library:
https://github.com/PatrickZH/DeepCore/tree/74ec709f5ccba3b1e921b06c6e312d57741de3f8/deepcore/methods
"""

import torch
import numpy as np
import joblib as jbl

from .base import BenchPred, set_random_seed


class KCenterGreedyPred(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None

    @staticmethod
    def metric(x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # dist.addmm_(1, -2, x, y.t())
        dist.addmm_(x, y.t(), beta=1, alpha=-2)  # Use keyword arguments here
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        matrix = source_full_scores.T.astype(np.float32)
        budget = coreset_size

        if type(matrix) is torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) is np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False)

        sample_num = matrix.shape[0]
        assert sample_num >= 1

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        with torch.no_grad():
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True

            num_of_already_selected = np.sum(select_result)

            # Initialize a (num_of_already_selected+budget-1)*sample_num matrix
            # storing distances of pool points from each clustering center.
            dis_matrix = -1 * torch.ones(
                [num_of_already_selected + budget - 1, sample_num], requires_grad=False
            )

            dis_matrix[:num_of_already_selected, ~select_result] = self.metric(
                matrix[select_result], matrix[~select_result]
            )

            mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

            for i in range(budget):
                p = torch.argmax(mins).item()
                select_result[p] = True

                if i == budget - 1:
                    break
                mins[p] = -1
                dis_matrix[num_of_already_selected + i, ~select_result] = self.metric(
                    matrix[[p]], matrix[~select_result]
                )
                mins = torch.min(mins, dis_matrix[num_of_already_selected + i])

        self.compressed_data_indices = np.where(select_result)[0]
        return self

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return target_coreset_scores.mean(1)

    def get_coreset(self):
        return self.compressed_data_indices

    def save(self, path_save):
        jbl.dump(self.compressed_data_indices, path_save)

    def load(self, path_load):
        self.compressed_data_indices = jbl.load(path_load)
        return self


class ContextualDiversityPred(KCenterGreedyPred):
    def __init__(self):
        super().__init__()

    @staticmethod
    def metric(a_output, b_output):
        with torch.no_grad():
            aa = a_output.view(a_output.shape[0], 1, a_output.shape[1]).repeat(
                1, b_output.shape[0], 1
            )
            bb = b_output.view(1, b_output.shape[0], b_output.shape[1]).repeat(
                a_output.shape[0], 1, 1
            )
            return torch.sum(
                0.5 * aa * torch.log(aa / bb) + 0.5 * bb * torch.log(bb / aa), dim=2
            )


class HerdingPred(KCenterGreedyPred):
    def __init__(self):
        super().__init__()

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        matrix = source_full_scores.T.astype(np.float32)
        budget = coreset_size

        if type(matrix) is torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) is np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False)
        sample_num = matrix.shape[0]

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = np.arange(sample_num)
        with torch.no_grad():
            mu = torch.mean(matrix, dim=0)
            select_result = np.zeros(sample_num, dtype=bool)

            for i in range(budget):
                dist = self.metric(
                    ((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(
                        1, -1
                    ),
                    matrix[~select_result],
                )
                p = torch.argmax(dist).item()
                p = indices[~select_result][p]
                select_result[p] = True

        self.compressed_data_indices = np.where(select_result)[0]
        return self
