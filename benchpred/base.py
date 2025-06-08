import torch
import random
import numpy as np
from abc import abstractmethod
from abc import ABC


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class BenchPred(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        # num_model = source_full_scores.shape[0]
        # num_data = source_full_scores.shape[1]
        #
        # assert num_model > 1
        # assert num_data > 1
        # assert coreset_size > 1
        # assert num_data > coreset_size
        #
        # set_random_seed(seed)
        pass

    @abstractmethod
    def get_coreset(self):
        pass

    @abstractmethod
    def predict(self, target_coreset_scores):
        # if len(target_coreset_scores.shape) == 1:
        #     target_coreset_scores = target_coreset_scores.reshape(1, -1)
        pass

    @abstractmethod
    def save(self, path_save):
        pass

    @abstractmethod
    def load(self, path_load):
        pass
