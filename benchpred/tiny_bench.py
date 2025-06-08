"""
This code is adapted from the tinyBenchmarks repository:
https://github.com/felipemaiapolo/tinyBenchmarks/tree/9c7e20302301ad531bfdfd9a7288e6e916bf22e9/tutorials
"""

import numpy as np
import joblib as jbl

from typing import Optional
from ordered_set import OrderedSet
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer

from .base import BenchPred, set_random_seed
from .py_irt.dataset import Dataset
from .py_irt.training import IrtModelTrainer, IrtConfig


def sigmoid(z):
    """
    Compute the sigmoid function for the input z.

    Parameters:
    - z: A numeric value or numpy array.

    Returns:
    - The sigmoid of z.
    """

    return 1 / (1 + np.exp(-z))


def item_curve(theta, a, b):
    """
    Compute the item response curve for given parameters.

    Parameters:
    - theta: The ability parameter of the subject.
    - a: The discrimination parameter of the item.
    - b: The difficulty parameter of the item.

    Returns:
    - The probability of a correct response given the item parameters and subject ability.
    """
    z = np.clip(a * theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)


def estimate_ability_parameters(
    responses_test, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"
):
    """
    Estimates the ability parameters for a new set of test responses.

    Parameters:
    - responses_test: A 1D array of the test subject's responses.
    - A: The discrimination parameters of the IRT model.
    - B: The difficulty parameters of the IRT model.
    - theta_init: Initial guess for the ability parameters.
    - eps: A small value to avoid division by zero and log of zero errors.
    - optimizer: The optimization method to use.
    - weights: weighting for items according to their representativeness of the whole scenario

    Returns:
    - optimal_theta: The estimated ability parameters for the test subject.
    """

    D = A.shape[1]

    # Define the negative log likelihood function
    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A, B).squeeze()
        log_likelihood = np.sum(
            responses_test * np.log(P + eps)
            + (1 - responses_test) * np.log(1 - P + eps)
        )
        return -log_likelihood

    # Ensure the initial theta is a numpy array with the correct shape
    if type(theta_init) == np.ndarray:
        theta_init = theta_init.reshape(-1)
        assert theta_init.shape[0] == D
    else:
        theta_init = np.zeros(D)

    # Use the minimize function to find the ability parameters that minimize the negative log likelihood
    optimal_theta = minimize(neg_log_like, theta_init, method=optimizer).x[
        None, :, None
    ]

    return optimal_theta


class NewDataset(Dataset):
    @classmethod
    def from_list(cls, data_list, train_items: dict = None, amortized: bool = False):
        """Parse IRT dataset from jsonlines, formatted in the following way:
        * The dataset is in jsonlines format, each line representing the responses of a subject
        * Each row looks like this:
        {"subject_id": "<subject_id>", "responses": {"<item_id>": <response>}}
        * Where <subject_id> is a string, <item_id> is a string, and <response> is a number (usually integer)
        """
        item_ids = OrderedSet()
        subject_ids = OrderedSet()
        item_id_to_ix = {}
        ix_to_item_id = {}
        subject_id_to_ix = {}
        ix_to_subject_id = {}

        input_data = data_list
        for line in input_data:
            subject_id = line["subject_id"]
            subject_ids.add(subject_id)
            responses = line["responses"]
            for item_id in responses.keys():
                item_ids.add(item_id)

        for idx, item_id in enumerate(item_ids):
            item_id_to_ix[item_id] = idx
            ix_to_item_id[idx] = item_id

        for idx, subject_id in enumerate(subject_ids):
            subject_id_to_ix[subject_id] = idx
            ix_to_subject_id[idx] = subject_id

        if amortized:
            vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words="english")
            vectorizer.fit(item_ids)

        observation_subjects = []
        observation_items = []
        observations = []
        training_example = []
        for idx, line in enumerate(input_data):
            subject_id = line["subject_id"]
            for item_id, response in line["responses"].items():
                observations.append(response)
                observation_subjects.append(subject_id_to_ix[subject_id])
                if not amortized:
                    observation_items.append(item_id_to_ix[item_id])
                else:
                    observation_items.append(
                        vectorizer.transform([item_id]).todense().tolist()[0]
                    )
                if train_items is not None:
                    training_example.append(train_items[subject_id][item_id])
                else:
                    training_example.append(True)

        return cls(
            item_ids=item_ids,
            subject_ids=subject_ids,
            item_id_to_ix=item_id_to_ix,
            ix_to_item_id=ix_to_item_id,
            subject_id_to_ix=subject_id_to_ix,
            ix_to_subject_id=ix_to_subject_id,
            observation_subjects=observation_subjects,
            observation_items=observation_items,
            observations=observations,
            training_example=training_example,
        )


def create_irt_dataset(responses):
    """
    Creates a dataset suitable for IRT analysis from a given set of responses and saves it in a JSON lines format.

    Parameters:
    - responses: A numpy array where each row represents a subject and each column a question.
    - dataset_name: The name of the file where the dataset will be saved.
    """

    dataset = []
    for i in range(responses.shape[0]):
        aux = {}
        aux_q = {}

        # Iterate over each question to create a response dict
        for j in range(responses.shape[1]):
            aux_q["q" + str(j)] = int(responses[i, j])
        aux["subject_id"] = str(i)
        aux["responses"] = aux_q
        dataset.append(aux)

    return NewDataset.from_list(dataset)


class PIRTPred(BenchPred):
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.anchor_points = None
        self.coreset_size = None
        self.num_data = None
        self.anchor_weights = None

    def fit(
        self,
        source_full_scores,
        coreset_size,
        seed=42,
        model_type: str = "multidim_2pl",
        dims: Optional[int] = 10,
        epochs: Optional[int] = 2000,
        priors: Optional[str] = "hierarchical",
        lr: Optional[float] = 0.1,
        deterministic: bool = True,
        device: str = "cpu",
        dropout: Optional[float] = 0.5,
        hidden: Optional[int] = 100,
        log_every: int = 200,
        *args,
        **kwargs,
    ):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        config = IrtConfig(
            **{
                "priors": priors,
                "dims": dims,
                "lr": lr,
                "epochs": epochs,
                "model_type": model_type,
                "dropout": dropout,
                "hidden": hidden,
                "log_every": log_every,
                "deterministic": deterministic,
                "seed": seed,
            }
        )

        # Get IRT parameters
        trainer = IrtModelTrainer(
            config=config, dataset=create_irt_dataset(source_full_scores)
        )
        trainer.train(device=device)
        params = trainer.best_params
        self.A = np.array(params["disc"]).T[None, :, :]
        self.B = np.array(params["diff"]).T[None, :, :]

        # Get anchor points
        X = np.vstack((self.A.squeeze(), self.B.squeeze().reshape((1, -1)))).T
        kmeans = KMeans(
            n_clusters=coreset_size, n_init="auto", random_state=seed
        )
        kmeans.fit(X)

        self.anchor_points = pairwise_distances(
            kmeans.cluster_centers_, X, metric="euclidean"
        ).argmin(axis=1)
        self.anchor_weights = (
            np.array([np.sum(kmeans.labels_ == c) for c in range(coreset_size)])
            / num_data
        )
        self.coreset_size = coreset_size
        self.num_data = num_data
        return self

    def get_coreset(self):
        return self.anchor_points

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)
        thetas = [
            estimate_ability_parameters(
                target_coreset_scores[j],
                self.A[:, :, self.anchor_points],
                self.B[:, :, self.anchor_points],
            )
            for j in range(target_coreset_scores.shape[0])
        ]

        pirt_lambd = self.coreset_size / self.num_data
        ind_unseen = [i for i in range(self.num_data) if i not in self.anchor_points]
        pirt_pred = []
        for j in range(target_coreset_scores.shape[0]):
            data_part = target_coreset_scores[j].mean()
            irt_part = item_curve(thetas[j], self.A, self.B)[0, ind_unseen].mean()
            pirt_pred.append(pirt_lambd * data_part + (1 - pirt_lambd) * irt_part)
        pirt_pred = np.array(pirt_pred)

        return pirt_pred

    def save(self, path_save):
        jbl.dump(
            (
                self.A,
                self.B,
                self.anchor_points,
                self.coreset_size,
                self.num_data,
            ),
            path_save,
        )

    def load(self, path_load):
        self.A, self.B, self.anchor_points, self.coreset_size, self.num_data = (
            jbl.load(path_load)
        )
        return self


class GPIRTPred(PIRTPred):
    def __init__(self):
        super().__init__()
        self.lambd = None

    def fit(
        self,
        source_full_scores,
        coreset_size,
        seed=42,
        model_type: str = "multidim_2pl",
        dims: Optional[int] = 10,
        epochs: Optional[int] = 2000,
        priors: Optional[str] = "hierarchical",
        lr: Optional[float] = 0.1,
        deterministic: bool = True,
        device: str = "cpu",
        dropout: Optional[float] = 0.5,
        hidden: Optional[int] = 100,
        log_every: int = 200,
        *args,
        **kwargs,
    ):
        super().fit(
            source_full_scores,
            coreset_size,
            seed,
            model_type,
            dims,
            epochs,
            priors,
            lr,
            deterministic,
            device,
            dropout,
            hidden,
            log_every,
            *args,
            **kwargs,
        )
        config = IrtConfig(
            **{
                "priors": priors,
                "dims": dims,
                "lr": lr,
                "epochs": epochs,
                "model_type": model_type,
                "dropout": dropout,
                "hidden": hidden,
                "log_every": log_every,
                "deterministic": deterministic,
                "seed": seed,
            }
        )
        super().fit(
            source_full_scores,
            coreset_size,
            seed,
            model_type,
            dims,
            epochs,
            priors,
            lr,
            deterministic,
            device,
            dropout,
            hidden,
            log_every,
            *args,
            **kwargs,
        )

        val_ind = list(range(0, source_full_scores.shape[0], 5))  # Validation indices
        train_ind = [i for i in range(source_full_scores.shape[0]) if i not in val_ind]

        trainer = IrtModelTrainer(
            config=config, dataset=create_irt_dataset(source_full_scores[train_ind])
        )
        trainer.train(device=device)
        params = trainer.best_params
        A = np.array(params["disc"]).T[None, :, :]
        B = np.array(params["diff"]).T[None, :, :]

        seen_items = list(range(0, source_full_scores.shape[1], 2))
        unseen_items = list(range(1, source_full_scores.shape[1], 2))

        thetas = [
            estimate_ability_parameters(
                source_full_scores[val_ind][j][seen_items],
                A[:, :, seen_items],
                B[:, :, seen_items],
            )
            for j in range(len(val_ind))
        ]

        errors = []
        for i, val_model_i in enumerate(val_ind):
            errors.append(
                abs(
                    item_curve(thetas[i], A, B).squeeze()[unseen_items].mean()
                    - source_full_scores[val_model_i, unseen_items].mean()
                )
            )

        v = np.var(source_full_scores, axis=1).mean() / (4 * coreset_size)
        b = np.mean(errors)
        self.lambd = (b**2) / (v + (b**2))

        return self

    def predict(self, target_coreset_scores):
        anchor_pred = (target_coreset_scores * self.anchor_weights).sum(1)
        return (1 - self.lambd) * super().predict(
            target_coreset_scores
        ) + self.lambd * anchor_pred

    def save(self, path_save):
        jbl.dump(
            (
                self.A,
                self.B,
                self.anchor_points,
                self.anchor_weights,
                self.coreset_size,
                self.num_data,
                self.lambd,
            ),
            path_save,
        )

    def load(self, path_load):
        (
            self.A,
            self.B,
            self.anchor_points,
            self.anchor_weights,
            self.coreset_size,
            self.num_data,
            self.lambd,
        ) = jbl.load(path_load)
        return self
