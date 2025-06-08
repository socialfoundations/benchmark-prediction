import torch
import numpy as np
import joblib as jbl

from .base import BenchPred, set_random_seed


class DoubleOptimizePred(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.w = None

    @staticmethod
    def sample(p, num_medoids, method="hard"):
        if method == "hard":
            indices = torch.argsort(p, descending=True)[:num_medoids]
            ret = torch.zeros_like(p)
            ret[indices] = 1
        elif method == "soft":
            ret = (torch.rand_like(p) < p).float()
        else:
            raise NotImplementedError

        return ret + p - p.detach()

    @staticmethod
    def get_simplex_projection_lambda(v, k):
        """
        Projects a vector v onto the set {x in R^n | sum(x) = k, 0 <= x_i <= 1}

        Parameters:
        v (torch.Tensor): The original vector to be projected.
        k (float): The desired sum after projection.

        Returns:
        torch.Tensor: The projected vector.
        """
        v = v.clone().float()
        n = v.numel()

        # Ensure that k is within the feasible range [0, n]
        k = min(max(k, 0), n)

        # Initialize lambda bounds
        lambda_low = (v - 1).min()
        lambda_high = v.max()
        epsilon = 1e-6  # Tolerance for convergence

        # Define the function phi(lambda)
        def phi(lambd):
            x = torch.clamp(v - lambd, min=0.0, max=1.0)
            return x.sum() - k

        # Bisection method to find lambda*
        max_iter = 1000  # Maximum number of iterations
        iter = 0
        while (lambda_high - lambda_low > epsilon) and (iter < max_iter):
            lambda_mid = (lambda_low + lambda_high) / 2.0
            value = phi(lambda_mid)

            if value > 0:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
            iter += 1

        # Compute the projected vector
        lambda_star = (lambda_low + lambda_high) / 2.0
        return lambda_star

    def fit(
        self,
        source_full_scores,
        coreset_size,
        seed=42,
        verbose=False,
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

        source_full_scores = torch.tensor(source_full_scores)
        p = torch.nn.Parameter(torch.ones(num_data, requires_grad=True))
        w = torch.nn.Parameter(torch.randn(num_data, requires_grad=True))
        with torch.no_grad():
            lambda_star = self.get_simplex_projection_lambda(
                p.data, coreset_size
            )
            p.data = torch.clamp(p.data - lambda_star, min=0.0, max=1.0)
        optimizer_w = torch.optim.Adam([w], lr=0.1, weight_decay=1e-4)
        optimizer_p = torch.optim.Adam([p], lr=0.001, weight_decay=1e-4)

        order = np.random.permutation(num_model)
        train_models = order[: int(num_model * 0.8)]
        valid_models = order[int(num_model * 0.8) :]

        best_loss, best_w, best_p = 0x3F3F3F3F, None, None
        no_improvement, max_no_improvement = 0, 100
        for step in range(1000):
            sampled_indices = self.sample(p, coreset_size, "hard")
            pred = (source_full_scores * sampled_indices * w).sum(
                1
            ) / coreset_size
            loss_tr = torch.nn.functional.mse_loss(
                pred[train_models], source_full_scores[train_models].mean(1)
            )
            loss_val = torch.nn.functional.mse_loss(
                pred[valid_models], source_full_scores[valid_models].mean(1)
            )
            if loss_val.item() < best_loss:
                best_loss = loss_val.item()
                best_w = w.detach().clone()
                best_p = p.detach().clone()
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > max_no_improvement:
                    if verbose:
                        print("Early stopping at step %d" % step)
                    break

            if step % 100 == 0 and verbose:
                print(
                    "Step=%d,Train Loss=%.4lf,Valid Loss=%.4lf"
                    % (step, loss_tr.item(), loss_val.item())
                )

            optimizer_w.zero_grad()
            optimizer_p.zero_grad()
            loss_tr.backward()
            optimizer_w.step()
            optimizer_p.step()

            with torch.no_grad():
                lambda_star = self.get_simplex_projection_lambda(
                    p.data, coreset_size
                )
                p.data = torch.clamp(p.data - lambda_star, min=0.0, max=1.0)

        sampled_indices = (
            self.sample(best_p, coreset_size, "hard").detach().cpu()
        )

        self.compressed_data_indices = torch.where(sampled_indices)[0].numpy()
        self.w = best_w.detach().cpu().numpy()[self.compressed_data_indices]
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return (target_coreset_scores * self.w).mean(1)

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.w), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.w = jbl.load(path_load)
        return self
