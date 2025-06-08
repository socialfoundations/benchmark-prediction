# How Benchmark Prediction with Fewer Data Misses the Mark

**BenchPred** is a Python package that provides a suite of tools to predict benchmark performances with fewer data. It provides all methods that we examined in our [paper](https://arxiv.org/).

Our results shows that, these methods are most proficient at interpolating scores among similar models.  However, except Random-Sampling and AIPW, all methods face significant difficulties when predicting target models that differ substantially from those they have encountered before.

We caution against the indiscriminate use of benchmark prediction techniques and recommend to use **AIPW** or, alternatively, **Random-Sampling** with a larger sampling budget.

## Installation

Install the package with the following command,

```bash
pip install -e .
```

or only install the requirements,

```bash
pip install -r requirement.txt
```

## Example Usage

```python
from benchpred import all_methods


method = all_methods["aipw"]()
method.fit(
    source_full_scores,  # (n_source_model * n_data) binary matrix
    coreset_size=50,  
    seed=42
)
coreset = method.get_coreset()
target_coreset_scores = ...  # (n_target_model * coreset_size) binary matrix based on coreset
pred_acc = method.predict(target_coreset_scores)
```

## Quick Start

```python
python main.py --dataset_name imagenet --coreset_size 50 --no-multi_process --methods aipw --num_run 1 --seed_start 0 --no-use_git
```

The supported methods are listed as follows,

- Random-Sampling: --methods random_sampling
- AIPW: --methods aipw
- Random-Sampling-Learn: --methods random_sampling_and_learn
- Random-Search-Learn: --methods random_search_and_learn
- PCA: --methods pca
- Lasso: --methods lasso
- Double-Optimize: --methods double_optimize
- Anchor-Points-Weighted: --methods anchor_points_weighted
- Anchor-Points-Predictor: --methods anchor_points_predictor
- PIRT: --methods pirt
- GPIRT: --methods gpirt
