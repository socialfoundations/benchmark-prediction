import os
import joblib as jbl
import numpy as np
from multiprocessing import Pool, cpu_count

from zarth_utils.config import Config
from zarth_utils.recorder import Recorder
from zarth_utils.nn_utils import set_random_seed, get_all_paths
from zarth_utils.timer import Timer

from data_utils import get_scores
from benchpred import all_methods

config = Config(
    default_config_dict={
        "dataset_name": "commonsense",
        "dir_results": "./results",
        "exp_suffix": "",
        "coreset_size": 50,
        "methods": list(all_methods.keys()),
        "model_split_method": "random",
        "seed_start": 0,
        "num_run": 100,
        "multi_process": True,
        "use_git": True,
    },
    use_argparse=True,
)

full_scores = get_scores(config.dataset_name)
num_model, num_data = full_scores.shape


def process_run(seed):
    set_random_seed(seed)

    if config.model_split_method == "interpolation":
        num_target_models = int(0.25 * num_model)
        target_models = list(np.random.permutation(num_model)[:num_target_models])
        source_models = [i for i in range(num_model) if i not in target_models]
    elif config.model_split_method == "extrapolation":
        order = np.argsort(full_scores.mean(-1))
        source_models = order[: int(0.5 * num_model)]
        target_models = order[-int(0.3 * num_model) :]
    else:
        raise NotImplementedError

    real_acc = full_scores[target_models].mean(-1)

    for method_name in config.methods:
        print("Running %s" % method_name)
        dir_exp = os.path.join(
            config.dir_results,
            config.dataset_name + config.exp_suffix,
            method_name,
            str(seed),
        )
        if os.path.exists(os.path.join(dir_exp, "result.jbl")):
            continue
        all_paths = get_all_paths(dir_exp)
        config.dump(all_paths["path_config"])
        recorder = Recorder(
            all_paths["path_record"], config=config, use_git=config.use_git
        )

        timer = Timer()
        method = all_methods[method_name]()
        timer.start()
        method.fit(
            source_full_scores=full_scores[source_models],
            coreset_size=config.coreset_size,
            seed=seed,
        )
        recorder["training_time"] = timer.get_last_duration()
        method.save(all_paths["path_best_ckpt"])

        compressed_scores = full_scores[:, method.get_coreset()]
        timer.start()
        pred_acc = method.predict(compressed_scores[target_models])
        recorder["inference_time"] = timer.get_last_duration()

        recorder["gap"] = float(np.fabs(pred_acc - real_acc).mean())
        recorder.end_recording()

        jbl.dump(
            {
                "method_name": method_name,
                "seed": seed,
                "source_models": source_models,
                "target_models": target_models,
                "real_acc": real_acc,
                "est_acc": pred_acc,
            },
            os.path.join(dir_exp, "result.jbl"),
        )


if config.multi_process:
    with Pool(min(cpu_count(), config.num_run)) as pool:
        pool.map(
            process_run, range(config.seed_start, config.seed_start + config.num_run)
        )
else:
    for i in range(config.seed_start, config.seed_start + config.num_run):
        process_run(i)
