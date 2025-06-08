import os
import sys
import json
import logging
import platform
import shutil
import stat
from json import JSONDecodeError

import git
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from .general_utils import get_random_time_stamp, get_datetime
from .logger import logging_info

try:
    import wandb
except ModuleNotFoundError as err:
    logging.warning("WandB not installed!")
except TypeError as err:
    logging.warning("WandB not properly installed!")


class Recorder:
    def __init__(self, path_record, config=None, use_git=True, use_wandb=False):
        """
        Initialize the result recorder. The results will be saved in a temporary file defined by path_record.temp.
        To end recording and transfer the temporary files, self.end_recording() must be called.
        :param path_record: the saving path of the recorded results.
        :type path_record: str
        :param config: a record to be initialize with, usually the config in practice
        """
        self.__ending = False
        self.__record = dict()
        self.use_wandb = use_wandb

        self.path_temp_record = "%s.result.temp" % path_record
        self.path_record = "%s.result" % path_record

        if os.path.exists(self.path_temp_record):
            shutil.move(
                self.path_temp_record,
                self.path_temp_record + ".mv.%s" % get_random_time_stamp(),
            )
        if os.path.exists(self.path_record):
            shutil.move(
                self.path_record, self.path_record + ".mv.%s" % get_random_time_stamp()
            )

        if config is not None:
            for k in config.keys():
                self.__setitem__("config." + k, config[k])

        self.__setitem__("meta_data.operating_system", platform.system())
        self.__setitem__("meta_data.os_release", platform.release())
        self.__setitem__("meta_data.platform", platform.platform())
        self.__setitem__("meta_data.processor", platform.processor())
        self.__setitem__("meta_data.args", " ".join(sys.argv))
        self.__setitem__("meta_data.run_dir", os.getcwd())
        self.__setitem__("meta_data.start_time", get_datetime())

        if use_git:
            repo = git.Repo(path=os.getcwd())
            assert not repo.is_dirty()
            self.__setitem__("meta_data." + "git_commit", repo.head.object.hexsha)

        self.path_requirement = "%s.env.yml" % path_record
        if os.path.exists(self.path_requirement):
            shutil.move(
                self.path_requirement,
                self.path_requirement + ".mv.%s" % get_random_time_stamp(),
            )
        dir_conda = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
        )
        path_conda = os.path.join(dir_conda, "condabin", "conda")
        os.system("%s env export --file %s" % (path_conda, self.path_requirement))

    def write_record(self, line):
        """
        Add a line to the recorded result file.
        :param line: the content to be write
        :type line: str
        """
        with open(self.path_temp_record, "a", encoding="utf-8") as fin:
            fin.write(line + "\n")

    def keys(self):
        return self.__record.keys()

    def __getitem__(self, key):
        """
        Return the item based on the key.
        :param key:
        :type key:
        :return: results[key]
        """
        return self.__record[key]

    def __setitem__(self, key, value):
        """
        Set result[key] = value
        """
        assert not self.__ending
        assert key not in self.__record.keys()
        self.__record[key] = value
        self.write_record(json.dumps({key: value}))

    def add(self, key, value, epoch=None, step=None):
        if step is not None:
            key = "step_%d-%s" % (step, key)
        if epoch is not None:
            key = "epoch_%d-%s" % (epoch, key)
        if self.use_wandb:
            wandb.log({key: value})
        self.__setitem__(key, value)
        return key, value

    def add_with_logging(self, key, value, msg=None, epoch=None, step=None):
        """
        Add an item to results and also print with logging. The format of logging can be defined.
        :param key: the key
        :param value: the value to be added to the results
        :param msg: the message to the logger, format can be added. e.g. msg="Training set %s=%.4lf."
        :param epoch: current epoch
        :param step: current step
        """
        key, values = self.add(key, value, epoch, step)
        if msg is None:
            logging_info("%s: %s" % (key, str(value)))
        else:
            logging_info(msg % value)

    def update(self, new_record, epoch=None):
        """
        Update the results from new_record.
        """
        for k in new_record.keys():
            self.add(k, new_record[k], epoch)

    def end_recording(self):
        """
        End the recording. This function will remove the .temp suffix of the recording file and add an END signal.
        :return:
        :rtype:
        """
        assert "meta_data.end_time" not in self.keys()
        self.__setitem__("meta_data.end_time", get_datetime())
        self.__ending = True
        self.write_record("\n$END$\n")

        shutil.move(self.path_temp_record, self.path_record)
        os.chmod(self.path_record, stat.S_IREAD)

    def dump(self, path_dump):
        """
        Dump the result record in the path_dump.
        :param path_dump: the path to dump the result record
        :type path_dump: str
        """
        assert self.__ending
        path_dump = (
            "%s.result" % path_dump if not path_dump.endswith(".result") else path_dump
        )
        assert not os.path.exists(path_dump)
        shutil.copy(self.path_record, path_dump)

    def to_dict(self):
        """
        Return the results as a dict.
        :return: the results
        :rtype: dict
        """
        return self.__record

    def show(self):
        """
        To show the reuslts in logger.
        """
        logging_info(
            "\n%s"
            % json.dumps(
                self.__record, sort_keys=True, indent=4, separators=(",", ": ")
            )
        )


def load_result(path_record, return_type="dict"):
    """
    Load the result based on path_record.
    :param path_record: the path of the record
    :type path_record: str
    :param return_type: "dict" or "dataframe"
    :return: the result and whether the result record is ended
    :rtype: dict, bool
    """
    ret = dict()
    with open(path_record, "r", encoding="utf-8") as fin:
        ret["path"] = path_record
        for line in fin.readlines():
            if line.strip() == "$END$":
                return ret, True
            if len(line.strip().split()) == 0:
                continue
            ret.update(json.loads(line))
    if return_type == "dataframe":
        ret = pd.DataFrame(pd.Series(ret)).transpose()
    return ret, False


def collect_results(
    dir_results, collect_condition_func=None, pickled_filename=".pickled_results.jbl"
):
    """
    Collect all the ended results in dir_results.
    :param dir_results: the directory of the reuslts to be collected
    :type dir_results: str
    :param collect_condition_func: function to judge whether collect or not
    :param pickled_filename: filename of the pickled file
    :return: all ended result records
    :rtype: pd.DataFrame
    """
    assert os.path.exists(dir_results)
    path_pickled_results = os.path.join(dir_results, pickled_filename)
    if os.path.exists(path_pickled_results):
        data = joblib.load(path_pickled_results)
        already_collect_list = data["path"].values
    else:
        data = pd.DataFrame()
        already_collect_list = []

    updated = False
    to_be_read = []
    for path, dir_list, file_list in os.walk(dir_results):
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if not os.path.isdir(file_path) and file_path.endswith(".result"):
                if file_path not in already_collect_list:
                    if collect_condition_func is None or collect_condition_func(
                        file_path
                    ):
                        to_be_read.append(file_path)
    print("Got %d to be read." % len(to_be_read))

    new_data = list()
    for file_path in tqdm(to_be_read):
        try:
            result, ended = load_result(file_path)
            if ended:
                new_data.append(pd.DataFrame(pd.Series(result)).transpose())
                updated = True
        except JSONDecodeError:
            print("Collection Failed at %s" % file_path)
    print("Got %d new." % len(new_data))

    if updated:
        new_data = pd.concat(new_data, axis=0)
        data = pd.concat([data, new_data], axis=0)
        joblib.dump(data, path_pickled_results)
    return data.copy()


def collect_dead_results(dir_results):
    """
    Collect all un-ended results.
    :param dir_results: the directory of the reuslts to be collected
    :type dir_results: str
    :return: all un-ended result records.
    :rtype: pd.DataFrame
    """
    assert os.path.exists(dir_results)
    data = list()
    for path, dir_list, file_list in os.walk(dir_results):
        for file_name in file_list:
            path_file = os.path.join(path, file_name)
            if not os.path.isdir(path_file) and path_file.endswith(".result.temp"):
                result, ended = load_result(os.path.join(path_file))
                if not ended:
                    data.append(pd.DataFrame(pd.Series(result)).transpose())
    data = pd.concat(data, axis=0)
    return data


def get_max_epoch(data):
    data = data.dropna(axis=1, how="all")
    ret = -0x3F3F3F3F
    for c in data.columns:
        if c.startswith("epoch_"):
            try:
                epoch = int(c.split("-")[0].split("_")[1])
            except ValueError:
                continue
            ret = max(ret, epoch)
    return ret


def get_recorded_metrics(data):
    return set([c.split("-")[-1] for c in data.columns if c.startswith("epoch_0-")])


def get_trajectory(data, metric, filters=None, max_epoch=None):
    data_filtered = data[filters] if filters is not None else data
    assert len(data_filtered) == 1, "%d Files Located" % len(data_filtered)
    data_filtered = data_filtered.dropna(axis=1)
    max_epoch = get_max_epoch(data_filtered) if max_epoch is None else max_epoch

    x, y = [], []
    for c in data_filtered.columns:
        if c.startswith("epoch_") and c.endswith(metric):
            epoch = int(c.split("-")[0].split("_")[1])
            if epoch > max_epoch:
                continue
            v = data_filtered[c].values[0]
            if (type(v) in [str]) or (not np.isnan(v) and not np.isinf(v)):
                x.append(epoch)
                y.append(v)
            else:
                break

    assert len(x) == max_epoch, "%d != %d" % (len(x), max_epoch)
    order = sorted(list(range(len(x))), key=lambda i: x[i])
    x = [x[i] for i in order]
    y = [y[i] for i in order]
    return x, y


def fill_config_na(data, config_path, prefix="", suffix="", exclude_key=()):
    config = Config(default_config_file=config_path, use_argparse=False)
    for k in config.keys():
        if k not in exclude_key:
            try:
                data[prefix + k + suffix] = data[prefix + k + suffix].fillna(config[k])
            except TypeError:
                print("Unsupported Data Type: ", k)
    return data


def get_informative_columns(data, config_path):
    config = Config(default_config_file=config_path, use_argparse=False)
    config_keys = config.keys()
    columns_diff = []
    for c in config_keys:
        if c in data.columns:
            try:
                v = data[c].values
                if type(v[0]) is list:
                    v = [tuple(vi) for vi in v]
                if len(set(v)) != 1:
                    columns_diff.append(c)
            except TypeError:
                print("Unsupported Data or Contains NaN: ", c)
        else:
            print("Missed Key: ", c)
    return columns_diff


def get_columns_group_by(data, config_path, exclude_key=("exp_name", "random_seed")):
    ret = []
    config = Config(default_config_file=config_path)
    for k in config.keys():
        if (
            k not in exclude_key
            and len(set([(tuple(i) if type(i) == list else i) for i in data[k].values]))
            != 1
        ):
            ret.append(k)
    return ret


def remove_duplicate(data, keys=("phase", "exp_name")):
    data = data.drop_duplicates(subset=keys, keep="last")
    return data


def merge_phase(
    data, data_to_merge, merge_on_keys=("exp_name",), suffixes=("", "_eval")
):
    return data.merge(data_to_merge, how="inner", on=merge_on_keys, suffixes=suffixes)


def simple_read_results_pipeline(
    dir_results,
    collect_condition_func=None,
    pickled_filename=".pickled_results.jbl",
    column_filtering_func=lambda c: False,
    columns4show=None,
    columns4group=None,
    path_default_config=None,
):
    all_data = collect_results(dir_results, collect_condition_func, pickled_filename)
    columns = [c for c in all_data.columns if not column_filtering_func(c)]
    filtered_data = all_data[columns]

    if columns4show is None:
        return all_data, filtered_data

    if columns4group is None and path_default_config is not None:
        filtered_data = fill_config_na(filtered_data, path_default_config)
        columns_diff = get_informative_columns(filtered_data, path_default_config)
        columns_diff = [
            c for c in columns_diff if "exp_name" not in c and "random_seed" not in c
        ]
        columns4group = columns_diff

    assert columns4group is not None, "Group Keys Unprovided!"

    grouped_data_mean = pd.DataFrame(
        filtered_data[columns4group + columns4show].groupby(by=columns4group).mean()
    ).reset_index()
    grouped_data_std = pd.DataFrame(
        filtered_data[columns4group + columns4show].groupby(by=columns4group).std()
    ).reset_index()
    grouped_data_count = pd.DataFrame(
        filtered_data[columns4group + columns4show].groupby(by=columns4group).count()
    ).reset_index()

    grouped_data = merge_phase(
        grouped_data_mean,
        grouped_data_std,
        merge_on_keys=columns4group,
        suffixes=("", "_std"),
    )
    grouped_data = merge_phase(
        grouped_data,
        grouped_data_count,
        merge_on_keys=columns4group,
        suffixes=("", "_count"),
    )

    columns4show_sorted = list(
        np.concatenate([[c, c + "_std", c + "_count"] for c in columns4show], axis=0)
    )

    grouped_data = grouped_data[columns4group + columns4show_sorted]
    grouped_data = grouped_data.sort_values(by=columns4show)

    return all_data, filtered_data, grouped_data
