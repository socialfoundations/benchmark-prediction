import os
import json
import argparse
import logging

import yaml

from .general_utils import get_random_time_stamp, makedir_if_not_exist
from .logger import logging_info

try:
    import wandb
except ModuleNotFoundError as err:
    logging.warning("WandB not installed!")
except TypeError as err:
    logging.warning("WandB not properly installed!")

dir_configs = os.path.join(os.getcwd(), "configs")


def smart_load(path_file):
    if path_file.endswith("json"):
        return json.load(open(path_file, "r", encoding="utf-8"))
    elif path_file.endswith("yaml") or path_file.endswith("yml"):
        return yaml.safe_load(open(path_file, "r", encoding="utf-8"))
    else:
        logging.warning("Un-identified file type. It will be processed as json by default.")
        return json.load(open(path_file, "r", encoding="utf-8"))


class NestedDict(dict):
    def __init__(self, *args, **kwargs):
        """
        Every element could be visited by either attribute or dict manner.

        Examples:
            >>> a = NestedDict()
            >>> a["b"]["c"] = 1
            >>> a.b.c
            1
            >>> a.b.d = 2
            >>> a["b"]["d"]
            2
        """
        super(NestedDict, self).__init__(*args, **kwargs)
        for k in dict.keys(self):
            if type(k) is dict:
                dict.__setitem__(self, k, NestedDict(k))

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, key):
        ret = self
        for k in key.split("."):
            ret = dict.__getitem__(ret, k)
        return ret

    def __setitem__(self, key, value):
        key_list = key.split(".")
        cur = self
        for i in range(len(key_list)):
            key = key_list[i]
            if i == len(key_list) - 1:
                dict.__setitem__(cur, key, value)
            else:
                if key in dict.keys(cur):
                    assert type(dict.__getitem__(cur, key)) is NestedDict
                else:
                    dict.__setitem__(cur, key, NestedDict())
                cur = dict.__getitem__(cur, key)

    def update(self, new_dict, prefix=None):
        for k in new_dict:
            key = ".".join([prefix, k]) if prefix is not None else k
            value = new_dict[k]
            if type(value) is dict or type(value) is NestedDict:
                self.update(value, prefix=key)
            else:
                self[key] = value

    def keys(self, cur=None, prefix=None):
        if cur is None:
            cur = self

        ret = []
        for k in dict.keys(cur):
            v = cur[k]
            new_prefix = ".".join([prefix, k]) if prefix is not None else k
            if type(v) is dict or type(v) is NestedDict:
                ret += self.keys(cur=v, prefix=new_prefix)
            else:
                ret.append(new_prefix)
        return ret

    def get(self, item, default_value=None):
        if item in self.keys():
            return self[item]
        return default_value

    def show(self):
        """
        Show all the configs in logging. If get_logger is used before, then the outputs will also be in the log file.
        """
        logging_info("\n%s" % json.dumps(self, sort_keys=True, indent=4, separators=(',', ': ')))

    def to_dict(self):
        """
        Return the config as a dict
        :return: config dict
        :rtype: dict
        """
        return self

    def dump(self, path_dump=None):
        """
        Dump the config in the path_dump.
        :param path_dump: the path to dump the config
        :type path_dump: str
        """
        if path_dump is None:
            makedir_if_not_exist(dir_configs)
            path_dump = os.path.join(dir_configs, "%s.json" % get_random_time_stamp())
        path_dump = "%s.json" % path_dump if not path_dump.endswith(".json") else path_dump
        assert not os.path.exists(path_dump)
        with open(path_dump, "w", encoding="utf-8") as fout:
            json.dump(self, fout)


class Config(NestedDict):
    def __init__(self, default_config_file=None, default_config_dict=None, use_argparse=True, use_wandb=False):
        """
        Initialize the config. Note that either default_config_dict or default_config_file in json format must be
        provided! The keys will be transferred to argument names, and the type will be automatically detected. The
        priority is ``the user specified parameter (if the use_argparse is True)'' > ``user specified config file (if
        the use_argparse is True)'' > ``default config dict'' > ``default config file''.

        Examples:
        default_config_dict = {"lr": 0.01, "optimizer": "sgd", "num_epoch": 30, "use_early_stop": False}
        Then the following corresponding arguments will be added in this function if use_argparse is True:
        parser.add_argument("--lr", type=float)
        parser.add_argument("--optimizer", type=str)
        parser.add_argument("--num_epoch", type=int)
        parser.add_argument("--use_early_stop", action="store_true", default=False)
        parser.add_argument("--no-use_early_stop", dest="use_early_stop", action="store_false")

        :param default_config_dict: the default config dict
        :type default_config_dict: dict
        :param default_config_file: the default config file path
        :type default_config_file: str
        :param use_argparse: whether use argparse to parse the config
        :type use_argparse: bool
        :param use_wandb: whether init wandb with parent directory as project name and exp_name as run name
        :type use_wandb: bool
        """
        super(Config, self).__init__()

        # load from default config file
        if default_config_dict is None and default_config_file is None:
            if os.path.exists(os.path.join(os.getcwd(), "default_config.json")):
                default_config_file = os.path.join(os.getcwd(), "default_config.json")
            else:
                logging.error("Either default_config_file or default_config_dict must be provided!")
                raise NotImplementedError

        if default_config_file is not None:
            self.update(smart_load(default_config_file))
        if default_config_dict is not None:
            self.update(default_config_dict)

        # transform the param terms into argparse
        if use_argparse:
            parser = argparse.ArgumentParser()
            parser.add_argument("--config_file", type=str, default=None)
            # add argument parser
            for name_param in self.keys():
                value_param = self[name_param]
                if type(value_param) is bool:
                    parser.add_argument("--%s" % name_param, action="store_true", default=None)
                    parser.add_argument("--no-%s" % name_param, dest="%s" % name_param,
                                        action="store_false", default=None)
                elif type(value_param) is list:
                    parser.add_argument("--%s" % name_param, type=type(value_param[0]), nargs="+", default=None)
                else:
                    parser.add_argument("--%s" % name_param, type=type(value_param), default=None)
            args = parser.parse_args()

            updated_parameters = dict()
            args_dict = vars(args)
            for k in vars(args):
                if k != "config_file" and args_dict[k] is not None:
                    updated_parameters[k] = args_dict[k]

            if args.config_file is not None:
                self.update(smart_load(args.config_file))

            self.update(updated_parameters)

        if use_wandb:
            wandb.login()
            wandb.init(
                project=os.path.split(os.getcwd())[-1],
                name=self["exp_name"],
                config=self.to_dict()
            )


def get_parser():
    """
    Get a simple parser for argparse, which already contains the config_file argument.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    return parser


def parser2config(parser):
    """
    Parse the arguments from parser into a config.
    """
    return args2config(parser.parse_args())


def args2config(args):
    """
    Parse the arguments from args into a config.
    """
    args_dict = vars(args)
    return Config(default_config_dict=args_dict, use_argparse=False)


def are_configs_same(config_a, config_b, ignored_keys=("load_epoch",)):
    """
    Judge whether two configs are the same.
    Args:
        config_a: the first config
        config_b: the second config
        ignored_keys: thes keys that will be ignored when comparing

    Returns: True if the two configs are the same, False otherwise.

    """
    config_a, config_b = config_a.to_dict(), config_b.to_dict()

    # make sure config A is always equal or longer than config B
    if len(config_a.keys()) < len(config_b.keys()):
        swap_var = config_a
        config_a = config_b
        config_b = swap_var

    if len(config_a.keys() - config_b.keys()) > 1:
        logging.error(
            "Different config numbers: %d (Existing) : %d (New)!" % (len(config_a.keys()), len(config_b.keys())))
        return False
    elif len(config_a.keys() - config_b.keys()) == 1 and (config_a.keys() - config_b.keys())[0] != "config_file":
        logging.error(
            "Different config numbers: %d (Existing) : %d (New)!" % (len(config_a.keys()), len(config_b.keys())))
        return False
    else:
        for i in config_a.keys() & config_b.keys():
            _ai = tuple(config_a[i]) if type(config_a[i]) == list else config_a[i]
            _bi = tuple(config_b[i]) if type(config_b[i]) == list else config_b[i]
            if _ai != _bi and i not in ignored_keys:
                logging.error("Mismatch in %s: %s (Existing) - %s (New)" % (str(i), str(config_a[i]), str(config_b[i])))
                return False

    return True
