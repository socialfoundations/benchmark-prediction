import logging
import sys

from .general_utils import get_random_time_stamp


def get_logger(path_log="%s.log" % get_random_time_stamp(), force_add_handler=False):
    """
    Set up the logger. Note that the setting will also impact the default logging logger, which means that simply
    using logging.info() will output the logs to both stdout and the filename_log.
    :param path_log: the filename of the log
    :param force_add_handler: if True, will clear logging.root.handlers
    :type path_log: str
    """
    ret_logger = logging.getLogger()
    ret_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s: %(message)s", datefmt="%Y-%m-%d-%H:%M:%S"
    )

    if force_add_handler:
        ret_logger.handlers = []

    if not ret_logger.handlers:
        if path_log is not None:
            path_log = (
                "%s.log" % path_log if not path_log.endswith(".log") else path_log
            )
            fh = logging.FileHandler(path_log)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            ret_logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        ret_logger.addHandler(ch)

    return ret_logger


def logging_info(*args):
    if logging.root.level > logging.getLevelName("INFO"):
        logging.root.setLevel(logging.INFO)
        for handler in logging.root.handlers:
            handler.setLevel(logging.INFO)
        logging.info(*args)
    else:
        logging.info(*args)
