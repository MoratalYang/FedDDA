import os
import sys
import time
import yaml
import os.path as osp
from yacs.config import CfgNode as CN

from .tools import mkdir_if_missing
import copy
__all__ = ["Logger", "setup_logger"]


class Logger:
    """Write console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def setup_logger(cfg):
    output = cfg.OUTPUT_DIR
    if output is None:
        return
    output = write_cfg(cfg)

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = osp.join(output, "log.txt")

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    sys.stdout = Logger(fpath)
    return output

def write_cfg(cfg):
    directory_path = cfg.OUTPUT_DIR
    create_if_not_exists(directory_path)
    paragroup_dirs = os.listdir(directory_path)
    n_para = len(paragroup_dirs)
    final_check = False
    for para in paragroup_dirs:
        exist_para_cfg = True
        para_path = os.path.join(directory_path,para)
        cfg_path = para_path+'/cfg.yaml'
        # query_cfg = copy.deepcopy(cfg)
        f = open(cfg_path, 'r+')
        query_cfg = yaml.unsafe_load(f)
        # query_cfg.merge_from_file(cfg_path)
        for name, value1 in cfg.items():
            if isinstance(value1, CN):
                if name not in query_cfg or cfg_to_dict(query_cfg[name]) != cfg_to_dict(value1):
                    exist_para_cfg = False
        if exist_para_cfg == True:
            final_check = True
            break

    if not final_check:
        path = os.path.join(directory_path, 'para' + str(n_para + 1))
        k = 1
        while os.path.exists(path):
            path = os.path.join(directory_path, 'para' + str(n_para + k))
            k = k + 1
        create_if_not_exists(path)
        cfg_path = path + '/cfg.yaml'
        with open(cfg_path, 'w') as f:
            f.write(yaml.dump(cfg_to_dict(cfg)))
    else:
        path = para_path
    return path
def cfg_to_dict(cfg):
    d = {}
    for k, v in cfg.items():
        if isinstance(v, CN):
            d[k] = cfg_to_dict(v)
        else:
            d[k] = v
    return d

def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)

