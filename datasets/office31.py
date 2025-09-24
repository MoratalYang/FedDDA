import os

from utils.data_utils import prepare_data_domain_partition_train

class Office31():
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        train_set, test_set, global_test_set,classnames, lab2cname = prepare_data_domain_partition_train(cfg, root)

        self.data_test = global_test_set
        self.federated_train_x = train_set
        self.federated_test_x = test_set
        self.lab2cname = lab2cname
        self.classnames = classnames
        self.num_classes = len(classnames)



