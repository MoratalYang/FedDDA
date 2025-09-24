from collections import defaultdict

from utils.fed_utils import count_parameters, show_results, save_acc_csv, average_weights
from Dassl.dassl.utils import setup_logger, set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import setproctitle
import numpy as np
import argparse
import torch
import time
import copy


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    # Config for FEDDDA
    cfg.TRAINER.FEDDDA = CN()
    cfg.TRAINER.FEDDDA.N_CTX = 16  # number of context vectors
    cfg.TRAINER.FEDDDA.CSC = False  # class-specific context
    cfg.TRAINER.FEDDDA.CTX_INIT = False  # initialization words
    cfg.TRAINER.FEDDDA.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.FEDDDA.CLASS_TOKEN_POSITION = "middle"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.FEDDDA.N = args.num_prompt  # number of prompts
    cfg.TRAINER.FEDDDA.AVG_N = args.num_prompt / 2  # number of prompts to aggregate
    cfg.TRAINER.FEDDDA.TOP_PERCENT = 1
    cfg.TRAINER.FEDDDA.MAX_ITER = 100 

    current_trainer_cfg = cfg['TRAINER'][args.trainer]
    cfg['TRAINER'] = CN()
    cfg['TRAINER'][args.trainer] = current_trainer_cfg

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients

    # cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition

    cfg.DATASET.USEALL = args.useall  # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots

    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0  # repeat rate on each client

    cfg.OPTIM.ROUND = 1  # global round
    cfg.OPTIM.GAMMA = args.gamma  # gamma of single-step

    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()

    extend_cfg(cfg, args)

    if args.dataset:
        cfg.merge_from_file(f'configs/datasets/{args.dataset}.yaml')

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    reset_cfg(cfg, args)

    cfg.OUTPUT_DIR = f"output/{args.dataset}/beta_{args.beta}/{args.trainer}"

    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    args.para_dir = setup_logger(cfg)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    local_weights_0 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_1 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_2 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_3 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_4 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_5 = [[] for i in range(cfg.DATASET.USERS)]
    local_weights_6 = [[] for i in range(cfg.DATASET.USERS)]

    local_weights_per = [{} for i in range(cfg.DATASET.USERS)]

    local_trainer = build_trainer(args, cfg)
    local_trainer.fed_before_train()
    count_parameters(local_trainer.model, "prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")

    datanumber_client = []
    if args.trainer == 'CLIP':
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict[net_i].dataset))

        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    # Training
    start_epoch = 0
    end_epoch = cfg.OPTIM.ROUND
    global_test_acc_dict = {}
    global_time_list = []
    start = time.time()

    for epoch in range(start_epoch, end_epoch):

        if args.trainer == 'FEDDDA':
            idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)

            print("------------local train start epoch:", epoch, "-------------")

            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights_0[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][:args.avg_prompt])
                local_weights_1[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'][args.avg_prompt:args.num_prompt])
                local_weights_2[idx] = copy.deepcopy(local_weight['shared_adapter.layer.0.weight'])
                local_weights_3[idx] = copy.deepcopy(local_weight['shared_adapter.layer.2.weight'])
                local_weights_4[idx] = copy.deepcopy(local_weight['specific_adapter.layer.0.weight'])
                local_weights_5[idx] = copy.deepcopy(local_weight['specific_adapter.layer.2.weight'])
                local_weights_6[idx] = copy.deepcopy(local_weight['gate.wg.weight'])

            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights_0, idxs_users, datanumber_client, islist=True)
            adapter_weight_0 = average_weights(local_weights_2, idxs_users, datanumber_client, islist=True)
            adapter_weight_1 = average_weights(local_weights_3, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))

            for idx in all_users:
                local_weights_per[idx]['prompt_learner.ctx'] = torch.cat([global_weights, local_weights_1[idx]], dim=0)
                local_weights_per[idx]['shared_adapter.layer.0.weight'] = adapter_weight_0
                local_weights_per[idx]['shared_adapter.layer.2.weight'] = adapter_weight_1
                local_weights_per[idx]['specific_adapter.layer.0.weight'] = local_weights_4[idx]
                local_weights_per[idx]['specific_adapter.layer.2.weight'] = local_weights_5[idx]
                local_weights_per[idx]['gate.wg.weight'] = local_weights_6[idx]

            for idx in all_users:
                local_trainer.model.load_state_dict(local_weights_per[idx], strict=False)
                results.append(local_trainer.test(idx=idx))

            global_test_acc, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
            global_time_list.append(time.time() - start)
            print("------------local test finish-------------")

    for idx in idxs_users:
        local_trainer.fed_after_train()
    for key, global_test_acc_list in global_test_acc_dict.items():
        print(key, "global_test_acc_list:", global_test_acc_list)
        print(key, "maximum test acc:", max(global_test_acc_list))
        print(key, "mean of acc:", np.mean(global_test_acc_list[-5:]))
        print(key, "std of acc:", np.std(global_test_acc_list[-5:]))
    save_acc_csv(local_trainer.args.para_dir, global_test_acc_dict, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, default="FEDDDA", help="name of trainer, choose from: FEDDDA")
    parser.add_argument("--dataset", type=str, default="pacs", help="name of dataset, choose from: "
                                                                    " domainnet pacs OfficeHome  Office31 ")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--beta', type=float, default=0.0, help='The parameter for the dirichlet distribution')

    parser.add_argument('--num_users', type=int, default=4, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=128, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    parser.add_argument('--num_shots', type=int, default=2, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")

    parser.add_argument('--partition', type=str, default='noniid-labeldir',
                        help='the data partitioning strategy of cifar10 and cifar100,'
                             ' select from "noniid-labeluni, noniid-labeldir,noniid-labeldir100"')

    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")
    parser.add_argument('--num_prompt', type=int, default=2, help="number of prompts")
    parser.add_argument('--avg_prompt', type=int, default=1, help="half number of prompts")

    # parameters of path
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--root", type=str, default="/data0/yyh_data/.", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="output/..", help="output directory")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    # parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    args = parser.parse_args()

    setproctitle.setproctitle('{}_{}_{}'.format(args.trainer, args.backbone, args.dataset))

    main(args)
