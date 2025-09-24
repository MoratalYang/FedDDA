import torch
import copy
from prettytable import PrettyTable
import numpy as np
from torch.nn import functional as F

def show_results(cfg, results, epoch,global_test_acc_dict):

    global_test_acc = []
    global_test_error = []
    global_test_f1 = []
    for k, result in enumerate(results):
        global_test_acc.append(results[k]['accuracy'])
        global_test_error.append(results[k]['error_rate'])
        global_test_f1.append(results[k]['macro_f1'])

        if k in global_test_acc_dict:
            global_test_acc_dict[k].append(results[k]['accuracy'])
        else:
            global_test_acc_dict[k] = [results[k]['accuracy']]

        print(k, "--Local test acc:", results[k]['accuracy'])

    print("--Global test acc:", sum(global_test_acc) / len(global_test_acc))

    print(f"Epoch:{epoch}")
    return global_test_acc,global_test_acc_dict


def count_parameters(model, model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

import os

def save_acc_csv(para_dir,global_test_acc_dict,cfg):
    acc_path = os.path.join(para_dir, 'acc.csv')
    if os.path.exists(acc_path):
        with open(acc_path, 'a') as result_file:
            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(str(key) + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
    else:
        with open(acc_path, 'w') as result_file:
            result_file.write('idx,')
            for epoch in range(cfg.OPTIM.ROUND):
                result_file.write('epoch_' + str(epoch))
                if epoch != cfg.OPTIM.ROUND - 1:
                    result_file.write(',')
                else:
                    result_file.write('\n')

            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(str(key) + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')


def average_weights(w, idxs_users, datanumber_client, islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg