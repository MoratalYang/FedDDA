import pandas as pd
import os
import numpy as np
import yaml
from yacs.config import CfgNode as CN

path = 'output/'
dataset = 'Office31'
beta_value  = 'beta_0.0'

aim_args_dict = {
    # 'parti_num': 1,
}

aim_cfg_dict = {
    # 'DATASET': {
    #     'parti_num':10
    #     # 'beta': 0.5
    #     # 'backbone': "resnet18"
    # }
}


# PairFlip RandomNoise min_sum
def mean_metric(specific_path):
    acc_dict = {}
    idx_dict = {}
    experiment_index = 0
    for model in os.listdir(specific_path):
        # if model  in method_include_list:
            model_path = os.path.join(specific_path, model)
            if os.path.isdir(model_path):
                for para in os.listdir(model_path):
                    para_path = os.path.join(model_path, para)
                    # args_pargs_pathath = para_path + '/args.csv'
                    cfg_path = para_path + '/cfg.yaml'
                    is_same = select_cfg(cfg_path)
                    if is_same:
                        if len(os.listdir(para_path)) >= 3:
                            if os.path.isdir(para_path):
                                data = pd.read_table(para_path + '/' + 'acc.csv', sep=",")
                                data = data.loc[:, data.columns]
                                acc_value = data.values[:,1:]
                                times = str(len(acc_value))
                                if acc_value.shape[0] == 0:
                                    continue
                                if type(acc_value[0][0]) == str:
                                    pass
                                else:
                                    mean_acc_value = np.mean(acc_value, axis=0)
                                    mean_acc_value = mean_acc_value.tolist()
                                    mean_acc_value = [round(item, 2) for item in mean_acc_value]
                                    last_acc_vale = mean_acc_value[-5:]
                                    last_acc_vale = np.mean(last_acc_vale)
                                    idx_acc_vale = acc_value[:, 45:50]
                                    idx_acc_vale = np.mean(idx_acc_vale, axis=1)
                                    idx_acc_vale = [round(item, 2) for item in idx_acc_vale]
                                    max_acc_vale = np.max(acc_value, axis=1)
                                    max_acc_vale = [round(item, 2) for item in max_acc_vale]
                                    mean_acc_value.append(round(last_acc_vale, 2))
                                    acc_dict[experiment_index] = [model+times,para] + mean_acc_value 
                                    idx_dict[experiment_index] = [model+times,para] + max_acc_vale
                                experiment_index += 1
    return acc_dict, idx_dict


def all_metric(structure_path, scale_num):
    acc_dict = {}
    experiment_index = 0
    for model in os.listdir(structure_path):
        if model != '':
            # if model != '' and model in method_list:
            model_path = os.path.join(structure_path, model)
            if os.path.isdir(model_path):  # Check this path = path to folder
                for para in os.listdir(model_path):
                    para_path = os.path.join(model_path, para)
                    args_path = para_path + '/args.csv'
                    cfg_path = para_path + '/cfg.yaml'
                    is_same = select_cfg(args_path, cfg_path)
                    if is_same:
                        if len(os.listdir(para_path)) >= 3:
                            data = pd.read_table(para_path + '/' + 'acc.csv', sep=",")
                            data = data.loc[:, data.columns]
                            acc_value = data.values[:, 1:]
                            # parti_num = args_pd['parti_num'][0]
                            times = int(len(acc_value) / scale_num)
                            if times == 0:
                                times = 1
                            mean_acc_value = []
                            for i in range(scale_num):
                                domain_acc_value = acc_value[[scale_num * j + i for j in range(times)]]
                                domain_mean_acc_value = np.mean(domain_acc_value, axis=0)
                                last_mean_acc_value = domain_mean_acc_value[-50:]
                                # last_mean_acc_value = np.max(domain_mean_acc_value)
                                last_mean_acc_value = np.mean(last_mean_acc_value)
                                mean_acc_value.append(last_mean_acc_value)  # 添加accuracy
                            mean_acc_value = [round(item, 3) for item in mean_acc_value]
                            mean_acc_value.append(np.mean(mean_acc_value))
                            acc_dict[experiment_index] = [model + str(times), para] + mean_acc_value
                            experiment_index += 1
    return acc_dict, scale_num


def select_cfg(cfg_path):
    is_same = True
    now_cfg = CN()

    with open(cfg_path, encoding="utf-8") as f:
        result = f.read()
        now_dict = yaml.full_load(result)

    is_same = dict_eval(aim_cfg_dict, now_dict, is_same)
    for sub_k in aim_cfg_dict:
        try:
            now_sub_dict = now_dict[sub_k]
            aim_sub_dict = aim_cfg_dict[sub_k]
            for para_name in aim_sub_dict:
                if aim_sub_dict[para_name] != now_sub_dict[para_name]:
                    is_same = False
                    break
        except:
            pass

        if not is_same:
            break

    return is_same


def dict_eval(aim_cfg_dict, now_dict, is_same):
    for sub_k in aim_cfg_dict:
        now_sub = now_dict[sub_k]
        aim_sub = aim_cfg_dict[sub_k]
        if isinstance(now_sub, dict):
            for para_name in aim_sub:
                if isinstance(aim_sub[para_name], dict):
                    is_same = dict_eval(aim_sub[para_name], now_sub[para_name], is_same)
                    return is_same
                else:
                    if aim_sub[para_name] != now_sub[para_name]:
                        is_same = False
                        return is_same
        else:
            if now_sub != aim_sub:
                is_same = False
                return is_same
    return is_same


if __name__ == '__main__':
    print('**************************************************************')
    specific_path = os.path.join(path,dataset, str(beta_value))

    all_metric_dict = {}
    all_mean = []
    all_name = []
    print("Dataset: {}_{}".format(dataset, beta_value))

    mean_acc_dict, idx_dict = mean_metric(specific_path)
    mean_df = pd.DataFrame(mean_acc_dict)
    idx_df = pd.DataFrame(idx_dict)
    if mean_df.columns.size > 0:
        mean_df = mean_df.T
        column_mean_acc_list = ['method','para'] + ['E: ' + str(i) for i in range(50)] + ['MEAN']
        mean_df.columns = column_mean_acc_list
        print(mean_df)
        all_mean.append(mean_df['MEAN'])
    data_dict = {'method': mean_df['method'],'para': mean_df['para']}
    for i in range(len(all_mean)):
        data_dict[i] = all_mean[i]
    mean_all_mean = np.mean(np.array(all_mean), axis=0)
    data_dict['mean_all_mean'] = mean_all_mean
    print(pd.DataFrame(data_dict))

    print('**************************************************************')


