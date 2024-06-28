import pandas as pd
import os
import torch
from scipy.stats import kendalltau as kt
from scipy.stats import rankdata as rd

def kendal_tau(gold, pred):
    rank_gold = rd(gold)
    rank_pred = rd(pred)
    return kt(rank_gold, rank_pred)

def getDistributionFromFilename(filename):
    if filename.__contains__('normal'):
        return 'normal'
    elif filename.__contains__('beta'):
        return 'beta'
    elif filename.__contains__('poisson'):
        return 'poisson'
    elif filename.__contains__('uniform'):
        return 'uniform'
    else:
        return 'unknown'

def getDistributionIdFromFilename(filename):
    distribution_id = os.path.splitext(filename)[1].replace(".", "")
    if distribution_id == "":
        distribution_id = 0
    return int(distribution_id)

def writeToFile(test_loss_dict, file_path, model_name):
    df = pd.DataFrame(test_loss_dict)
    df.index.name = "id"
    df[['beta-mse', 'beta-r-mse', 'beta-correlation', 'beta-p_value']] = df['beta'].apply(pd.Series)
    df[['norm-mse', 'norm-r-mse', 'norm-correlation', 'norm-p_value']] = df['normal'].apply(pd.Series)
    df[['poisson-mse', 'poisson-r-mse', 'poisson-correlation', 'poisson-p_value']] = df['poisson'].apply(pd.Series)
    df[['unif-mse', 'unif-r-mse', 'unif-correlation', 'unif-p_value']] = df['uniform'].apply(pd.Series)
    df = df.drop(columns=['beta', 'normal', 'poisson', 'uniform'])
    df.to_csv(file_path + model_name + ".csv")
    sorted_df = df.sort_index()
    print(sorted_df)
    sorted_df.to_csv(file_path + "sorted_" + model_name + ".csv")

    avg_std = pd.read_csv(file_path + "sorted_" + model_name + ".csv")
    avg = avg_std.mean()
    std = avg_std.std()
    avg_std = avg_std.append(avg, ignore_index=True)
    avg_std = avg_std.append(std, ignore_index=True)
    avg_std.to_csv(file_path + "avg_std_sorted_" + model_name + ".csv")
