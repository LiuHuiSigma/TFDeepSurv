# coding=utf-8
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys
import time
import json
import pandas as pd
import numpy as np
import hyperopt as hpt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

import dataset
import L2DeepSurv

global Logval, eval_cnt, time_start
global train_X, train_y, NUM_features

MAX_EVALS = 250

def loadSimulatedData(hr_ratio=2000, n=2000, m=10, num_var=2):
    data_config = SimulatedData(hr_ratio, num_var = num_var, num_features = m)
    data = data_config.generate_data(n)
    data_X = data['x']
    data_y = {'e': data['e'], 't': data['t']}
    return data_X, data_y

def loadData(feature_set_file, filename = "data//tbout_all_idfs_y5_aly.csv", tgt='idfs_y5', split=True):
    # selected features
    with open(feature_set_file, "r") as f:
        COL_TO_SET = json.load(f)

    data_all = pd.read_csv(filename)

    target = ['idfs_y5']
    ID = 'patient_id'
    L = target + [ID]
    x_cols = [x for x in data_all.columns if x not in L]

    X = data_all[x_cols]
    y = data_all[target]

    if not split:
        train_X, train_y = X, y
    else:
        sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 64)
        for train_index, test_index in sss.split(X, y):
            print train_index, test_index
            train_X = X.loc[train_index, :]
            train_y = y.loc[train_index, :]

    cols = [x for x in train_X.columns if COL_TO_SET[x] == 'O']
    train_X = train_X[cols]
    train_y = train_y[tgt]
    print "Number of rows: ", len(train_X)
    print "X cols: ", len(train_X.columns)
    print "X.column name:", train_X.columns
    return train_X, train_y

def argsTrans(args):
    params = {}
    params["hidden_layer1"] = args["hidden_layer1"] + 1
    params['hidden_layer2'] = args["hidden_layer2"] + 1
    params["learning_rate"] = args["learning_rate"] * 0.001 + 0.001
    params["learning_rate_decay"] = args["learning_rate_decay"]
    params['activation'] = args["activation"]
    params['optimizer'] = args["optimizer"]
    params['L1_reg'] = args["optimizer"] * 0.1
    params['L2_reg'] = args["optimizer"] * 0.1
    return params

def estimate_time():
    global time_start, eval_cnt
    time_now = time.clock()
    total = (time_now - time_start) / eval_cnt * (MAX_EVALS - eval_cnt)
    th = int(total / 3600)
    tm = int((total - th * 3600) / 60)
    ts = int(total - th * 3600 - tm * 60)
    print 'Estimate the remaining time: %dh %dm %ds' % (th, tm, ts)

def trainDeepSurv(args):
    global Logval, eval_cnt, time_start
    global train_X, train_y

    params = argsTrans(args)
    ds = L2DeepSurv.L2DeepSurv(self, train_X, train_y,
                                 train_X.shape[1], [params['hidden_layer1'], params['hidden_layer2']], 1,
                                 learning_rate=params['learning_rate'], 
                                 earning_rate_decay=params['earning_rate_decay'],
                                 activation=params['activation'],
                                 optimizer=params['optimizer'],
                                 L1_reg=params['L1_reg'], 
                                 L2_reg=params['L2_reg'], 
                                 dropout_keep_prob=1.0)
    ds.train(num_epoch=3000)

    ci = ds.eval(train_X, train_y)
    ds.close()

    Logval.append({'params': params, 'ci': ci})
    eval_cnt += 1
    if eval_cnt % 10 == 0:
        estimate_time()
    
    return -ci

def wtFile(filename, var):
    with open(filename, 'w') as f:
        json.dump(var, f)

def SearchParams(output_file, max_evals = 100):
    global Logval
    space = {
              "hidden_layer1": hpt.hp.randint("hidden_layer1", NUM_features),  # [1, num] = [0, num-1] + 1
              "hidden_layer2": hpt.hp.randint("hidden_layer2", NUM_features),  # [1, num] = [0, num-1] + 1
              "learning_rate": hpt.hp.randint("learning_rate", 10), # [0.001, 0.010] = 0.001 * ([0, 9] + 1)
              "learning_rate_decay": hpt.hp.choice("learning_rate_decay", [1.0, 0.999]),
              "activation": hpt.hp.choice("activation", ["relu", "sigmoid", "tanh"]),
              "optimizer": hpt.hp.choice("optimizer", ["sgd", "adam"]),
              "L1_reg": hpt.hp.randint("L1_reg", 11), # [0.0, 1.0] = 0.1 * [0, 10]
              "L2_reg": hpt.hp.randint("L2_reg", 11) # [0.0, 1.0] = 0.1 * [0, 10]
            }

    best = hpt.fmin(trainDeepSurv, space, algo = hpt.tpe.suggest, max_evals = max_evals)
    wtFile(output_file, Logval)

    print "best params:", argsTrans(best)
    print "best auc:", -trainDeepSurv(best)

def main(feature_set_file, 
         output_file,
         split = True,
         use_simulated_data=False):
    global Logval, eval_cnt, time_start, SEED
    global train_X, train_y, NUM_features
    if use_simulated_data:
        train_X, train_y = loadSimulatedData()
    else:
        train_X, train_y = loadData(feature_set_file = feature_set_file, 
                                    filename = "data//tbout_all_idfs_y5_aly_v1.csv",
                                    split = split)
    NUM_features = len(train_X[0])
    Logval = []
    eval_cnt = 0
    time_start = time.clock()
    print "Data set for SearchParams: ", len(train_X)
    SearchParams(output_file = output_file, max_evals = MAX_EVALS)

if __name__ == "__main__":
    main(feature_set_file=None, 
         output_file="data//hyperopt_log_simulated.json",
         split = False,
         use_simulated_data=True)