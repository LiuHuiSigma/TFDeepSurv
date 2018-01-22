import numpy as np
import random
import pandas as pd
from sklearn.model_selection import ShuffleSplit

def prepare_data(x, label):
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    # Sort Training Data for Accurate Likelihood
    sort_idx = np.argsort(t)[::-1]
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return x, {'e': e, 't': t}

def parse_data(x, label):
    # sort data by t
    x, label = prepare_data(x, label)
    e, t = label['e'], label['t']

    failures = {}
    atrisk = {}
    n, cnt = 0, 0

    for i in range(len(e)):
        if e[i]:
            if t[i] not in failures:
                failures[t[i]] = [i]
                n += 1
            else:
                # ties occured
                cnt += 1
                failures[t[i]].append(i)

            if t[i] not in atrisk:
                atrisk[t[i]] = []
                for j in range(0, i+1):
                    atrisk[t[i]].append(j)
            else:
                atrisk[t[i]].append(i)
    # when ties occured frequently
    if cnt >= n / 2:
        ties = 'efron'
    elif cnt > 0:
        ties = 'breslow'
    else:
        ties = 'noties'

    return x, e, t, failures, atrisk, ties

def loadSimulatedData(hr_ratio=2000, n=2000, m=10, num_var=2, seed=1):
    data_config = SimulatedData(hr_ratio, num_var = num_var, num_features = m)
    data = data_config.generate_data(n, seed=seed)
    data_X = data['x']
    data_y = {'e': data['e'], 't': data['t']}
    return data_X, data_y

def loadData(filename = "data//surv_aly_idfs.csv", 
             tgt={'e': 'idfs_bin', 't': 'idfs_month'}, 
             split=1.0,
             Normalize=True,
             seed=40):
    data_all = pd.read_csv(filename)

    ID = 'patient_id'
    target = list(tgt.values())
    L = target + [ID]
    x_cols = [x for x in data_all.columns if x not in L]

    X = data_all[x_cols]
    y = data_all[target]
    # Normalized data
    if Normalize:
        for col in X.columns:
            X.loc[:, col] = (X.loc[:, col] - X.loc[:, col].mean()) / (X.loc[:, col].max() - X.loc[:, col].min())
    # Split data
    if split == 1.0:
        train_X, train_y = X, y
    else:
        sss = ShuffleSplit(n_splits = 1, test_size = 1-split, random_state = seed)
        for train_index, test_index in sss.split(X, y):
            train_X, test_X = X.loc[train_index, :], X.loc[test_index, :]
            train_y, test_y = y.loc[train_index, :], y.loc[test_index, :]
    # print information about train data
    print("Number of rows: ", len(train_X))
    print("X cols: ", len(train_X.columns))
    print("Y cols: ", len(train_y.columns))
    print("X.column name:", train_X.columns)
    print("Y.column name:", train_y.columns)
    # Transform type of data to np.array
    train_X = train_X.values.astype(np.float32)
    train_y = {'e': train_y[tgt['e']].values.astype(np.int32),
               't': train_y[tgt['t']].values.astype(np.float32)}
    if split == 1.0:
        return train_X, train_y
    else:
        test_X = test_X.values.astype(np.float32)
        test_y = {'e': test_y[tgt['e']].values.astype(np.int32),
                  't': test_y[tgt['t']].values.astype(np.float32)}
        return train_X, train_y, test_X, test_y

def readData(file0, file1, discount):
    random.seed(1)
    # read file
    data0 = pd.read_csv(file0)
    data1 = pd.read_csv(file1)
    names = np.array(list(data0))
    if data0.isnull().any().any():
        data0 = data0.fillna(0)
    if data1.isnull().any().any():
        data1 = data1.fillna(0)

    TRAIN_NUM0 = data0.shape[0]
    TRAIN_NUM1 = data1.shape[0]
    FEATURE_NUM = data0.shape[1]-2

    # randomly split data0 and data1 to train and test
    cnt0 = range(TRAIN_NUM0)
    random.shuffle(cnt0)
    trainidx0 = cnt0[0:int(TRAIN_NUM0 * discount)]
    testidx0 = cnt0[int(TRAIN_NUM0 * discount) + 1:TRAIN_NUM0]

    cnt1 = range(TRAIN_NUM1)
    random.shuffle(cnt1)
    trainidx1 = cnt1[0:int(TRAIN_NUM1 * discount)]
    testidx1 = cnt1[int(TRAIN_NUM1 * discount) + 1:TRAIN_NUM1]

    # generate train data0
    x0 = data0.values[trainidx0, 0:FEATURE_NUM]
    t0 = data0.values[trainidx0, FEATURE_NUM]
    e0 = data0.values[trainidx0, FEATURE_NUM+1]

    # generate train data1
    x1 = data1.values[trainidx1, 0:FEATURE_NUM]
    t1 = data1.values[trainidx1, FEATURE_NUM]
    e1 = data1.values[trainidx1, FEATURE_NUM+1]

    x = np.concatenate((x0, x1), axis=0)
    t = np.concatenate((t0, t1), axis=0)
    e = np.concatenate((e0, e1), axis=0)
    #normlize
    df = pd.DataFrame(x)
    df = df.replace(-1, 9)
    ndf = (df - df.min()) / (df.max() - df.min())
    x = ndf.values
    train_data = {
        'x': x.astype(np.float32),
        't': t.astype(np.int8),
        'e': e.astype(np.int8)
    }

    #train_data['x'] -= np.mean(train_data['x'], axis=0)
    # print 'train t=1: min-', min(t1), ' max-', max(t1), ' mean-', np.mean(t1)

    # generate test data
    x0 = data0.values[testidx0, 0:FEATURE_NUM]
    t0 = data0.values[testidx0, FEATURE_NUM]
    e0 = data0.values[testidx0, FEATURE_NUM+1]

    x1 = data1.values[testidx1, 0:FEATURE_NUM]
    t1 = data1.values[testidx1, FEATURE_NUM]
    e1 = data1.values[testidx1, FEATURE_NUM+1]

    x = np.concatenate((x0, x1), axis=0)
    t = np.concatenate((t0, t1), axis=0)
    e = np.concatenate((e0, e1), axis=0)
    #normlize
    df = pd.DataFrame(x)
    df = df.replace(-1, 9)
    ndf = (df - df.min()) / (df.max() - df.min())
    x = ndf.values

    test_data = {
        'x': x.astype(np.float32),
        't': t.astype(np.int8),
        'e': e.astype(np.int8)
    }
    #test_data['x'] -= np.mean(test_data['x'], axis=0)
    # print 'test t=1: min-', min(t1), ' max-', max(t1), ' mean-', np.mean(t1)
    return (train_data, test_data, names)