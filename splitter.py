import torch
import pickle
import collections
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

with open('data/mcm140', 'rb') as f:
    mcm = pickle.load(f)
with open('data/train_labels', 'rt') as f:
    labels = f.readlines()
labels = [x.split(',') for x in labels]
labels = {x[1]: int(x[3]) for x in labels}

def load_test(filename, vmode=False):
    with open(f'data/{filename}', 'rb') as f:
        adata = pickle.load(f)
    temp = []
    for x in adata:
        if vmode:
            temp.append(([(torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0), torch.tensor(x[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)) for y in x[0]], labels[x[2]]))
        else:
            temp.append(([(torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0), torch.tensor(x[1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)) for y in x[0]], x[2]))
    return temp

def load_train(ratio1, ratio2):
    with open(f'data/feats', 'rb') as f:
        feats = pickle.load(f)
    feats = {x[1]: x[0] for x in feats}
    with open(f'data/specs', 'rb') as f:
        specs = pickle.load(f)
    specs = {x[1]: x[0] for x in specs}
    #adata = subgroup([(specs[x].astype(np.float16), feats[x], labels[x]) for x in labels], ratio1, 500, 10)
    adata = split_data([(specs[x].astype(np.float16), feats[x], labels[x]) for x in labels], ratio2)
    return adata

def subgroup(dataset, ratio, upper_lim, lower_lim):
    temp = []
    counts = collections.Counter([x[2] for x in dataset])
    for x in counts:
        cat_count = 0
        i = 0
        split_goal = round(counts[x] * ratio)
        if split_goal > upper_lim:
            split_goal = upper_lim
        if split_goal < lower_lim:
            split_goal = lower_lim
        if split_goal > counts[x]:
            split_goal = counts[x]
        while cat_count < split_goal:
            if dataset[i][2] == x:
                temp.append(dataset.pop(i))
                cat_count += 1
            else:
                i += 1
    return dataset, temp

def split_data(dataset: list[tuple[torch.Tensor, int]], ratio: float) -> tuple[list[list[tuple[torch.Tensor, int]]], list[tuple[torch.Tensor, int]]]:
    np.random.shuffle(dataset)
    tsplit, vsplit = subgroup(dataset, ratio, 999999, 0)
    with mp.Pool(processes=6) as pool:
        tdata = list(tqdm(pool.imap_unordered(spectro_split, tsplit, chunksize=256), total=len(tsplit)))
    tdata = [y for x in tdata for y in x]
    with mp.Pool(processes=6) as pool:
        vdata = list(tqdm(pool.imap_unordered(spectro_split, vsplit, chunksize=256), total=len(vsplit)))
    vdata = [y for x in vdata for y in x]
    return tdata, vdata

def spectro_split(pack):
    shards = int(pack[0].shape[1] / 128 / 1.4)
    temp = []
    for i in range(0, pack[0].shape[1] - 128, int(pack[0].shape[1] / shards)):
        temp.append(((pack[0][:, i:i+128] - mcm, pack[1]), pack[2]))
    return tuple(temp)

def tensorfy(item):
    return ((torch.from_numpy(np.expand_dims(item[0][0], axis=0).astype(np.float32)), torch.from_numpy(np.expand_dims(item[0][1], axis=0).astype(np.float32))), item[1])

def tconvert(pack):
    with mp.Pool(processes=6) as pool:
        tpack = list(tqdm(pool.imap_unordered(tensorfy, pack, chunksize=512), total=len(pack)))
    return tpack
