import numpy as np
import pickle
import collections
import os
import librosa
import time
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image


with open('data/train_labels', 'rt') as f:
    labels = f.readlines()
labels = [x.split(',') for x in labels]
labels = {x[1]: int(x[3]) for x in labels}
with open('data/mcm140', 'rb') as f:
    mcm = pickle.load(f)
with open('data/fmm140', 'rb') as f:
    fmm = pickle.load(f)

def norm(item):
    return -1 + (((item - -54.06) * 2) / (55.62 - -54.06))

def norm_ftrs(ftrs, fmm):
    temp = []
    for x in fmm:
        target = ftrs[x[3]:x[4]]
        target = target - x[0]
        target = -1 + (((target - x[1]) * 2) / (x[2] - x[1]))
        temp.append(target)
    return np.vstack((temp[0], temp[1], temp[2], temp[3], temp[4], temp[5])).astype(np.float16)

def train_split(item, random=True):
    match labels[item[1]]:
        case (8|10):
            factor = 20
        case (1|7):
            factor = 10
        case (13|17):
            factor = 6
        case _:
            factor = 3
    splits = []
    item_len = item[0].shape[1]
    if random:
        for _ in range(factor):
            selection = np.random.randint(0, item_len - 129)
            splits.append((item[0][:, selection:selection+128] - mcm).astype(np.float16))
    else:
        for i in range(round(item_len * 0.05), round(item_len * 0.95) - 128, round(item_len * 0.9 / factor)):
            splits.append((item[0][:, i:i+128] - mcm).astype(np.float16))
    return splits

def test_split(item):
    splits = []
    samples = 10
    item_len = item.shape[1]
    for i in range(0, item_len - 128, round(item_len / samples)):
        splits.append(norm((item[:, i:i+128] - mcm).astype(np.float16)))
    return splits

def fmm_extract(pack):
    temp = []
    seps = [20, 21, 22, 23, 35, 41]
    for i, x in enumerate(seps):
        if i == 0:
            lx = 0
        else:
            lx = seps[i - 1]
        target = [y[1][lx:x] for y in pack]
        avg = np.mean(target, axis=0)
        mn, mx = np.min(target), np.max(target)
        temp.append((avg, mn, mx, lx, x))
    return temp

def mcm_extract(pack1, pack2):
    mn1 = np.mean([x[0] for x in pack1], axis=0)
    mn2 = np.mean([x[0] for x in pack2], axis=0)
    return ((mn1 + mn2) / 2).astype(np.float16)

def select_group(ratio, even=False) -> list[str]:
    names = list(os.listdir(f'C:/Users/BBA/Coding/Audio/Classification/train'))
    count = collections.Counter(labels.values())
    np.random.shuffle(names)
    selected = []
    i = 0
    expected = round(ratio * (max(count.values()) - min(count.values())) / 2)
    for k in range(19):
        j = 0
        temp = []
        if even:
            if count[k] < expected:
                amount = count[k] - 1
            else:
                amount = expected
        else:
            amount = round(ratio * count[k])
        while j <= amount:
            try:
                if labels[names[i]] == k:
                    temp.append(names.pop(i))
                    j += 1
                else:
                    i += 1
            except IndexError:
                i = 0
        selected.extend(temp)
    np.random.shuffle(selected)
    return selected

def unified_extract(filename):
    with open(f'test/{filename}', 'rb') as f:
        song, sr = librosa.load(f, sr=None)
    scale = sr / 22050
    sgrm = librosa.stft(song, n_fft=int(2048 * scale))
    sgrm = librosa.feature.melspectrogram(S=np.abs(sgrm)**2, sr=sr, n_mels=140, fmax=6144)
    sgrm = (librosa.power_to_db(sgrm, ref=np.max) + 80).astype(np.float32)
    ftrs = np.vstack((
        np.array(Image.fromarray(librosa.feature.mfcc(y=song, sr=sr)).resize((160, 20))),
        np.array(Image.fromarray(librosa.feature.spectral_centroid(y=song, sr=sr)).resize((160, 1))),
        np.array(Image.fromarray(librosa.onset.onset_strength(y=song, sr=sr)).resize((160, 1))),
        np.array(Image.fromarray(librosa.feature.zero_crossing_rate(song)).resize((160, 1))),
        np.array(Image.fromarray(librosa.feature.chroma_stft(y=song, sr=sr)).resize((160, 12))),
        np.array(Image.fromarray(librosa.feature.tonnetz(y=song, sr=sr)).resize((160, 6)))
    ))
    ftrs = norm_ftrs(ftrs, fmm)
    sgrm = test_split(sgrm)
    #sgrm = train_split((sgrm, filename))
    return sgrm, ftrs, filename

def main():
    names = list(os.listdir(f'C:/Users/BBA/Coding/Audio/Classification/test'))
    #names = select_group(0.5, False)
    with mp.Pool(processes=7) as pool:
        data = list(tqdm(pool.imap_unordered(unified_extract, names, chunksize=8), total=len(names)))
    with open('data/f3test', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()

