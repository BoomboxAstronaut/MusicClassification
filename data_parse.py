import numpy as np
import collections
import librosa
import pickle
import os
import cv2 as cv
import multiprocessing as mp
from PIL import Image


with open('train_labels', 'rt') as f:
    labels = f.readlines()
labels = [x.split(',') for x in labels]
labels = {x[1]: int(x[3]) for x in labels}


def extract_audftrs(filename):
    with open(f'train/{filename}', 'rb') as f:
        sraw, sr = librosa.load(f, sr=None)
    scale = sr / 22050
    sgrm = librosa.stft(sraw, n_fft=int(2048 * scale))
    sgrm = librosa.feature.melspectrogram(S=np.abs(sgrm)**2, sr=sr, n_mels=140, fmax=6144)
    #cgram = np.uint8(np.array(Image.fromarray(librosa.feature.chroma_stft(S=np.abs(sgrm), sr=sr)).resize((240, 12))) * 100)
    sgrm = librosa.power_to_db(sgrm, ref=np.max) + 80
    sgrm = sgrm - np.mean(sgrm)
    sgrm[sgrm < 0] = 0
    return (labels[filename], np.uint8(sgrm))


def test_prep(filename):
    samples = 5
    with open(f'testing/{filename}', 'rb') as f:
        sraw, sr = librosa.load(f, sr=None)
    scale = sr / 22050
    sgrm = librosa.stft(sraw, n_fft=int(2048 * scale))
    sgrm = librosa.feature.melspectrogram(S=np.abs(sgrm)**2, sr=sr, n_mels=140, fmax=6144)
    sgrm = librosa.power_to_db(sgrm, ref=np.max) + 80
    sgrm = sgrm - np.mean(sgrm)
    sgrm[sgrm < 0] = 0
    sgrm = np.uint8(sgrm)
    portions = []
    if sgrm.shape[1] < 128:
        sgrm = np.array(Image.fromarray(sgrm).resize((128, 128)))
        return ([sgram], filename)
    for i in range(0, sgrm.shape[1] - 129, round(sgrm.shape[1] / samples) - 2):
        portions.append(sgrm[:, i:i+128])
    return portions, filename


def split_data1(items):
    match items[0]:
        case (0|1|2|3|4):
            factor = 7
        case (5|6|7|8|9|10):
            factor = 8
        case (11|12|13):
            factor = 8
        case 14:
            factor = 12
        case (15|16):
            factor = 14
        case 17:
            factor = 20
        case 18:
            factor = 36
    splits = []
    for i in range(0, items[1].shape[1] - 129, round(items[1].shape[1] / factor) - 2):
        splits.append((items[1][:, i:i+128], items[0]))
    return splits


def split_data2(items):
    match items[0]:
        case (0|1|2|3|4):
            factor = 7
        case (5|6|7|8|9|10):
            factor = 8
        case (11|12|13):
            factor = 8
        case 14:
            factor = 12
        case (15|16):
            factor = 14
        case 17:
            factor = 20
        case 18:
            factor = 36
    splits = []
    for _ in range(factor):
        selection = np.random.randint(0, items[1].shape[1] - 129)
        splits.append((items[1][:, selection:selection+128], items[0]))
    return splits


def main_1():
    #Extract and prep for testing
    names = list(os.listdir(r'C:\Users\BBA\Coding\Audio\Classification\testing'))
    with mp.Pool(processes=6) as pool:
        data = pool.map(test_prep, names)
    with open('data/testing140f', 'wb') as f:
        pickle.dump(data, f)


def main_2():
    #Full extracting from train set
    names = [x for x in labels]
    with mp.Pool(processes=6) as pool:
        odata = pool.map(extract_audftrs, names)
    with open('data/140ffull', 'wb') as f:
        pickle.dump(odata, f)
    #with open(f'data/xgroup1', 'rb') as f:
    #    odata = pickle.load(f)
    with mp.Pool(processes=6) as pool:
        data = pool.map(split_data1, odata)
    temp = []
    for x in data:
        for y in x:
            temp.append(y)
    with open('data/140feven', 'wb') as f:
        pickle.dump(temp, f)
    with mp.Pool(processes=6) as pool:
        data = pool.map(split_data2, odata)
    temp = []
    for x in data:
        for y in x:
            temp.append(y)
    with open('data/140frand', 'wb') as f:
        pickle.dump(temp, f)


def main_3():
    #grab samples from trainset
    with open(f'data/full140f', 'rb') as f:
        audio_data = pickle.load(f)
    counts = collections.Counter([x[0] for x in audio_data])
    np.random.shuffle(audio_data)
    sdata = []
    for x in counts.items():
        gcounter = 0
        i = 0
        while gcounter < int(x[1] * 0.03):
            if audio_data[i][0] == x[0]:
                sdata.append(audio_data.pop(i))
                gcounter += 1
            i += 1
    for x in audio_data:
        if x[0] == 18:
            sdata.append(x)
            break
    with mp.Pool(processes=6) as pool:
        data = pool.map(split_data1, sdata)
    temp = []
    for x in data:
        for y in x:
            temp.append(y)
    np.random.shuffle(temp)
    with open('data/samp140f', 'wb') as f:
        pickle.dump(temp, f)


if __name__ == '__main__':
    main_3()