import numpy as np
import librosa
import pickle
import os
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
    dbsg = librosa.feature.melspectrogram(S=np.abs(sgrm)**2, sr=sr, fmax=4096)
    #cgram = np.uint8(np.array(Image.fromarray(librosa.feature.chroma_stft(S=np.abs(sgrm), sr=sr)).resize((240, 12))) * 100)
    return (labels[filename], np.uint8(np.abs(librosa.power_to_db(dbsg, ref=np.max)) * 2))


def test_prep(filename):
    samples = 6
    fmax = 4096
    with open(f'test/{filename}', 'rb') as f:
        sraw, sr = librosa.load(f, sr=None)
    scale = sr / 22050
    sgram = librosa.stft(sraw, n_fft=int(2048 * scale))
    msgram = librosa.feature.melspectrogram(S=np.abs(sgram)**2, sr=sr, fmax=fmax)
    dbmsgram = np.uint8(np.abs(librosa.power_to_db(msgram, ref=np.max)) * 2)
    portions = []
    if dbmsgram.shape[1] < 128:
        dbmsgram = np.array(Image.fromarray(np.uint8(dbmsgram)).resize((128, 128)))
    for i in range(round(dbmsgram.shape[1] * 0.05), dbmsgram.shape[1] - 129, round(dbmsgram.shape[1] / samples)):
        portions.append(dbmsgram[:, i:i+128])
    return portions, filename


def split_data(items):
    match items[0]:
        case (5|6):
            factor = 8
        case (7|8|11):
            factor = 3
        case 10:
            factor = 10
        case (16|15):
            factor = 20
        case 17:
            factor = 28
        case _:
            factor = 2
    splits = []
    for _ in range(factor):
        selection = np.random.randint(0, items[1].shape[1] - 129)
        splits.append((items[1][:, selection:selection+128], items[0]))
    return splits


if __name__ == '__main__':
    names = list(os.listdir(r'C:\Users\BBA\Coding\Audio\Classification\test'))
    with mp.Pool(processes=6) as pool:
        data = pool.map(test_prep, names)
    with open('tgroup1', 'wb') as f:
        pickle.dump(data, f)