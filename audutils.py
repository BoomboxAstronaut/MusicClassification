import numpy as np
import librosa

cofs = ['c', 'g', 'd', 'a', 'e', 'b', 'f_g', 'c_d', 'g_a', 'd_e', 'a_b', 'f']
pitches = ['c', 'c_d', 'd', 'd_e', 'e', 'f', 'f_g', 'g', 'g_a', 'a', 'a_b', 'b']
sharps = ['f_g', 'c_d', 'g_a', 'd_e', 'a_b', 'f', 'c']
flats = ['a_b', 'd_e', 'g_a', 'c_d', 'f_g', 'b', 'e']
ksigs = dict()

for i in range(7):
    j = i
    temp = sharps.copy()
    while j >= 0:
        temp[6-j] = pitches[pitches.index(sharps[6-j]) - 1]
        j -= 1
    ksigs.update({tuple(sorted(temp)): cofs[6-i]})

for i  in range(6):
    j = i
    temp = flats.copy()
    while j >= 0:
        if pitches.index(flats[6-j]) == 11:
            temp[6-j] = pitches[0]
        else:
            temp[6-j] = pitches[pitches.index(flats[6-j]) + 1]
        j -= 1
    ksigs.update({tuple(sorted(temp)): cofs[6+i]})

ksigs.update({tuple(sorted(sharps)): 'c_d'})
ksigs.update({tuple(sorted(flats)): 'b'})

def samptime(seconds):
    return 22050 * seconds

def key_finder(chrkey, tones):
    chrkey = np.array([chrkey[0], chrkey[7], chrkey[2], chrkey[9], chrkey[4], chrkey[11], chrkey[6], chrkey[1], chrkey[8], chrkey[3], chrkey[10], chrkey[5]])
    sums = []
    tones = [np.sum(x) for x in tones]
    tones = [tones[x] + tones[x+1] for x in range(0, 5, 2)]
    for i in range(-1, 5):
        if np.sum(chrkey[i]) > np.sum(chrkey[i-5]):
            sums.append((np.sum(chrkey[i]), cofs[i]))
        else:
            sums.append((np.sum(chrkey[i-5]), pitches[i-5]))
    print(sums)
    k = 0
    while k < 4:
        code = tuple(sorted([x[1] for x in sorted(sums, reverse=True)[:7-k]]))
        for x in ksigs.keys():
            if all((y in x for y in code)):
                primary = ksigs[x]
                k = 4
        k += 1
    if 'primary' not in locals():
        return False
    if tones[1] + tones[2] < 0:
        return f'{cofs[cofs.index(primary) - 9]}_m'
    else:
        return f'{primary}_M'

def split_bass(sgram, max_ref):
    bassgram = librosa.amplitude_to_db(np.abs(sgram[4:37]), ref=max_ref)
    vocgram = librosa.amplitude_to_db(np.abs(sgram[38:400]), ref=max_ref)
    return bassgram, vocgram