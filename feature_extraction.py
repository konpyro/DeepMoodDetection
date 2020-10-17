import scipy
import librosa
import csv
import numpy as np
from scipy import signal
import hickle as hkl
import librosa.display

sr = 22050
n_fft = 1024
hop_length = 512
n_mels = 60
n_mfcc = 60
n_chroma = 60
n_bands = 5
window = scipy.signal.hanning

margin_i, margin_v = 2, 10
power = 2
err = 0

def chunking(song, sr):
    chunk_length_ms = 10
    chunk_length = chunk_length_ms * sr
    total_chunks = song.shape[0] // chunk_length
    chunks = [song[chunk_length * i:chunk_length * (i + 1)] for i in range(total_chunks)]
    return chunks


def specting(y, sr):
    '''
    S_full, phase = librosa.magphase(librosa.stft(y))

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)
    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    D_foreground = S_foreground * phase
    y_foreground = librosa.istft(D_foreground)
    D_background = S_background * phase
    y_background = librosa.istft(D_background)
'''
    MEL = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    LOGMEL = librosa.power_to_db(MEL)
    
    MFCC = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc)
   
    CHROMA = librosa.feature.chroma_stft(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    TONNETZ = librosa.feature.tonnetz(y, sr=sr)
    CONTRAST = librosa.feature.spectral_contrast(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    '''
    FOREGR = librosa.feature.melspectrogram(y_foreground, sr=sr, n_fft=n_fft,
                                            hop_length=hop_length,
                                            n_mels=n_mels)
    BACKGR = librosa.feature.melspectrogram(y_background, sr=sr, n_fft=n_fft,
                                            hop_length=hop_length,
                                            n_mels=n_mels)
    '''
    # fix dimensions to 60x1292 for all features
    TONNETZ = np.repeat(TONNETZ, 10, axis=0)
    CONTRAST = np.repeat(CONTRAST, 10, axis=0)
   
    return MEL, LOGMEL, MFCC, CHROMA, TONNETZ, CONTRAST ##, FOREGR, BACKGR


# %%

mels = []
logmels = []
mfccs = []
chromas = []
tonnetzs = []
contrasts = []
##foregrs = []
##backgrs = []
labels = []
ids = []

# %%

with open('/data/data1/users/konpyro/HandcraftedDataset2.csv') as rfile:
    csv_reader = csv.reader(rfile, delimiter=",")

    count = 0
    for row in csv_reader:
        if count > 0:
            try:
                y, sr = librosa.load(path="/data/data1/users/konpyro/HandDataset/" + row[1] + ' ' + row[2] + '.mp3', sr=sr, mono=True)
            except:
                print("Error loading: " + row[0] + '-' + row[1] + ' ' + row[2])
                err = err + 1
                continue

            chunks = chunking(y, sr)
            for i, chunk in enumerate(chunks):
                mel, logmel, mfcc, chroma, tonnetz, contrast = specting(chunk, sr)
                mels.append(mel)
                logmels.append(logmel)
                mfccs.append(mfcc)
                chromas.append(chroma)
                tonnetzs.append(tonnetz)
                contrasts.append(contrast)
                ##foregrs.append(foregr)
                ##backgrs.append(backgr)
                labels.append(row[3])
                ids.append(row[0])

        count += 1


hkl.dump(mels, "/data/data1/users/konpyro/feats/h_mels.hkl")
hkl.dump(logmels,"/data/data1/users/konpyro/feats/h_logmels.hkl")
hkl.dump(mfccs, "/data/data1/users/konpyro/feats/h_mfccs.hkl")
hkl.dump(chromas, "/data/data1/users/konpyro/feats/h_chromas.hkl")
hkl.dump(tonnetzs, "/data/data1/users/konpyro/feats/h_tonnetzs.hkl")
hkl.dump(contrasts, "/data/data1/users/konpyro/feats/h_contrasts.hkl")
##hkl.dump(foregrs, "foregrs.hkl")
##hkl.dump(backgrs, "backgrs.hkl")
hkl.dump(labels, "/data/data1/users/konpyro/feats/h_labels.hkl")
hkl.dump(ids, "/data/data1/users/konpyro/feats/h_ids.hkl")

print(err)
print(len(logmels))
'''
n_fft = 1024
hop_length = 512
n_mels = 60
n_mfcc = 60
n_chroma = 60
n_bands = 5
window = scipy.signal.hanning

margin_i, margin_v = 2, 10
power = 2

y, sr = librosa.load('chunk0.wav')
print(sr)

S_full, phase = librosa.magphase(librosa.stft(y))

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)
mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

S_foreground = mask_v * S_full
S_background = mask_i * S_full
D_foreground = S_foreground * phase
y_foreground = librosa.istft(D_foreground)
D_background = S_background * phase
y_background = librosa.istft(D_background)

MEL = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   n_mels=n_mels)
LOGMEL = librosa.power_to_db(MEL)
MFCC = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc)
CHROMA = librosa.feature.chroma_stft(y, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   n_chroma=n_chroma)
TONNETZ = librosa.feature.tonnetz(y, sr=sr)
CONTRAST = librosa.feature.spectral_contrast(y, sr=sr, n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_bands=n_bands)

FOREGR = librosa.feature.melspectrogram(y_foreground, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   n_mels=n_mels)
BACKGR = librosa.feature.melspectrogram(y_background, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   n_mels=n_mels)

#fix dimensions to 60x1292 for all features
TONNETZ = np.repeat(TONNETZ, 10, axis=0)
CONTRAST = np.repeat(CONTRAST, 10, axis=0)
print(MEL.shape, LOGMEL.shape, MFCC.shape, CHROMA.shape, TONNETZ.shape, CONTRAST.shape)

plt.figure(figsize=(12, 16))
plt.subplot(6, 1, 1)
librosa.display.specshow(MEL, sr=sr)
plt.title('MEL')
plt.colorbar()

plt.subplot(6, 1, 2)
librosa.display.specshow(LOGMEL, sr=sr)
plt.title('LOGMEL')
plt.colorbar()

plt.subplot(6, 1, 3)
librosa.display.specshow(MFCC, sr=sr)
plt.title('MFCC')
plt.colorbar()

plt.subplot(6, 1, 4)
librosa.display.specshow(CHROMA, sr=sr)
plt.title('CHROMA')
plt.colorbar()

plt.subplot(6, 1, 5)
librosa.display.specshow(TONNETZ, sr=sr)
plt.subplot(6, 1, 5)
librosa.display.specshow(TONNETZ, sr=sr)
plt.subplot(6, 1, 5)
librosa.display.specshow(TONNETZ, sr=sr)
plt.title('TONNETZ')
plt.colorbar()

plt.subplot(6, 1, 6)
librosa.display.specshow(CONTRAST, sr=sr)
plt.title('CONTRAST')
plt.colorbar()
'''
