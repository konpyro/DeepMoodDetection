import numpy as np
import keras
import pickle as pkl
from collections import Counter

def whole_song_pred(IDs, X, Y):
    mean_song_predictions = []
    song_labels = []
    song_ids = []
    cnt = 0
    id_dict = Counter(IDs)
    for x in id_dict.values():
        song_ids.append(IDs[cnt]) 
        song_labels.append(Y[cnt])
        song_data = X[cnt:(cnt + x)]
        cnt = cnt + x
        meso = np.mean(song_data, axis=0)
        mean_song_predictions.append(meso)
    return song_ids, mean_song_predictions, song_labels


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            with open('/data/data1/users/konpyro/feats/mels/' + ID + '_mel.pkl', 'rb') as file:
                x1 = pkl.load(file)
            with open('/data/data1/users/konpyro/feats/logmels/'+ ID + '_logmel.pkl', 'rb') as file:
                x2 = pkl.load(file)
            with open('/data/data1/users/konpyro/feats/mfccs/' + ID + '_mfcc.pkl', 'rb') as file:
                x3 = pkl.load(file)	      
            with open('/data/data1/users/konpyro/feats/chromas/' + ID + '_chroma.pkl', 'rb') as file:
                x4 = pkl.load(file)
            with open('/data/data1/users/konpyro/feats/tonnetzs/' + ID + '_tonnetz.pkl', 'rb') as file:
                x5 = pkl.load(file)
                x5 = np.repeat(x5, 10, axis=0)
            with open('/data/data1/users/konpyro/feats/contrasts/' + ID + '_contrast.pkl', 'rb') as file:
                x6 = pkl.load(file)
                x6 = np.repeat(x6, 10, axis=0)
            X[i,] = np.stack((x1, x2, x3, x4, x5, x6), axis=2)
            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
