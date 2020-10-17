# !CAUTION mporoun na xrisimopoithoun oles oi provlepseis kai ohi mono oses proerxontai apo tragoudia pou ehoun kai stihous kai iho.

from keras.models import Sequential, load_model, model_from_json
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Layer
from keras import optimizers
from sklearn.model_selection import train_test_split
from my_classes import DataGenerator, whole_song_pred
import bertlayer 
import os
import numpy as np
import pickle as pkl
import hickle as hkl

# Loading audio data
data_len = 37989
#ints = np.arange(data_len)
ids = ['' for x in range(data_len)]
for i in range(data_len):
    ids[i] = '{:05d}'.format(i)
targets = hkl.load('/data/data1/users/konpyro/feats/labels.hkl')
audio_ids = hkl.load('/data/data1/users/konpyro/feats/ids.hkl')
labels = {}
a_labels = np.zeros((data_len,1))
for i in range(data_len):
    target = targets.pop(0)
    if target == 'happy':
        labels[ids[i]] = 0
        a_labels[i] = 0
    elif target == 'angry':
        labels[ids[i]] = 1
        a_labels[i] = 1
    elif target == 'sad':
        labels[ids[i]] = 2
        a_labels[i] = 2
    elif target == 'relaxed':
        labels[ids[i]] = 3
        a_labels[i] = 3
#np.random.shuffle(ids)

#Loading text data
with open('/data/data1/users/konpyro/text_feats/input_ids.pkl', 'rb') as file:
    input_ids = pkl.load(file)
with open('/data/data1/users/konpyro/text_feats/input_masks.pkl', 'rb') as file:
    input_masks = pkl.load(file)
with open('/data/data1/users/konpyro/text_feats/segment_ids.pkl', 'rb') as file:
    segment_ids = pkl.load(file)
with open('/data/data1/users/konpyro/text_feats/labels.pkl', 'rb') as file:
    t_labels = pkl.load(file)
with open('/data/data1/users/konpyro/text_feats/ids.pkl', 'rb') as file:
    text_ids = pkl.load(file)


# Parameters
params = {'dim': (60, 431),
          'batch_size': 16,
          'n_classes': 4,
          'n_channels': 6,
          'shuffle': False}

epochs = 10

# Generators
generator = DataGenerator(ids, labels, **params)

# Late fuse model
model_A1 = load_model('/data/data1/users/konpyro/model_A1.h5')
#model_T2 = tf.keras.models.load_model('/data/data1/users/konpyro/model_T2.h5', custom_objects={'BertLayer': bertlayer.BertLayer})
# load json and create model
json_file = open('/data/data1/users/konpyro/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_T2 = model_from_json(loaded_model_json, custom_objects={'BertLayer': bertlayer.BertLayer})
# load weights into new model
model_T2.load_weights("/data/data1/users/konpyro/model_weights.h5")
print("Loaded model from disk")


input_shape = (8,)
model_fuse = Sequential()
model_fuse.add(Dense(16, input_shape=input_shape))
model_fuse.add(Activation('relu'))
model_fuse.add(Dense(4))
model_fuse.add(Activation('softmax'))

# Get predictions
print('Making predictions...')
p1 = model_A1.predict_generator(generator)
p2 = model_T2.predict([input_ids, input_masks, segment_ids], batch_size=128)

print(p1.shape, p2.shape)

# Get predictions per song
i_a, p_a, l_a = whole_song_pred(audio_ids, p1, a_labels)
i_t, p_t, l_t = whole_song_pred(text_ids, p2, t_labels)

print('Audio:', len(i_a), len(p_a), len(l_a), i_a[:5])
print('Text:', len(i_t), len(p_t), len(l_t), i_t[:5])

# Parallelize data
intersection = list(set(i_a).intersection(set(i_t)))
print('Interxestion:', intersection[:10])
pred_a = []
pred_t = []
labe_a = []
labe_t = []
for key in intersection:
    id1 = i_a.index(key)
    id2 = i_t.index(key)
    pred_a.append(p_a[id1])
    labe_a.append(l_a[id1])
    pred_t.append(p_t[id2])
    labe_t.append(l_t[id2])
#    print(key, id1, id2, l_a[id1], l_t[id2])

pred = np.concatenate((pred_a, pred_t), axis=1)
print('Pred shape', pred.shape, pred[0])
# Model fit
labe = to_categorical(labe_t)
x_train, x_test, y_train, y_test = train_test_split(pred, labe, test_size = 0.2, random_state = 42)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model_fuse.compile(loss='categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])

history = model_fuse.fit(x_train, y_train, batch_size=16, epochs=epochs, validation_split=0.1, verbose=2)

loss, acc = model_fuse.evaluate(x_test, y_test)
print('Model accuracy:', acc)
print('Model loss:', loss)

with open('/data/data1/users/konpyro/history/fuse.pkl', 'wb') as file:
    pkl.dump(history.history, file)

