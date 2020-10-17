#%%

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l1, l2
from keras import optimizers
from my_classes import DataGenerator
import hickle as hkl
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

#%%
data_len = 37989
ids = ['' for x in range(data_len)]
for i in range(data_len):
    ids[i] = '{:05d}'.format(i)
# ints = np.arange(data_len)
# ids = np.char.mod('%d', ints)
targets = hkl.load('/data/data1/users/konpyro/feats/labels.hkl')
labels = {}
for i in range(data_len):
    target = targets.pop()
    if target == 'happy':
        labels[ids[i]] = 0
    elif target == 'angry':
        labels[ids[i]] = 1
    elif target == 'sad':
        labels[ids[i]] = 2
    elif target == 'relaxed':
        labels[ids[i]] = 3
np.random.shuffle(ids)
stop = int((data_len * 8) / 10)  # validation data 20%
partition = {
    'train': ids[:stop],
    'validation':  ids[stop:]
}
# with open('/home/konpyro/SHMMY/Thesis/Code/Datasets/AudioFeatures/test/0_mel.pkl', 'rb') as file:
#    input1 = pkl.load(file)
input_shape = (60, 431, 6)

model = Sequential()
model.add(Conv2D(128, (6, 6), input_shape=input_shape, kernel_regularizer=l2(0.00001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (5, 5), padding='same', kernel_regularizer=l2(0.00001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (4, 4), padding='same', kernel_regularizer=l2(0.00001)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(16, kernel_regularizer=l2(0.00001)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))
# model.add(Dense(16))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.summary()

#%%
# Parameters
params = {'dim': (60, 431),
          'batch_size': 16,
          'n_classes': 4,
          'n_channels': 6,
          'shuffle': True}

epochs = 10

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

#%%

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])

#%%

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    verbose=1)

loss, acc = model.evaluate_generator(validation_generator)
print('Test loss:', loss)
print('Test accuracy:', acc)
#model.save('/data/data1/users/konpyro/model_A1.h5')
#with open('/data/data1/users/konpyro/history/b16_m_c_t_s.pkl', 'wb') as file_pi:
#    pkl.dump(history.history, file_pi)
