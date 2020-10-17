#%%
from my_classes import whole_song_pred
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l1, l2
from keras import optimizers
from sklearn.model_selection import train_test_split
import hickle as hkl
import numpy as np

#%%

mels = hkl.load('/data/data1/users/konpyro/feats/mels.hkl')
#logmels = hkl.load('/data/data1/users/konpyro/AudioFeatures/logmels.hkl')
#mfccs = hkl.load('/data/data1/users/konpyro/AudioFeatures/mfccs.hkl')
#chromas = hkl.load('/data/data1/users/konpyro/AudioFeatures/chromas.hkl')
#contrasts = hkl.load('/data/data1/users/konpyro/AudioFeatures/contrasts.hkl')
#tonnetzs = hkl.load('/data/data1/users/konpyro/AudioFeatures/tonnetzs.hkl')
labels = hkl.load('/data/data1/users/konpyro/feats/labels.hkl')
##labels2 = hkl.load('/data/data1/users/konpyro/labels_10s.hkl')
ids = hkl.load('/data/data1/users/konpyro/feats/ids.hkl')

#%%

##mels = np.asarray(mels)
##logmels = np.asarray(logmels)
##mfccs = np.asarray(mfccs)
features = np.asarray(mels) 
##features = np.stack((mels, logmels, mfccs, chromas, tonnetzs, contrasts), axis=3)
labels = np.asarray(labels)
print(features.shape, labels.shape)

#%%

Y = labels
y_t = np.zeros((Y.shape[0], 4))
for i  in range(Y.shape[0]):
    if Y[i] == 'happy':
        y_t[i][:] = [1, 0, 0, 0]
    elif Y[i] == 'angry':
        y_t[i][:] = [0, 1, 0, 0]
    elif Y[i] == 'sad':
        y_t[i][:] = [0, 0, 1, 0]
    elif Y[i] == 'relaxed':
        y_t[i][:] = [0, 0, 0, 1]
labels = y_t
Y = None
y_t = None

#%%

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# features = None
# labels = None

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])


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

epochs = 10
batch_size = 16

#%%

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy',
			optimizer=adam,
			metrics=['accuracy'])

#%%

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=False, verbose=2)

#%%

loss, acc = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', acc)

#%%

features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
pred = model.predict(features, batch_size=batch_size)
i, p, l = whole_song_pred(ids, pred, labels)
p = np.asarray(p)
l = np.asarray(l)
#print(i)
#%%

model2 = Sequential()
model2.add(Dense(16, input_shape=(4,)))
model2.add(Activation('relu'))
model2.add(Dense(4))
model2.add(Activation('softmax'))


#%%

p_train, p_test, l_train, l_test = train_test_split(p, l ,test_size = 0.2, random_state = 42)
model2.compile(loss='categorical_crossentropy',
		optimizer=adam,
		metrics=['accuracy'])

#%%

model2.fit(p_train, l_train, batch_size=batch_size, epochs=epochs, validation_data=(p_test, l_test),verbose=2)

#%%

loss, acc = model2.evaluate(p_test, l_test)
print('Test loss:', loss)
print('Test accuracy:', acc)

