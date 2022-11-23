import numpy as np
import os, json, shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
import pickle

pieces_array = np.load('feats array.npy')
tag_array = np.load('tags array.npy')
print(pieces_array.shape, tag_array.shape)

X_train, X_test, y_train, y_test = train_test_split(pieces_array, tag_array, test_size = .2, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = .5, random_state = 42)

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)
y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))
print(y_train)
print(y_train.shape, y_test.shape)
num_labels = 1

test_scores = []
train_scores = []
val_scores = []

for i in [.1, .2, .3, .4, .5, .6, .7]:

    model = Sequential()

    model.add(Dense(100, input_shape = (40, )))
    model.add(Activation('relu'))
    model.add(Dropout(.8))

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(i))

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(i))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(i))

    model.add(Dense(num_labels))
    model.add(Activation('sigmoid'))

    # Display model architecture summary 
    model.summary()
    '''
    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

    num_epochs = 256
    num_batch_size = 512

    checkpointer = ModelCheckpoint(filepath = 'saved_models/'  + str(i) + '/new weights.hdf5', 
                                   verbose = 1, save_best_only = True)
    start = datetime.now()

    model.fit(X_train, y_train, batch_size = num_batch_size, epochs = num_epochs, validation_data = (X_test, y_test), callbacks = [checkpointer], verbose = 1)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    train_accuracy = model.evaluate(X_train, y_train, verbose = 0)
    test_accuracy = model.evaluate(X_test, y_test, verbose = 0)
    test_scores.append(test_accuracy[1])
    train_scores.append(train_accuracy[1])

    val_accuracy = model.evaluate(X_val, y_val, verbose = 0)
    val_scores.append(val_accuracy[1])

    filename = 'model ' + str(i) + '.h5' 
    model.save(filename)

print(train_scores)
print(test_scores)
print(val_scores)
'''