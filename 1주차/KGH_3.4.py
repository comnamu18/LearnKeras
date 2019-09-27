import keras
from keras.datasets import imdb
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print(train_data.shape)

# print(train_data[0])
# print(train_labels[0])

# 0~9999 : 10000개의 word index
# print(max([max(sequence) for sequence in train_data]))

word_index = imdb.get_word_index()
# print(word_index)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# print(reverse_word_index)

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# print(x_train.shape)
# print(x_train[8134])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))    # ReLU / tanh
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'] )

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

print(model.predict(x_test))