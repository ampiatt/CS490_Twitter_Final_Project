import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from time import time
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
import numpy


#read in data
trainingdata = pd.read_csv('training.csv', encoding='latin1')
testingdata = pd.read_csv('testing.csv', encoding='latin1')

# reorder the data so that it mixes up the rows (keeps columns together
trainingdata = trainingdata.sample(frac=1).reset_index(drop=True)
testingdata = testingdata.sample(frac=1).reset_index(drop=True)

# tokenize data
train_data = numpy.array(trainingdata.iloc[:, 1].values)
test_data = numpy.array(testingdata.iloc[:, 1].values)
tokenize = Tokenizer(10000)
tokenize.fit_on_texts(train_data)
tokenize.fit_on_texts(test_data)
seq_train = tokenize.texts_to_sequences(train_data)
seq_test = tokenize.texts_to_sequences(test_data)

# pad or truncate the data to standardize tweet length
x_train = pad_sequences(seq_train, 50, padding='pre', truncating='pre')
x_test = pad_sequences(seq_test, 50, padding='pre', truncating='pre')

# store classification results in array
y_train = numpy.array(trainingdata.iloc[:,0].values)
y_test = numpy.array(testingdata.iloc[:,0].values)

shape = x_train.shape

tb = TensorBoard(log_dir='logs/{}'.format(time()))

lstm_model = Sequential()
lstm_model.add(Embedding(10000, 8))
lstm_model.add(Dropout(.25))
lstm_model.add(Conv1D(64, 5, padding='valid', activation='tanh'))
lstm_model.add(MaxPooling1D(pool_size=4))
lstm_model.add(LSTM(70))
lstm_model.add(Dense(1))
lstm_model.add(Activation('sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(x_train, y_train, epochs = 5, batch_size=15, callbacks=[tb])

score, accuracy = lstm_model.evaluate(x_test, y_test, verbose=0)