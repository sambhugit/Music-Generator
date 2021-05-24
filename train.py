from __future__ import print_function
import keras
import numpy as np
import os
import random
import sys
import time
import codecs
import collections
from six.moves import cPickle
from keras import layers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy

current_dir = os.path.dirname(os.path.realpath('inference.py'))

def loadandprocess():
    
    with open(os.path.join(current_dir,"Datasets/data1.txt")) as n:
        training_data1 = n.readlines()

    with open(os.path.join(current_dir,"Datasets/data2.txt")) as l:
        training_data2 = l.readlines()

    with open(os.path.join(current_dir,"Datasets/data3.txt")) as m:
        training_data3 = m.readlines()

    training_data = training_data1 + training_data2 +training_data3

    z_n = ''

    for x in training_data:
        x_n = [word for word in x if word not in ('\n','\t','.','(',')','2','-')]
        x_n = ''.join(x_n)
        x_n = x_n.lower()
        z_n = z_n + ' ' + x_n

    wordlist = z_n.split()

    return wordlist

def vocabcreate( wordlist ):

    word_counts = collections.Counter(wordlist)

    # Mapping from index to word : that's the vocabulary
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Mapping from word to index
    vocab = {x: i for i, x in enumerate(vocabulary_inv)}
    words = [x[0] for x in word_counts.most_common()]

    #size of the vocabulary
    vocab_size = len(words)
    print("vocab size: ", vocab_size)

    #save the words and vocabulary
    with open(os.path.join(current_dir,'vocab.txt'), 'wb') as f:
        cPickle.dump((words, vocab, vocabulary_inv), f)

    sequences = []
    next_words = []
    seq_length = 30
    sequences_step = 1
    for i in range(0, len(wordlist) - seq_length, sequences_step):
        sequences.append(wordlist[i: i + seq_length])
        next_words.append(wordlist[i + seq_length])

    print('nb sequences:', len(sequences))

    return sequences , next_words , vocab_size , vocab

def generator(batch_size, sequences, seq_length, vocab_size, vocab, next_words):
    
    while True:    
      index = 0    
      X_batch = np.zeros((batch_size, seq_length, vocab_size), dtype=np.bool)
      y_batch = np.zeros((batch_size, vocab_size), dtype=np.bool)
      print(X_batch.shape)
      print(y_batch.shape)
      for i in range(batch_size):
          for t, w in enumerate(sequences[index % len(sequences)]):
              X_batch[i, t, vocab[w]] = 1
              y_batch[i, vocab[next_words[index % len(sequences)]]] = 1
              index = index + 1
         
          yield X_batch, y_batch

def lstm_model(seq_length,vocab_size):

    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='categorical_accuracy')]
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = [categorical_accuracy])

    return model

if __name__ == '__main__':

    seq_length = 30
    

    wl = loadandprocess()

    sequences , next_words , vocab_size , vocab = vocabcreate(wl)

    rnn_size = 256 # size of RNN
    seq_length = 30 # sequence length
    learning_rate = 0.000000001 #learning rate

    md = lstm_model(seq_length, vocab_size)
    md.summary()

    batch_size = 500 # minibatch size
    num_epochs = 50 # number of epochs

    callbacks=[EarlyStopping(patience=4, monitor='categorical_accuracy'),
               ModelCheckpoint(filepath=os.path.join(current_dir,'Model/weight_file.hdf5'),
                               monitor='categorical_accuracy', verbose=0, mode='auto', period=2)]
    
    #fit the model
    history = md.fit(generator(batch_size, sequences, seq_length, vocab_size, vocab, next_words), shuffle=True, epochs=num_epochs, callbacks=callbacks,steps_per_epoch=500)

    #save the model
    md.save(os.path.join(current_dir,'Model/weight_file.hdf5'))
    
