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

seq_length = 30

def sample(preds, temperature=1.0):

    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

def sengen(vocabulary_inv,vocab_size,words,vocab,model):

    words_number = 30 # number of words to generate
    seed_sentences = input()  # seed sentence to start the generating.
    seq_length = 30

    #initiate sentences
    generated = ''
    sentence = []

    #we shate the seed accordingly to the neural netwrok needs:
    for i in range (seq_length):
        sentence.append("urukum")

    seed = seed_sentences.split()

    for i in range(len(seed)):
        sentence[seq_length-i-1]=seed[len(seed)-i-1]

    generated += ' '.join(sentence)

    #the, we generate the text
    for i in range(words_number):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            x[0, t, vocab[word]] = 1

        #calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 30)
        next_word = vocabulary_inv[next_index]
        #print(preds)
        #add the next word to the text
        generated += " " + next_word
        # shift the sentence by one, and and the next word at its end
        sentence = sentence[1:] + [next_word]

    return generated

if __name__ == "__main__":

    #load vocabulary
    print("loading vocabulary...")

    w3 = []
    generated = ''
    
    with open(os.path.join(current_dir,"vocab.txt"), 'rb') as f:
            words, vocab, vocabulary_inv = cPickle.load(f)

    vocab_size = len(words)

    from keras.models import load_model
    # load the model
    print("loading model...")
    model = load_model(os.path.join(current_dir,'Model/weight_file.hdf5'))

    generated_ini = sengen(vocabulary_inv,vocab_size,words,vocab,model)

    d1 = generated_ini.split(' ',-1)

    for x in d1:
        if x != "urukum":
            w3.append(x)
    for each_word in w3:
        generated = generated + " " + each_word

    print("\n",generated,"\n")




