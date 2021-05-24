
########################## Imports #############################################
from flask import Flask, request, jsonify, render_template
import keras
from keras.models import Sequential, Model, load_model
import time
import codecs
import collections
from six.moves import cPickle
import json
import numpy as np

############################ Loaders ############################################

app = Flask('Sithara_Lyrics_Gen',template_folder='template')

model = load_model("weight_file.hdf5")
vocab_file =  "vocab.txt"

with open("vocab.txt", 'rb') as f:
       words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

######################### Functions ############################################

def sample(preds, temperature=20):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def modpred(seed):
    seq_length = 30
    words_number = 30 # number of words to generate
    seed_sentences = seed #seed sentence to start the generating.

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

    #then, we generate the text
    for i in range(words_number):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            x[0, t, vocab[word]] = 1.

        #calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 30)
        next_word = vocabulary_inv[next_index]
        #add the next word to the text
        generated += " " + next_word
        # shift the sentence by one, and add the next word at its end
        sentence = sentence[1:] + [next_word]
    return generated

# Predict function

@app.route('/predict', methods=['POST'])
def predict():
    w3 = []
    generated = ''
    seed = request.form['seed']
    generated_ini = modpred(seed)
    d1 = generated_ini.split(' ',-1)
    for x in d1:
        if x != "urukum":
            w3.append(x)
    for each_word in w3:
        generated = generated + " " + each_word

    return render_template('final.html',generated = generated)

# start function

@app.route('/',methods = ['GET'])
def ping():
    
    return render_template('initial.html')

# Main

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)
