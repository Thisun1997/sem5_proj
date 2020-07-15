from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
import pickle as p
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import zipfile
import bz2
from waitress import serve
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import load_model
import tensorflow as tf

graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)
tokenizer_rnn_file = open('model_14/tokenizer_18.pickle','rb')
tokenizer_rnn = p.load(tokenizer_rnn_file)
tokenizer_rnn_file.close()
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words.extend(['news', 'report', 'affect' ,'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year', 'people', 'heavy', 'government', 'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people','fuck','sex','good'])


def preprocess(msg):
    s = re.sub(r'http[s]?[ ]?:[ ]?//\S+', '', msg.lower())
    s = re.sub(r'[^a-zA-Z ]', '', s)
    s = tokenizer.tokenize(s)
    s = [w for w in s if w not in stop_words]
    s = [stemmer.stem(i) for i in s]
    s = [lemmatizer.lemmatize(i) for i in s]
    s = [' '.join(s)]
    list_tokenized_sample = tokenizer_rnn.texts_to_sequences(s)
    X_sample = pad_sequences(list_tokenized_sample, maxlen=50)
    return X_sample

@app.route('/')
def home():
    return render_template('home.html',len = 0, message = None, prediction = None)

@app.route('/predictTokens',methods=['POST'])
def predictTokens():
    if request.method == 'POST':
        message1 = request.get_json()['message']
        message =request.get_json()['message']
        message = preprocess(message)
        global graph
        with graph.as_default():
            rnn_file = 'model_14/RNN_18cat_1_20.pickle'
            rnn = p.load(open(rnn_file,'rb'))
            result = rnn.predict(message)
        category_names = ['medical_and_other_aids',
                         'army_and_police',
                         'needs',
                         'disasters',
                         'infrastructure',
                         'search_and_rescue',
                         'child_alone',
                         'shelter',
                         'clothing',
                         'money',
                         'missing_people',
                         'refugees',
                         'death',
                         'floods',
                         'storm',
                         'fire',
                         'earthquake',
                         'cold']
        l = []
        res = sorted(range(len(result[0])), key = lambda sub: result[0][sub])[-5:]
        for i in res:
          if result[0][i] > 0.3:
            l.append(category_names[i])
    return jsonify(
        message = message1,
        tokens = l
    )

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8000)