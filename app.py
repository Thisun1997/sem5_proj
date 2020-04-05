from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle as p
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import zipfile
import bz2

## Definitions
# def remove_pattern(input_txt,pattern):
#     r = re.findall(pattern,input_txt)
#     for i in r:
#         input_txt = re.sub(i,'',input_txt)
#     return input_txt
# def count_punct(text):
#     count = sum([1 for char in text if char in string.punctuation])
#     return round(count/(len(text) - text.count(" ")),3)*100


app = Flask(__name__)


# data = pd.read_csv("sentiment.tsv",sep = '\t')
# data.columns = ["label","body_text"]
# # Features and Labels
# data['label'] = data['label'].map({'pos': 0, 'neg': 1})
# data['tidy_tweet'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
# tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())
# stemmer = PorterStemmer()
# tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
# for i in range(len(tokenized_tweet)):
#     tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
# data['tidy_tweet'] = tokenized_tweet
# data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
# data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
# X = data['tidy_tweet']
# y = data['label']
# print(type(X))
# # Extract Feature With CountVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(X) # Fit the Data
# X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)
# from sklearn.model_selection import train_test_split
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# ## Using Classifier
# clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='warn', n_jobs=None, penalty='l2',
#                    random_state=None, solver='warn', tol=0.0001, verbose=0,
#                    warm_start=False)
# clf.fit(X,y)


# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         message = request.form['message']
#         data = [message]
#         vect = pd.DataFrame(cv.transform(data).toarray())
#         body_len = pd.DataFrame([len(data) - data.count(" ")])
#         punct = pd.DataFrame([count_punct(data)])
#         total_data = pd.concat([body_len,punct,vect],axis = 1)
#         my_prediction = clf.predict(total_data)
#     return render_template('result.html',prediction = my_prediction)


# if __name__ == '__main__':
#     app.run(debug = True)


def preprocess(msg):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    stop_words.extend(['news', 'report', 'affect' ,'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year', 'people', 'heavy', 'government', 'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people','fuck','sex','good'])

    s = re.sub(r'http[s]?[ ]?:[ ]?//\S+', '', msg.lower())
    s = re.sub(r'[^a-zA-Z ]', '', s)
    s = tokenizer.tokenize(s)
    s = [w for w in s if w not in stop_words]
    s = [stemmer.stem(i) for i in s]
    s = [lemmatizer.lemmatize(i) for i in s]
    s = [' '.join(s)]
    return s

@app.route('/')
def home():
    return render_template('home.html',len = 0, message = None, prediction = None)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message1 = request.form['message']
        message = request.form['message']
        message = preprocess(message)
        data = tv.transform(message)
        print(data.data)
        if len(data.data) != 0 :
            my_prediction = clf.predict(data)
            categories = ['related',
                    'request',
                    'offer',
                    'medical_and_other_aids',
                    'search_and_rescue',
                    'army_and_police',
                    'child_alone',
                    'needs',
                    'missing_people',
                    'refugees',
                    'death',
                    'disasters',
                    'fire',
                    'direct_report']
            l = []
            for i in range (len(my_prediction[0])):
                if(my_prediction[0][i] == 1):
                    l.append(categories[i])
            print(l)
        else:
            l = []
            prediction = None
       
    return render_template('home.html', len = len(l), message = message1, prediction = l)


if __name__ == '__main__':
    clffile = bz2.BZ2File('model_14\smallerfile', 'rb')
  #  clffile = 'model_14/clf_RF_14.pickle'
    tvfile = 'E:/tfidf_RF_14.pickle'
    clf = p.load(clffile)
    tv = p.load(open(tvfile, 'rb'))
    app.run(debug = True)