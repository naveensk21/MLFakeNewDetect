import re
import string

import pandas as pd
import seaborn as sns
from matplotlib.pylab import plt

from flask import Flask, request, render_template

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from sklearn import metrics
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence

import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.stem import WordNetLemmatizer

# load model
loaded_model = pickle.load(open('model/rf_model.sav', 'rb'))

# load vectorizer
loaded_vectorizer = pickle.load(open("model/vectorizer.pickle", "rb"))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    pred = []
    text = request.form
    print(text.to_dict())
    for key, value in text.items():
        X = loaded_vectorizer.transform([value])
        y_predict = loaded_model.predict(X)
        pred.append(y_predict)

    if 1 in pred:
        result = 'Real'
    elif 0 in pred:
        result = 'Fake'
    else:
        result = 'Error'

    return render_template('predictions.html', result=result, text=text)

if __name__ == '__main__':
    app.debug = True
    app.run()