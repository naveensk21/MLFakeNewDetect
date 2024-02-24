import errno
from keras.models import load_model
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle


from flask import Flask, request, render_template

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

# load model
rf_loaded_model = pickle.load(open('model/rf_model.sav', 'rb'))
svm_loaded_model = pickle.load(open('model/svm_model.sav', 'rb'))
lstm_loaded_model = load_model('model/lstm_best_model.epoch08-loss0.07.hdf5')

# load vectorizer
rf_loaded_vectorizer = pickle.load(open("model/rf_vectorizer.pickle", "rb"))
svm_loaded_vectorizer = pickle.load(open('model/svm_vectorizer.pickle', 'rb'))
with open('model/lstm_tokenizer.pickle', 'rb') as handle:
    lstm_loaded_tokenizer = pickle.load(handle)

precision_scores = {'ML_RF_precision': '94%', 'ML_SVM_precision': '96%', 'DL_LSTM_precision': '97%'}


def dl_model_pred(text, model, tokenizer):
    max_len=1000

    seq = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = model.predict(padded_text)
    if pred > 0.7:
        return 1
    else:
        return 0


def model_predict(requested_text, vect, model):
    for key, value in requested_text.items():
        X = vect.transform([value])
        y_predict = model.predict(X)
        return y_predict


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    pred = []
    text = request.form
    text_value = list(text.values())[0]

    model_select = request.form.get('modelSelect')

    if model_select == 'ML_RF_Model':
        predicted = model_predict(text, rf_loaded_vectorizer, rf_loaded_model)
        pred.append(predicted)
    elif model_select == 'ML_SVM_Model':
        predicted = model_predict(text, svm_loaded_vectorizer, svm_loaded_model)
        pred.append(predicted)
    elif model_select == 'DL_LSTM_Model':
        predicted = dl_model_pred(text_value, lstm_loaded_model, lstm_loaded_tokenizer)
        pred.append(predicted)
    print(pred)

    if 1 in pred:
        result = 'Real'
    elif 0 in pred:
        result = 'Fake'
    else:
        result = 'Error'

    return render_template('predictions.html', result=result, text=text, text_value=text_value, model_select=model_select,
                           precision_scores=precision_scores)


if __name__ == '__main__':
    app.debug = True
    app.run()