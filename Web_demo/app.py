from flask import Flask, render_template, request
import tensorflow as tf
from pyvi import ViTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import re
import os
import string

app = Flask(__name__)

# Load the pre-trained models
current_directory = os.getcwd()

model_LSTM_topic_path = os.path.join(current_directory, r'Web_demo\Model\model_LSTM_topic.h5')
model_LSTM_sentiment_path = os.path.join(current_directory, r'Web_demo\Model\model_LSTM_sentiment.h5')

w2v_model_path = os.path.join(current_directory, r'Web_demo\Model\model_w2v')

# Load models and preprocessors
model_topic_lstm = load_model(model_LSTM_topic_path)
model_sentiment_lstm = load_model(model_LSTM_sentiment_path)

w2v_model = tf.keras.models.load_model(w2v_model_path, compile=False)

punctuations=list(string.punctuation)
def normalize_text(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\d', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

def tokenizer(text):
    tokens = []
    for sent in sent_tokenize(text):
        words = word_tokenize(sent)
        tokens.extend(words)
    return tokens

def get_sentence_vector(sentence, model):
    vector_sum = np.zeros(model.vector_size)
    count = 0.
    for word in sentence:
        if word in model.wv:
            vector_sum += model.wv[word]
            count += 1.
    if count != 0:
        vector_sum /= count
    return vector_sum

def preprocess_and_convert(sentence):
    sentence = normalize_text(sentence)
    tokens = tokenizer(sentence)
    sentence_vector = get_sentence_vector(tokens, w2v_model)
    return sentence_vector

@app.route('/', methods=['GET', 'POST'])
def index():
    topic_result_lstm = None
    sentiment_result_lstm = None

    if request.method == 'POST':
        user_input = request.form['user_input']
        if user_input:
            embedding_dim = w2v_model.vector_size
            # Preprocess and convert sentence to vector
            sentence_vector = preprocess_and_convert(user_input)
            sentence_vector = sentence_vector.reshape(-1, 1, embedding_dim)

            # Predict topic and sentiment with LSTM model
            topic_result_lstm = model_topic_lstm.predict(sentence_vector)
            sentiment_result_lstm = model_sentiment_lstm.predict(sentence_vector)

    return render_template('demo.html', 
                           topic_result_lstm=topic_result_lstm, sentiment_result_lstm=sentiment_result_lstm)

if __name__ == '__main__':
    app.run(debug=True)
