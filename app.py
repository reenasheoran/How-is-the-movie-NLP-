from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

Emoji_Folder = os.path.join('static', 'emoji_images')

app = Flask(__name__)

app.config['IMG_FOLDER'] = Emoji_Folder


model = load_model('LSTM_model.h5')
    


@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("home.html")

@app.route('/predict', methods = ['POST', "GET"])
def sentimentprediction():
    if request.method=='POST':
        text = request.form['text']
        Sentiment = ''
        review_length = 1383
        word_to_id = imdb.get_word_index()
        remove_specialchars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(remove_specialchars, "", text.lower())

        words = text.split() #split string into a list
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=review_length) 
        vector = np.array([x_test.flatten()])
    
        prob = model.predict_proba(np.asarray([vector][0]))[0][0]
        print(prob)
        class1 = model.predict_classes(np.asarray([vector][0]))[0][0]
        if class1 == 0:
            Sentiment = 'Negative'
            img_filename = os.path.join(app.config['IMG_FOLDER'], 'sad.jpg')
        else:
            Sentiment = 'Positive'
            img_filename = os.path.join(app.config['IMG_FOLDER'], 'happy.jpg')
    return render_template('home.html', text=text, sentiment=Sentiment, probability=prob, image=img_filename)


if __name__ == "__main__":
    app.run()

