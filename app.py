# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:36:53 2021

@author: S7012205
"""

from flask import Flask, request, render_template
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle

cv = pickle.load(open('spam_transform.pkl','rb'))
clf = pickle.load(open('spam_classifier.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method=='POST':
        message = request.form['message']
        data = [message]
        my_vector = cv.transform(data).toarray()
        my_prediction = clf.predict(my_vector)
    return render_template('result.html', prediction=my_prediction)


if __name__=='__main__':
    app.run()