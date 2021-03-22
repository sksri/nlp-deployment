# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:52:35 2021

@author: S7012205
"""

import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import pickle


sms = pd.read_csv('SMSSpamCollection', sep='\t', names=('Label','Messages'))


# Cleaning text
corpus = []
for i in range(len(sms)):
    review = re.sub('[^a-zA-z]',' ', sms['Messages'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    

# Creating TF-IDF model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus)

y_label = pd.get_dummies(sms['Label'])
y = y_label.iloc[:,1].values

pickle.dump(cv, open('spam_transform.pkl','wb'))


# Splitting into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=101)

# build classifier model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
model = nb.fit(X_train, y_train)

# Creating pickle file
pickle.dump(model, open('spam_classifier.pkl','wb'))

y_pred = model.predict(X_test)

# Evaluating model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
confusion_m = confusion_matrix(y_test,y_pred)
accuracy_s = accuracy_score(y_test,y_pred)
classification_r = classification_report(y_test,y_pred)


