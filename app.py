from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer


import pandas as pd 
import numpy as np
import re
import string



def count_punct(txt):
    count = sum([1 for char in txt if char in string.punctuation])
    return round(count/(len(txt) - txt.count(" ")),3)*100

def pattern_remove(inputT,pattern):
    all = re.findall(pattern,inputT)
    for i in all:
        inputT = re.sub(i,'',inputT)
    return inputT

app = Flask(__name__)


data = pd.read_csv("sentiment.tsv",sep = '\t')
data.columns = ["label","body_text"]


data['label'] = data['label'].map({'pos': 0, 'neg': 1})
data['clean_txt'] = np.vectorize(pattern_remove)(data['body_text'],"@[\w]*")
tokenized_txt = data['clean_txt'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_txt = tokenized_txt.apply(lambda x: [stemmer.stem(i) for i in x]) 
for i in range(len(tokenized_txt)):
    tokenized_txt[i] = ' '.join(tokenized_txt[i])
data['clean_txt'] = tokenized_txt
data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['ponctuation'] = data['body_text'].apply(lambda x:count_punct(x))
X = data['clean_txt']
y = data['label']


cv = CountVectorizer()
X = cv.fit_transform(X) 
X = pd.concat([data['body_len'],data['ponctuation'],pd.DataFrame(X.toarray())],axis = 1)


clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
clf.fit(X,y)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len,punct,vect],axis = 1)
        my_prediction = clf.predict(total_data)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000)
