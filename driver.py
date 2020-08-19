import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import csv
import sklearn
import string
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier as SGD


train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text



def imdb_data_preprocess(inpath, outpath, name, mix):
    neg = 'neg'
    pos = 'pos'
    row_number = 0
    stopWords = open(outpath + 'stopwords.en.txt', 'r', encoding='ISO-8859-1')
    stop = stopWords.read()
    stop = stop.split('\n')

    with open(name, 'w', newline='', encoding='ISO-8859-1') as csvfile:
        fieldnames = ['', 'text', 'polarity']
        writer = csv.writer(csvfile)
        writer.writerow(('', 'text', 'polarity'))

        posDir = os.listdir(inpath + pos)
        for txtName in posDir:
            inputTxt = open(inpath + pos + '/' + txtName, 'r', encoding='ISO-8859-1')
            text = inputTxt.read()
            text = re.sub('(<br\s*/?>)+', ' ', text)
            text = text.replace('_', '')
            text_ = text.split(' ')
            filt = [x for x in text_ if x not in stop]
            filtered = ' '.join(filt)
            writer.writerow((row_number, filtered, str(1)))

            inputTxt.close()
            row_number += 1


        negDir = os.listdir(inpath + neg)
        for txtName in negDir:
            inputTxt = open(inpath + neg + '/' + txtName, 'r', encoding='ISO-8859-1')
            text = inputTxt.read()
            text = re.sub('(<br\s*/?>)+', ' ', text)
            text = text.replace('_', '')
            text_ = text.split(' ')
            filt = [x for x in text_ if x not in stop]
            filtered = ' '.join(filt)
            writer.writerow((row_number, filtered, str(0)))

            inputTxt.close()
            row_number += 1
    csvfile.close()



    # pd.set_option('display.max_rows', None)

    df_te = pd.read_csv('C:/Users/vlad1/Desktop/AI/Week 11/Assignment 5/lib/publicdata/imdb_te.csv', encoding='ISO-8859-1')
    data = df_te.iloc[:, [0, 1]].values
    _, test_text = tuple(zip(*data))
    imdb_test = pd.DataFrame({'_': _, 'text': test_text})


    df = pd.read_csv(outpath + name, encoding='ISO-8859-1')
    data = df.iloc[:, [1, 2]].values
    train_text, train_label = tuple(zip(*data))

    imdb_train = pd.DataFrame({'text': train_text, 'label': train_label})
    y_train = imdb_train['label'].values


    # ################ Unigram ################
    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), encoding='ISO-8859-1', stop_words=stop, preprocessor=preprocess_text, token_pattern=r"(?u)\b\w[\w'|-]*\w\b")
    unigram_vectorizer.fit(imdb_train['text'].values)
    X_train_unigram = unigram_vectorizer.transform(imdb_train['text'].values)

    clfUni = SGD(loss='hinge', penalty='l1')
    # score = cross_val_score(clfUni, X_train_unigram, y_train, cv=5)
    clfUni.fit(X_train_unigram, y_train)
    X_test1 = unigram_vectorizer.transform(imdb_test['text'].values)

    unigram_output = open("unigram.output.txt", "w", encoding='ISO-8859-1')
    for i in X_test1:
        pred = clfUni.predict(i)
        unigram_output.write(str(''.join(map(str, pred))) + '\n')
    unigram_output.close()


    ################# Unigram Tf-Idf ################
    unigram_tf_idf_transformer = TfidfTransformer()
    unigram_tf_idf_transformer.fit(X_train_unigram)
    X_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_train_unigram)

    clfUniTf = SGD(loss='hinge', penalty='l1')
    clfUniTf.fit(X_train_unigram_tf_idf, y_train)
    X_test2 = unigram_tf_idf_transformer.transform(X_test1)

    unigram_tfidf = open("unigramtfidf.output.txt", "w", encoding='ISO-8859-1')
    for i in X_test2:
        pred = clfUniTf.predict(i)
        unigram_tfidf.write(str(''.join(map(str, pred))) + '\n')
    unigram_tfidf.close()



    # ################# Bigram ################
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), encoding='ISO-8859-1', stop_words=stop, preprocessor=preprocess_text, token_pattern=r"(?u)\b\w[\w'|-]*\w\b")
    bigram_vectorizer.fit(imdb_train['text'].values)
    X_train_bigram = bigram_vectorizer.transform(imdb_train['text'].values)

    clfBigram = SGD(loss='hinge', penalty='l1')
    clfBigram.fit(X_train_bigram, y_train)
    X_test3 = bigram_vectorizer.transform(imdb_test['text'].values)

    bigram_output = open("bigram.output.txt", "w", encoding='ISO-8859-1')
    for i in X_test3:
        pred = clfBigram.predict(i)
        bigram_output.write(str(''.join(map(str, pred))) + '\n')
    bigram_output.close()



    ################ Bigram tf-idf ################
    bigram_tf_idf_transformer = TfidfTransformer()
    bigram_tf_idf_transformer.fit(X_train_bigram)
    X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)

    clfBigramTF = SGD(loss='hinge', penalty='l1')
    clfBigramTF.fit(X_train_bigram_tf_idf, y_train)
    X_test4 = bigram_tf_idf_transformer.transform(X_test3)

    bigramtfidf_output = open("bigramtfidf.output.txt", "w", encoding='ISO-8859-1')
    for i in X_test4:
        pred = clfBigramTF.predict(i)
        bigramtfidf_output.write(str(''.join(map(str, pred))) + '\n')
    bigramtfidf_output.close()







if __name__ == "__main__":
    imdb_data_preprocess('C:/Users/vlad1/Desktop/AI/Week 11/Assignment 5/aclImdb/train/', 'C:/Users/vlad1/Desktop/AI/Week 11/Assignment 5/', 'imdb_tr.csv', False)

    # '''train a SGD classifier using unigram representation,
    # predict sentiments on imdb_te.csv, and write output to
    # unigram.output.txt'''
    #
    # '''train a SGD classifier using bigram representation,
    # predict sentiments on imdb_te.csv, and write output to
    # bigram.output.txt'''
    #
    #  '''train a SGD classifier using unigram representation
    #  with tf-idf, predict sentiments on imdb_te.csv, and write
    #  output to unigramtfidf.output.txt'''
    #
    #  '''train a SGD classifier using bigram representation
    #  with tf-idf, predict sentiments on imdb_te.csv, and write
    #  output to bigramtfidf.output.txt'''
