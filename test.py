import string
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
import numpy as np

def processTestData(line, stop):
    line = line.translate(str.maketrans('', '', string.punctuation))
    line = re.sub('(<br\s*/?>)+', ' ', line)
    line = re.sub(r'\d+', '', line)
    return line

fin = open('C:/Users/vlad1/Desktop/AI/Week 11/Assignment 5/aclImdb/train/neg/4328_3.txt', 'r', encoding='ISO-8859-1')

stopWords = open('C:/Users/vlad1/Desktop/AI/Week 11/Assignment 5/stopwords.en.txt', 'r', encoding='ISO-8859-1')

stop = stopWords.read()
text = fin.read()#.lower()

# a = text.translate(str.maketrans(' ', ' ', string.punctuation))
# a = text.split(' ')
stop = stop.split('\n')

df = pd.read_csv('C:/Users/vlad1/Desktop/AI/Week 11/Assignment 5/lib/publicdata/imdb_te.csv', encoding='ISO-8859-1')
train_data = df['text'].astype(str)

for text in train_data:
    text_ = processTestData(text, stop)
    print(text_)
    break



# for line in train_data:
#     print(processTestData(str(line), stop))
#     break


# preprocessor=preprocess_text,