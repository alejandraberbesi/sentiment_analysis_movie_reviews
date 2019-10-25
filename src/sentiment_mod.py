import numpy as np
import nltk
import random
import os
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
#nltk.download('wordnet')

# Context Manager to open movie reviews .txt files
with open("txt/positive.txt","rb") as obj:
    short_pos = str(obj.read()).lower()
with open("txt/negative.txt","rb") as obj:
    short_neg = str(obj.read()).lower()

allowed_word_types=["J"] #adjectives
stopw = set(stopwords.words('english'))
exclude=stopw.union(string.punctuation)
word_Lemmatized = WordNetLemmatizer()
documents=[]

def segment_words(document,typesent):
    document=document.replace("\\r","")
    for p in document.split("\\n"):
        reviews_words=[]
        p=p.replace("\\'","")  
        words=word_tokenize(p)
        pos_t=nltk.pos_tag(words)
        for w in pos_t:
            if w[1][0] in allowed_word_types: #first letter has to be J
                if w[0] not in exclude and w[0].isalpha():
                    reviews_words.append(word_Lemmatized.lemmatize(w[0]))
        documents.append((" ".join(reviews_words),typesent))
         
segment_words(short_pos,"positive")
segment_words(short_neg,"negative")

random.Random(4).shuffle(documents)
words, labels = map(list, zip(*documents)) 

X_train, X_test, y_train, y_test = train_test_split(words, labels, test_size=0.33, random_state=42)

Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(words)
X_train_Tfidf = Tfidf_vect.transform(X_train)
X_test_Tfidf = Tfidf_vect.transform(X_test)

#print(Tfidf_vect.vocabulary_)

# Classifier - Algorithm - SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_Tfidf,y_train)
predictions_SVM = SVM.predict(X_test_Tfidf)

# accuracy_score
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)

a=Tfidf_vect.transform(["amaizing wonderful brilliant","terrible disgusting bad"])
print(Encoder.inverse_transform(SVM.predict(a)))

