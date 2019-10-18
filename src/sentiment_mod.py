import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify import ClassifierI
from statistics import mode
import string
from nltk.corpus import stopwords
import os


# Context Manager for movie reviews
with open("txt/positive.txt","rb") as obj:
    short_pos = str(obj.read())
with open("txt/negative.txt","rb") as obj:
    short_neg = str(obj.read())

short_pos=short_pos.replace('\\r','')
short_neg=short_neg.replace('\\r','')

allowed_word_types=["J"] #adjectives
reviews_words=[]
stopwords=stopwords.words('english')
stopw = set(stopwords)
exclude=stopw.union(string.punctuation)
documents=[]

def segment_words(document,typesent):
    for p in document.split('\\n'):
        documents.append((p,typesent))   
        words=word_tokenize(p)
        pos_t=nltk.pos_tag(words)
        for w in pos_t:
            if w[1][0] in allowed_word_types: #first letter has to J,R,V
                if w[0] not in exclude and w[0].isalpha():
                    reviews_words.append(w[0].lower())  
         
            
segment_words(short_pos,"positive")
segment_words(short_neg,"negative")
random.shuffle(documents)

#to train against a certain number of words:
reviews_words = nltk.FreqDist(reviews_words)
word_features=list(reviews_words.keys())[:500]
def find_features(document):
    words=set(document)
    features={}
    for w in word_features: 
        features[w]=(w in words)  
    return features

featuresets= [(find_features(rev), category) for (rev, category) in documents]

training_set=featuresets[:8000]
testing_set=featuresets[8000:]

classifier=nltk.NaiveBayesClassifier.train(training_set)
MNBclassifier=SklearnClassifier(MultinomialNB())
MNBclassifier.train(training_set)
Bernoulli_classifier=SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)

#combining algorithms with a vote:
class voting_process(ClassifierI):
    
    def __init__(self,*classifiers): 
        self._classifiers=classifiers    
    
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        
        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)    
        return conf     

voted_classifier=voting_process(classifier,MNBclassifier,Bernoulli_classifier)

def sentiment(text):
    feats=find_features(text)
    return voted_classifier.classify(feats),(voted_classifier.confidence(feats)*100)

