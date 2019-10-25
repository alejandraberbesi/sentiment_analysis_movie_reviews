# sentiment_analysis_movie_reviews

Based on https://www.kaggle.com/iashreya/moviereviewsdataset

Script to evaluate reviews about movies, classifying them with positive or negative polarities.

There are two aproaches: 
On "master" branch, a model with SVM that involves a LabelEncoder and TfidfVectorizer is used. The inputs words are adjectives with a previous lemmatization process.
On "naive_bayes" branch, a simple model with a class that returns the result of a voting process between Naive Bayes, Multinomial Naive Bayes and Bernoulli Naive Bayes classifiers. The input words are adjectives.

Libraries: Nltk, Sklearn.
