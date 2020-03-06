# sentiment_analysis_movie_reviews

Based on https://www.kaggle.com/iashreya/moviereviewsdataset

Script to evaluate reviews about movies, classifying them with positive or negative polarities.

There are two aproaches: 
**Master branch**: a model with a SVM that involves a LabelEncoder and TfidfVectorizer. The inputs words are adjectives with a previous lemmatization process.

**Naive_bayes branch**: a model with a class that returns the result of a voting process between Naive Bayes, Multinomial Naive Bayes and Bernoulli Naive Bayes classifiers. The input words are adjectives.
