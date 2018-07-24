# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords') # download stopwords (e.g. 'and', 'or', 'the', ...)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stopwords_keep = ['not', 'no'] # add the words you'd like to keep here
stopwords_drop = set(stopwords.words('english')) - set(stopwords_keep)
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # replace non-alphabet letters with space
    review = review.lower() # make text lowercase
    review = review.split() # split at spaces to get separate words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords_drop] # select only non-stopwords and take their stems
    review = ' '.join(review) # create text again from the list of words
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # keep only the 1500 most frequent words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)