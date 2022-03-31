# Fit a simple linear regression model to the
# classic iris flower dataset to predict petal
# width from petal length. Write the fitted
# model to the file model.pkl

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import cdsw

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import cml.data_v1 as cmldata

CONNECTION_NAME = "se-sandbox-cdw-hive"
conn = cmldata.get_connection(CONNECTION_NAME)

EXAMPLE_SQL_QUERY = "select * from airlinedata.tweets where text is not null"
dataset = conn.get_pandas_dataframe(EXAMPLE_SQL_QUERY)

# **#return percentage of every columns missing value , cols which have >90% missing values then drop them**

(len(dataset) - dataset.count())/len(dataset)

dataset = dataset.drop(['tweets.airline_sentiment_gold','tweets.negativereason_gold','tweets.tweet_coord'],axis=1)

mood_count=dataset['tweets.airline_sentiment'].value_counts()

mood_count

import seaborn as sns
import matplotlib.pyplot as plt

dataset['tweets.negativereason'].value_counts()

# **Data Cleaning and Preprocessing :-**

import re
import nltk
import time

nltk.download('stopwords')

start_time = time.time()
#remove words which are starts with @ symbols
dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:re.sub('@\w*','',str(x)))
#remove special characters except [a-zA-Z]
dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
#remove link starts with https
dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:re.sub('http.*','',str(x)))
end_time = time.time()

#total time consume to filter data
end_time-start_time

dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:str(x).lower())

from nltk.corpus import stopwords

corpus = []

# * Remove stopwords from comments 
# * Not used  PorterStemmer to make words pure

none=dataset['tweets.text'].map(lambda x:corpus.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))])))                                     

corpus[:4]

# %% [markdown]
# **Training Part :-**

X = pd.DataFrame(data=corpus,columns=['tweets.comment_text'])

y = dataset['tweets.airline_sentiment'].map({'neutral':1,'negative':-1,'positive':1})

# Split data into Train and Test:-

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Use TfidfVectorizer for feature extraction :-

from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(stop_words='english',sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1,1),max_features=30000)

#token_patten #2 for word length greater than 2>=

X_train_word_feature = vector.fit_transform(X_train['tweets.comment_text']).toarray()

X_test_word_feature = vector.transform(X_test['tweets.comment_text']).toarray()

# Model Training :-

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

classifier = LogisticRegression()

classifier.fit(X_train_word_feature,y_train)

y_pred = classifier.predict(X_test_word_feature)

cm = confusion_matrix(y_test,y_pred)

acc_score = accuracy_score(y_test,y_pred)

y_pred_prob = classifier.predict_proba(X_train_word_feature)

# To determine probability of negative or positive comment :-

y_pred_prob[:5]#


# Mean squared error
#mean_sq = mean_squared_error(score_y, classifier)
#cdsw.track_metric("mean_sq_err", mean_sq)
#print("Mean squared error: %.2f"% mean_sq)

# Explained variance
#r2 = r2_score(score_y, predictions)
#cdsw.track_metric("r2", r2)
#print('Variance score: %.2f' % r2)

# Output
filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
cdsw.track_file(filename)