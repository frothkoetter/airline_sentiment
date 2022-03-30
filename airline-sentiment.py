# %% [code] {"jupyter":{"outputs_hidden":true}}
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import cml.data_v1 as cmldata

CONNECTION_NAME = "se-sandbox-cdw-hive"
conn = cmldata.get_connection(CONNECTION_NAME)

EXAMPLE_SQL_QUERY = "select * from airlinedata.tweets"
dataset = conn.get_pandas_dataframe(EXAMPLE_SQL_QUERY)
print(dataset)

dataset.head()

# %% [markdown]
# **#return percentage of every columns missing value , cols which have >90% missing values then drop them**

(len(dataset) - dataset.count())/len(dataset)

dataset = dataset.drop(['tweets.airline_sentiment_gold','tweets.negativereason_gold','tweets.tweet_coord'],axis=1)

dataset.head(3)

mood_count=dataset['tweets.airline_sentiment'].value_counts()

mood_count

import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# **Graphical representation of airline sentiment :-**

sns.countplot(x='tweets.airline_sentiment',data=dataset,order=['negative','neutral','positive'])
plt.show()

# %% [markdown]
# **Graphical representation of airline sentiment with airlines:-** 

sns.factorplot(x = 'tweets.airline_sentiment',data=dataset,
               order = ['negative','neutral','positive'],kind = 'count',col_wrap=3,col='tweets.airline',size=4,aspect=0.6,sharex=False,sharey=False)
plt.show()

dataset['tweets.negativereason'].value_counts()

# %% [markdown]
# **Graphical representation of negativereason towards airlines:-**

sns.factorplot(x = 'tweets.airline',data = dataset,kind = 'count',hue='tweets.negativereason',size=12,aspect=.9)
plt.show()

# %% [markdown]
# both above and below graph are same and for show the negative comment reason on different airlines

sns.factorplot(x = 'tweets.negativereason',data=dataset,kind='count',col='tweets.airline',size=9,aspect=.8,col_wrap=2,sharex=False,sharey=False)
plt.show()

# %% [markdown]
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

dataset['tweets.text'].head()

dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:str(x).lower())

dataset['tweets.text'].head(2)

from nltk.corpus import stopwords

corpus = []

# %% [markdown]
# * Remove stopwords from comments 
# * Not used  PorterStemmer to make words pure

none=dataset['tweets.text'].map(lambda x:corpus.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))])))                                     

corpus[:4]

# %% [markdown]
# **Training Part :-**

X = pd.DataFrame(data=corpus,columns=['tweets.comment_text'])

X.head()

y = dataset['tweets.airline_sentiment'].map({'neutral':1,'negative':-1,'positive':1})

y.head(2)

# %% [markdown]
# Split data into Train and Test:-

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# %% [markdown]
# Use TfidfVectorizer for feature extraction :-

from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(stop_words='english',sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1,1),max_features=30000)
#token_patten #2 for word length greater than 2>=

X_train_word_feature = vector.fit_transform(X_train['tweets.comment_text']).toarray()

X_test_word_feature = vector.transform(X_test['tweets.comment_text']).toarray()

print(X_train_word_feature.shape,X_test_word_feature.shape)

# %% [markdown]
# Model Training :-

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

classifier = LogisticRegression()

classifier.fit(X_train_word_feature,y_train)

y_pred = classifier.predict(X_test_word_feature)

cm = confusion_matrix(y_test,y_pred)

acc_score = accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)

y_pred_prob = classifier.predict_proba(X_train_word_feature)

# %% [markdown]
# To determine probability of negative or positive comment :-

y_pred_prob[:5]