# %% [code] {"jupyter":{"outputs_hidden":true}}
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import cml.data_v1 as cmldata

CONNECTION_NAME = "se-sandbox-cdw-hive"
conn = cmldata.get_connection(CONNECTION_NAME)

## Sample Usage to get pandas data frame
EXAMPLE_SQL_QUERY = "select * from airlinedata.tweets"
dataset = conn.get_pandas_dataframe(EXAMPLE_SQL_QUERY)
print(dataset)

# dataset = pd.read_csv(r'tweets.csv')
# %% [code] {"jupyter":{"outputs_hidden":true}}
dataset.head()

# %% [markdown]
# **#return percentage of every columns missing value , cols which have >90% missing values then drop them**

# %% [code] {"jupyter":{"outputs_hidden":true}}
(len(dataset) - dataset.count())/len(dataset)

# %% [code] {"jupyter":{"outputs_hidden":true}}
dataset = dataset.drop(['tweets.airline_sentiment_gold','tweets.negativereason_gold','tweets.tweet_coord'],axis=1)

# %% [code] {"jupyter":{"outputs_hidden":true}}
dataset.head(3)

# %% [code] {"jupyter":{"outputs_hidden":true}}
mood_count=dataset['tweets.airline_sentiment'].value_counts()

# %% [code] {"jupyter":{"outputs_hidden":true}}
mood_count

# %% [code] {"jupyter":{"outputs_hidden":true}}
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# **Graphical representation of airline sentiment :-**

# %% [code] {"jupyter":{"outputs_hidden":true}}
sns.countplot(x='tweets.airline_sentiment',data=dataset,order=['negative','neutral','positive'])
plt.show()

# %% [markdown]
# **Graphical representation of airline sentiment with airlines:-** 

# %% [code] {"jupyter":{"outputs_hidden":true}}
sns.factorplot(x = 'tweets.airline_sentiment',data=dataset,
               order = ['negative','neutral','positive'],kind = 'count',col_wrap=3,col='tweets.airline',size=4,aspect=0.6,sharex=False,sharey=False)
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":true}}
dataset['tweets.negativereason'].value_counts()

# %% [markdown]
# **Graphical representation of negativereason towards airlines:-**

# %% [code] {"jupyter":{"outputs_hidden":true}}
sns.factorplot(x = 'tweets.airline',data = dataset,kind = 'count',hue='tweets.negativereason',size=12,aspect=.9)
plt.show()

# %% [markdown]
# both above and below graph are same and for show the negative comment reason on different airlines

# %% [code] {"jupyter":{"outputs_hidden":true}}
sns.factorplot(x = 'tweets.negativereason',data=dataset,kind='count',col='tweets.airline',size=9,aspect=.8,col_wrap=2,sharex=False,sharey=False)
plt.show()

# %% [markdown]
# **Data Cleaning and Preprocessing :-**

# %% [code] {"jupyter":{"outputs_hidden":true}}
import re
import nltk
import time

# %% [code] {"jupyter":{"outputs_hidden":true}}
start_time = time.time()
#remove words which are starts with @ symbols
dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:re.sub('@\w*','',str(x)))
#remove special characters except [a-zA-Z]
dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
#remove link starts with https
dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:re.sub('http.*','',str(x)))
end_time = time.time()

# %% [code] {"jupyter":{"outputs_hidden":true}}
#total time consume to filter data
end_time-start_time

# %% [code] {"jupyter":{"outputs_hidden":true}}
dataset['tweets.text'].head()

# %% [code] {"jupyter":{"outputs_hidden":true}}
dataset['tweets.text'] = dataset['tweets.text'].map(lambda x:str(x).lower())

# %% [code] {"jupyter":{"outputs_hidden":true}}
dataset['tweets.text'].head(2)

# %% [code] {"jupyter":{"outputs_hidden":true}}
from nltk.corpus import stopwords

# %% [code] {"jupyter":{"outputs_hidden":true}}
corpus = []

# %% [markdown]
# * Remove stopwords from comments 
# * Not used  PorterStemmer to make words pure

# %% [code] {"jupyter":{"outputs_hidden":true}}
none=dataset['tweets.text'].map(lambda x:corpus.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))])))                                     

# %% [code] {"jupyter":{"outputs_hidden":true}}
corpus[:4]

# %% [markdown]
# **Training Part :-**

# %% [code] {"jupyter":{"outputs_hidden":true}}
X = pd.DataFrame(data=corpus,columns=['tweets.comment_text'])

# %% [code] {"jupyter":{"outputs_hidden":true}}
X.head()

# %% [code] {"jupyter":{"outputs_hidden":true}}
y = dataset['tweets.airline_sentiment'].map({'neutral':1,'negative':-1,'positive':1})

# %% [code] {"jupyter":{"outputs_hidden":true}}
y.head(2)

# %% [markdown]
# Split data into Train and Test:-

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.model_selection import train_test_split

# %% [code] {"jupyter":{"outputs_hidden":true}}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# %% [code] {"jupyter":{"outputs_hidden":true}}
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# %% [markdown]
# Use TfidfVectorizer for feature extraction :-

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.feature_extraction.text import TfidfVectorizer

# %% [code] {"jupyter":{"outputs_hidden":true}}
vector = TfidfVectorizer(stop_words='english',sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1,1),max_features=30000)
#token_patten #2 for word length greater than 2>=

# %% [code] {"jupyter":{"outputs_hidden":true}}
X_train_word_feature = vector.fit_transform(X_train['tweets.comment_text']).toarray()

# %% [code] {"jupyter":{"outputs_hidden":true}}
X_test_word_feature = vector.transform(X_test['tweets.comment_text']).toarray()

# %% [code] {"jupyter":{"outputs_hidden":true}}
print(X_train_word_feature.shape,X_test_word_feature.shape)

# %% [markdown]
# Model Training :-

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.linear_model import LogisticRegression

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# %% [code] {"jupyter":{"outputs_hidden":true}}
classifier = LogisticRegression()

# %% [code] {"jupyter":{"outputs_hidden":true}}
classifier.fit(X_train_word_feature,y_train)

# %% [code] {"jupyter":{"outputs_hidden":true}}
y_pred = classifier.predict(X_test_word_feature)

# %% [code] {"jupyter":{"outputs_hidden":true}}
cm = confusion_matrix(y_test,y_pred)

# %% [code] {"jupyter":{"outputs_hidden":true}}
acc_score = accuracy_score(y_test,y_pred)

# %% [code] {"jupyter":{"outputs_hidden":true}}
print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)

# %% [code] {"jupyter":{"outputs_hidden":true}}
y_pred_prob = classifier.predict_proba(X_train_word_feature)

# %% [markdown]
# To determine probability of negative or positive comment :-

# %% [code] {"jupyter":{"outputs_hidden":true}}
y_pred_prob[:5]