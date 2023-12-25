import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

#loading the dat from csv file to a pandas dataframe
data = pd.read_csv('mail_data.csv')

data.head()

#replace the null values with a null string
mail_data = data.where(pd.notnull(data), '')

mail_data.shape

#label spam mail as 0; ham mail as 1;
mail_data.loc[mail_data['Category']=='spam', 'Category',]=0
mail_data.loc[mail_data['Category']=='ham', 'Category',]=1

#seprating text and label
X= mail_data['Message']
y= mail_data['Category']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=3)

X.shape, X_train.shape, X_test.shape

 #transform the text data to feature vectors that can be used as input to the Logistic reression

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase = True)

X_train_features= feature_extraction.fit_transform(X_train)
X_test_features= feature_extraction.transform(X_test)


 #convert y_train and y_test values as integers
y_train = y_train.astype('int')
y_test= y_test.astype('int')


lr= LogisticRegression()


lr.fit(X_train_features, y_train)

with open('spam_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(feature_extraction, vectorizer_file)
