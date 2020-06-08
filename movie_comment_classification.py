from nltk.corpus import stopwords
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import nltk
re1 = re.compile('[/(){}\[\]\|@,;]')
re2 = re.compile('[^0-9a-z #+_]')
re3 = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re1.sub(" ", text)
    text = re2.sub(" ", text)
    text = " ".join(word for word in text.split() if word not in re3)
                    #if word not in re3)
    return text

df = pd.read_csv("/IMDB_Dataset.csv")
df['review'] = df['review'].apply(clean_text)
df = df.sort_values("sentiment", ascending=True)
X = df["review"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 1)

#Naive Bayes
nbmodel = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
nbmodel.fit(X_train, y_train)
y_pred = nbmodel.predict(X_test)
accuracyNB = accuracy_score(y_pred, y_test)
print(accuracyNB)

#SVM
sgdmodel = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgdmodel.fit(X_train, y_train)
y_pred = sgdmodel.predict(X_test)
accuracy_sgd = accuracy_score(y_pred, y_test)
print(accuracy_sgd)

#logistic Regression
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=100, max_iter = 500)),
               ])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy_log = accuracy_score(y_pred, y_test)
print(accuracy_log)
