import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('cardio_train_cleaned.csv', sep=';')

X = df.drop(['cardio', 'id'],axis=1).values
y = df['cardio'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
clf = LogisticRegression(random_state=0, max_iter=7600).fit(X_train, y_train)

score = clf.score(X_test, y_test)

print("Score:")
print(score)
