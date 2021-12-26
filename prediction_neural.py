import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from keras.metrics import FalseNegatives

df = pd.read_csv('cardio_train_cleaned.csv', sep=';')

scaler = MinMaxScaler()

X = df.drop(['cardio', 'id'],axis=1).values
y = df['cardio'].values

scaler.fit(X)
print(scaler.data_max_)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(30, input_dim=11, activation='tanh'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sample_weight = (1 + y_train)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', FalseNegatives()])
model.fit(X_train, y_train, epochs=10, verbose=1, sample_weight=sample_weight)

model.summary()
score = model.evaluate(X_test, y_test, verbose=0)

print(score)
print('Accuracy ',score[1])

predicted = model.predict(X_test)
predicted = (predicted > 0.5).astype(np.int_)
print(predicted)
print(confusion_matrix(y_test, predicted))