import pandas as pd

df = pd.read_csv('cardio_train_cleaned.csv', sep=';', nrows=1000)

print(df.describe())
df.plot.scatter(x="ap_hi", y="cardio")
