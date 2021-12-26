import pandas as pd
import seaborn as sns

df = pd.read_csv('cardio_train_cleaned.csv', sep=';', nrows=10000)

print(df.columns.values)
for colname in df.columns.values[1:-1]:
    sns.lmplot(x=colname,y='cardio' ,data=df, fit_reg=True) 
