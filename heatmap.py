import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

sns.set(style="ticks")

df = pd.read_csv('cardio_train_cleaned.csv', sep=';', nrows=1000)

print(df.describe())
df.plot.scatter(x="ap_hi", y="cardio")

heat_map = sns.heatmap(df.corr(method='pearson'), annot=True,
fmt='.2f', linewidths=2)
heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45);
plt.rcParams["figure.figsize"] = (50,50)