import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cardio_train_cleaned.csv', sep=';', nrows=10000)

ap_hi = df["ap_hi"].to_numpy()
cardio = df["cardio"].to_numpy()

sumX = 0
sumX2 = 0
sumY = 0
sumXY = 0

for ap, car in zip(ap_hi, cardio):
	sumX = sumX + ap
	sumX2 = sumX2 + ap*ap
	sumY = sumY + car
	sumXY = sumXY + ap*car

n = len(ap_hi)
b = (n * sumXY - sumX * sumY)/(n*sumX2 - sumX * sumX)
a = (sumY - b*sumX)/n

print(a)
print(b)

x = np.linspace(min(ap_hi),max(ap_hi),100)
y = b*x+a

plt.plot(x, y, '-r', label='y=2x+1')
plt.plot(ap_hi, cardio, 'ro')
plt.show()

sns.lmplot(x='ap_hi',y='cardio' ,data=df, fit_reg=True) 
plt.show()